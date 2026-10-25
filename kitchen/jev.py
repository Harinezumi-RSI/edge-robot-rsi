"""Qwen を Jev(速い判定器)に改造する: Qwen の途中の層の状態に、10 個の目標それぞれの判定の頭(ロジスティック回帰)を付ける。

画像 1 枚を 1 回読むだけで、10 個の目標の達成確率が一度に出る(問いを 10 回投げる方式より 10 倍速い)。
正解はシミュレータの目標判定から自動で付く(collect_judge_data.py)。.venv-mlx で実行。

  python kitchen/jev.py extract data/judge            # 特徴を取り出す(途中から再開できる)
  python kitchen/jev.py train data/judge              # 初期状態で分けた交差検証 → 頭を models/jev_heads.npz に保存
  python kitchen/jev.py predict <画像>                # 10 個の目標の確率
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kitchen import brain  # noqa: E402
from kitchen.skills import CHECK_Q  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADS = os.path.join(ROOT, "models", "jev_heads.npz")
SYS = "あなたは家庭用ロボットの確認係です。"
ASK = "この台所の状態を確認してください。"
N_GOALS = len(CHECK_Q)


def _layers(model):
    n = len(model.language_model.model.layers)
    return [n // 4, n // 2, 3 * n // 4]  # 4B は 32 層 → 8, 16, 24 層目(+ 最終層)


def features(img_path, upto=None):
    """[特徴の種類, 2560]: 各層(8/16/24/最終)の最後のトークンの状態と、画像トークンの平均。
    upto=L: 改造(途中で打ち切り)。L 層目までだけ計算し、語彙への出力(logits)も計算しない。返すのは L 層目の 2 種類だけ。"""
    import mlx.core as mx
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs
    model, processor = brain._load()
    if upto is not None:
        return _features_upto(model, processor, img_path, upto)
    cfg = model.config
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": ASK}]
    prompt = apply_chat_template(processor, cfg, msgs, num_images=1, enable_thinking=False)
    img_tok = getattr(cfg, "image_token_index", None) or getattr(cfg, "image_token_id", None)
    inp = prepare_inputs(processor, images=[img_path], prompts=prompt, image_token_index=img_tok)
    kw = {k: v for k, v in inp.items() if k not in ("input_ids", "pixel_values", "attention_mask")}
    out = model(inp["input_ids"], pixel_values=inp["pixel_values"], mask=inp.get("attention_mask"),
                capture_layer_ids=_layers(model), return_hidden=True, **kw)
    ids = inp["input_ids"][0]
    is_img = (ids == img_tok).astype(mx.float32)
    feats = []
    for h in out.hidden_states:
        h = h[0].astype(mx.float32)
        feats.append(h[-1])
        feats.append((h * is_img[:, None]).sum(0) / mx.maximum(is_img.sum(), 1))
    f = mx.stack(feats)
    mx.eval(f)
    return np.array(f, dtype=np.float32)


FEAT_NAMES = [f"{w}_{p}" for w in ("L8", "L16", "L24", "最終") for p in ("末尾", "画像平均")]


def _features_upto(model, processor, img_path, upto):
    """L 層目までで打ち切る(Gromov ら 2025: 深い層は抜いても保たれる/本作品: 判定には 8 層目で足りる)。"""
    import mlx.core as mx
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs
    cfg = model.config
    lm = model.language_model.model
    full = lm.layers
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": ASK}]
    prompt = apply_chat_template(processor, cfg, msgs, num_images=1, enable_thinking=False)
    img_tok = getattr(cfg, "image_token_index", None) or getattr(cfg, "image_token_id", None)
    inp = prepare_inputs(processor, images=[img_path], prompts=prompt, image_token_index=img_tok)
    kw = {k: v for k, v in inp.items() if k not in ("input_ids", "pixel_values", "attention_mask")}
    try:
        lm.layers = full[: upto + 1]
        out = model(inp["input_ids"], pixel_values=inp["pixel_values"], mask=inp.get("attention_mask"),
                    capture_layer_ids=[upto], return_hidden=True, skip_logits=True, **kw)
    finally:
        lm.layers = full
    h = out.hidden_states[0][0].astype(mx.float32)
    is_img = (inp["input_ids"][0] == img_tok).astype(mx.float32)
    f = mx.stack([h[-1], (h * is_img[:, None]).sum(0) / mx.maximum(is_img.sum(), 1)])
    mx.eval(f)
    return np.array(f, dtype=np.float32)


def extract(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "index.jsonl"))]
    out = os.path.join(d, f"feats_{brain.MODEL.split('/')[-1]}.npz")
    have = {}
    if os.path.exists(out):
        z = np.load(out, allow_pickle=True)
        have = dict(zip(z["img"].tolist(), z["X"]))
    t0, new = time.time(), 0
    for r in rows:
        if r["img"] in have:
            continue
        path = r["img"] if os.path.isabs(r["img"]) else os.path.join(d, "img", r["img"])
        have[r["img"]] = features(path).astype(np.float16)
        new += 1
        if new % 50 == 0:
            print(f"  {len(have)}/{len(rows)} 枚 {(time.time() - t0) / new:.2f} 秒/枚", flush=True)
            _save(out, have, rows)
    _save(out, have, rows)
    print(f"特徴 {len(have)} 枚(新規 {new})→ {out}")


def _save(out, have, rows):
    meta = {r["img"]: r for r in rows}
    imgs = [k for k in have if k in meta]
    np.savez(out, img=np.array(imgs), X=np.stack([have[k] for k in imgs]),
             Y=np.array([[meta[k]["labels"][str(g)] for g in range(N_GOALS)] for k in imgs], dtype=bool),
             init=np.array([meta[k]["init"] for k in imgs]), skill=np.array([meta[k]["skill"] for k in imgs]),
             phase=np.array([meta[k].get("phase", "after") for k in imgs]))


def _fit(X, y, C):
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit((X - mu) / sd, y)
    return clf, mu, sd


def train(d, C=0.05):
    from sklearn.metrics import roc_auc_score
    z = np.load(os.path.join(d, f"feats_{brain.MODEL.split('/')[-1]}.npz"), allow_pickle=True)
    X, Y, groups = z["X"].astype(np.float32), z["Y"], z["init"]
    print(f"画像 {len(X)} 枚、初期状態 {sorted(set(groups.tolist()))}、目標ごとの正例 {Y.sum(0).tolist()}")
    # 初期状態でまとめて 5 つに分けた交差検証(同じ試行の似た画像が学習と評価にまたがらない)。
    # 使う特徴(層)はここ(学習データの中)だけで決める。検証の初期状態の成績では選ばない。
    from sklearn.model_selection import GroupKFold
    folds = list(GroupKFold(n_splits=min(5, len(set(groups.tolist())))).split(X, groups=groups))
    res = {}
    for fi, fname in enumerate(FEAT_NAMES):
        P = np.full(Y.shape, np.nan)
        for g in range(N_GOALS):
            for tr, te in folds:
                if Y[tr, g].all() or not Y[tr, g].any():
                    P[te, g] = Y[tr, g].mean()
                    continue
                clf, mu, sd = _fit(X[tr, fi], Y[tr, g], C)
                P[te, g] = clf.predict_proba((X[te, fi] - mu) / sd)[:, 1]
        aucs = [roc_auc_score(Y[:, g], P[:, g]) if 0 < Y[:, g].sum() < len(Y) and not np.isnan(P[:, g]).any() else np.nan
                for g in range(N_GOALS)]
        res[fname] = (np.nanmean(aucs), aucs, P)
        print(f"  {fname:12s} 平均 AUC {np.nanmean(aucs):.3f}  " + " ".join(f"{a:.2f}" for a in aucs), flush=True)
    best = max(res, key=lambda k: res[k][0])
    fi = FEAT_NAMES.index(best)
    print(f"→ 使う特徴: {best}(平均 AUC {res[best][0]:.3f})")
    W, b, MU, SD = [], [], [], []
    for g in range(N_GOALS):
        clf, mu, sd = _fit(X[:, fi], Y[:, g], C)
        W.append(clf.coef_[0]); b.append(clf.intercept_[0]); MU.append(mu); SD.append(sd)
    os.makedirs(os.path.dirname(HEADS), exist_ok=True)
    np.savez(HEADS, W=np.array(W), b=np.array(b), mu=np.array(MU), sd=np.array(SD), feat=fi, feat_name=best,
             model=brain.MODEL)
    np.save(os.path.join(d, "cv_pred.npy"), res[best][2])
    print(f"頭を保存 → {HEADS}")


_heads = None


EARLY_EXIT = os.environ.get("PAI_JEV_EARLY_EXIT", "1") == "1"


def predict(img_path):
    """10 個の目標それぞれの達成確率(画像 1 回の読み込み)。
    使う特徴が途中の層なら、その層で打ち切って計算する(PAI_JEV_EARLY_EXIT=0 で全層を計算する比較用)。"""
    global _heads
    if _heads is None:
        _heads = dict(np.load(HEADS, allow_pickle=True))
    if "spec" in _heads:  # 第 2 版: 複数の特徴をつないだ頭(jev2.py)
        spec = [str(x) for x in _heads["spec"]]
        layers = {n.split("_")[0] for n in spec}
        if EARLY_EXIT and len(layers) == 1 and next(iter(layers)).startswith("L"):
            L = int(next(iter(layers))[1:])
            two = features(img_path, upto=L)
            f = np.concatenate([two[0 if n.endswith("末尾") else 1] for n in spec])
        else:
            allf = features(img_path)
            f = np.concatenate([allf[FEAT_NAMES.index(n)] for n in spec])
    else:
        name = str(_heads["feat_name"])
        layer, pool = name.split("_")
        if EARLY_EXIT and layer.startswith("L"):
            f = features(img_path, upto=int(layer[1:]))[0 if pool == "末尾" else 1]
        else:
            f = features(img_path)[int(_heads["feat"])]
    logit = ((f - _heads["mu"]) / _heads["sd"] * _heads["W"]).sum(1) + _heads["b"]
    return 1 / (1 + np.exp(-logit))


def from_runs(out_dir, *runs_and_dirs):
    """実行器の結果(jsonl)と画像の置き場の組から、改造 Jev のデータの目録(index.jsonl)を作る。"""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "index.jsonl"), "w") as f:
        for runs, snap_dir in zip(runs_and_dirs[::2], runs_and_dirs[1::2]):
            for r in map(json.loads, open(runs)):
                if "img" in r:
                    f.write(json.dumps({"img": os.path.abspath(os.path.join(snap_dir, r["img"])), "labels": r["labels"],
                                        "skill": r["skill"], "init": r["init"], "phase": "after", "ok": r["ok"]}) + "\n")


def evaltest(train_dir, test_dir, zero=False, C=0.05):
    """学習用(train_dir)の全部で頭を学習 → 検証用(test_dir、初期状態が重ならない)で測る。ゼロショットとも同じ画像で比べる。"""
    from sklearn.metrics import roc_auc_score
    tag = brain.MODEL.split('/')[-1]
    tr = np.load(os.path.join(train_dir, f"feats_{tag}.npz"), allow_pickle=True)
    te = np.load(os.path.join(test_dir, f"feats_{tag}.npz"), allow_pickle=True)
    assert not set(tr["init"].tolist()) & set(te["init"].tolist()), "学習と検証の初期状態が重なっている"
    Xtr, Ytr, Xte, Yte = tr["X"].astype(np.float32), tr["Y"], te["X"].astype(np.float32), te["Y"]
    res = {}
    for fi, fname in enumerate(FEAT_NAMES):
        P = np.zeros(Yte.shape)
        for g in range(N_GOALS):
            if Ytr[:, g].all() or not Ytr[:, g].any():
                P[:, g] = Ytr[:, g].mean()
                continue
            clf, mu, sd = _fit(Xtr[:, fi], Ytr[:, g], C)
            P[:, g] = clf.predict_proba((Xte[:, fi] - mu) / sd)[:, 1]
        aucs = [roc_auc_score(Yte[:, g], P[:, g]) if 0 < Yte[:, g].sum() < len(Yte) else np.nan for g in range(N_GOALS)]
        acc = float(((P >= 0.5) == Yte).mean())
        res[fname] = {"auc_mean": float(np.nanmean(aucs)), "aucs": [None if np.isnan(a) else round(float(a), 3) for a in aucs],
                      "acc": round(acc, 3)}
        print(f"  {fname:12s} 平均 AUC {np.nanmean(aucs):.3f}  正解率 {acc:.3f}  " + " ".join("—" if np.isnan(a) else f"{a:.2f}" for a in aucs), flush=True)
    out = {"train": train_dir, "test": test_dir, "n_train": len(Xtr), "n_test": len(Xte),
           "pos_train": Ytr.sum(0).tolist(), "pos_test": Yte.sum(0).tolist(), "jev": res}
    if zero:  # ゼロショット(問いを 10 回)を同じ検証画像で
        imgs = te["img"].tolist()
        Pz = np.zeros(Yte.shape)
        t0 = time.time()
        for i, im in enumerate(imgs):
            for g in range(N_GOALS):
                Pz[i, g] = brain.p_yes(CHECK_Q[g], im)[0]
        aucs = [roc_auc_score(Yte[:, g], Pz[:, g]) if 0 < Yte[:, g].sum() < len(Yte) else np.nan for g in range(N_GOALS)]
        out["zero"] = {"auc_mean": float(np.nanmean(aucs)), "aucs": [None if np.isnan(a) else round(float(a), 3) for a in aucs],
                       "acc": round(float(((Pz >= 0.5) == Yte).mean()), 3), "sec_per_image": round((time.time() - t0) / len(imgs), 2)}
        print(f"  ゼロショット 平均 AUC {out['zero']['auc_mean']:.3f}  正解率 {out['zero']['acc']:.3f}  {out['zero']['sec_per_image']} 秒/画像(10 問)")
    json.dump(out, open(os.path.join(test_dir, f"evaltest_{tag}.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    cmd, arg = sys.argv[1], sys.argv[2]
    if cmd == "from_runs":
        from_runs(arg, *sys.argv[3:])
        sys.exit()
    if cmd == "evaltest":
        evaltest(arg, sys.argv[3], zero="--zero" in sys.argv)
        sys.exit()
    if cmd == "extract":
        extract(arg)
    elif cmd == "train":
        train(arg)
    elif cmd == "predict":
        p = predict(arg)
        for g in range(N_GOALS):
            print(f"{p[g]:.3f}  {CHECK_Q[g]}")
