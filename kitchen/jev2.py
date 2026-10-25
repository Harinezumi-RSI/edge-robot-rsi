"""改造 Jev の第 2 版: データを増やし、ミニ RSI で効く物差しで作りを選ぶ。.venv-mlx で実行。

  python kitchen/jev2.py build    # 学習用(練習の画像、初期状態 10〜39)と評価用(初期状態 0〜9)の目録と特徴
  python kitchen/jev2.py train    # 作り(層・つなぎ方・正則化)を交差検証で選ぶ → models/jev_heads_v2.npz
  python kitchen/jev2.py eval     # 評価用の画像で、第 1 版と第 2 版を比べる

物差し: 「やろうとした動作の目標」の判定(その動作の試行の画像だけで測る AUC と、しきい値ごとの精度・拾えた割合)。
全目標を混ぜた AUC は、明らかに違う目標(簡単な負例)が多くて高く出るので、作りの選択には使わない。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from kitchen import brain, jev  # noqa: E402
from kitchen.runlog import Timer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = [("outputs/rsi/r0/practice.jsonl", "data/snaps/train0"), ("outputs/rsi/r1/practice.jsonl", "data/rsi/r1/snaps"),
         ("outputs/rsi/r2/practice.jsonl", "data/rsi/r2/snaps"), ("outputs/rsi/ris1/practice.jsonl", "data/rsi/ris1/snaps"),
         ("outputs/rsi/ris2/practice.jsonl", "data/rsi/ris2/snaps")]
VAL = [("outputs/fast/val_all.jsonl", "data/snaps/val_all"), ("outputs/rsi/r0/val_same.jsonl", "data/snaps/val_rsi"),
       ("outputs/rsi/r1/val.jsonl", "data/snaps/val_rsi"), ("outputs/rsi/r2/val.jsonl", "data/snaps/val_rsi"),
       ("outputs/ris/val_s10_nohome.jsonl", "data/snaps/ris_val"), ("outputs/ris/val_s10_home.jsonl", "data/snaps/ris_val"),
       ("outputs/rsi/ris1/val.jsonl", "data/snaps/val_rsi"), ("outputs/rsi/ris2/val.jsonl", "data/snaps/val_rsi"),
       ("outputs/ris/val_s0_base.jsonl", "data/snaps/val_rsi"), ("outputs/rsi/ris1/val_s0.jsonl", "data/snaps/val_rsi"),
       ("outputs/ris/transfer_base.jsonl", "data/snaps/ris_val"), ("outputs/ris/transfer_ris1.jsonl", "data/snaps/ris_val"),
       ("outputs/ris/val_all_s10_base.jsonl", "data/snaps/ris_val")]
D_TR, D_VA = os.path.join(ROOT, "data/judge_v2_train"), os.path.join(ROOT, "data/judge_v2_val")
DEFAULT_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
TAG = "" if brain.MODEL == DEFAULT_MODEL else "_" + brain.MODEL.split("/")[-1]  # 別の土台(0.8B など)で作るときは別の名前で保存
HEADS2 = os.path.join(ROOT, "models", f"jev_heads_v2{TAG}.npz")
NAMES = jev.FEAT_NAMES  # L8/L16/L24/最終 × 末尾/画像平均
SPECS = [[n] for n in NAMES] + [["L8_末尾", "L16_末尾"], ["L8_末尾", "L8_画像平均"], ["L16_末尾", "L16_画像平均"],
                               ["L8_末尾", "L16_末尾", "L24_末尾"], ["L8_画像平均", "L16_画像平均"]]


def build():
    pairs = []
    for runs, snap in TRAIN:
        pairs += [os.path.join(ROOT, runs), os.path.join(ROOT, snap)]
    jev.from_runs(D_TR, *pairs)
    pairs = []
    for runs, snap in VAL:
        pairs += [os.path.join(ROOT, runs), os.path.join(ROOT, snap)]
    jev.from_runs(D_VA, *pairs)
    # 練習 0 周目の 300 枚は特徴を取ってあるので使い回す(同じ画像・同じ取り出し方)
    old = os.path.join(ROOT, "data/judge_train0", f"feats_{brain.MODEL.split('/')[-1]}.npz")
    new = os.path.join(D_TR, f"feats_{brain.MODEL.split('/')[-1]}.npz")
    if os.path.exists(old) and not os.path.exists(new):
        z = np.load(old, allow_pickle=True)
        rows = [json.loads(l) for l in open(os.path.join(D_TR, "index.jsonl"))]
        have = {k: v for k, v in zip(z["img"].tolist(), z["X"]) if k in {r["img"] for r in rows}}
        jev._save(new, have, rows)
        print(f"使い回し {len(have)} 枚")
    jev.extract(D_TR)
    jev.extract(D_VA)


def load(d):
    z = np.load(os.path.join(d, f"feats_{brain.MODEL.split('/')[-1]}.npz"), allow_pickle=True)
    return z["X"].astype(np.float32), z["Y"], z["init"], z["skill"], z["img"]


def feat(X, spec):
    return np.concatenate([X[:, NAMES.index(n)] for n in spec], axis=1)


def fit(Xs, y, C):
    from sklearn.linear_model import LogisticRegression
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    return LogisticRegression(C=C, max_iter=4000, class_weight="balanced").fit((Xs - mu) / sd, y), mu, sd


def attempted_auc(P, Y, skill):
    """やろうとした動作の目標だけで測る AUC(目標ごと → 平均)。"""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for g in range(Y.shape[1]):
        m = skill == g
        if m.sum() and 0 < Y[m, g].sum() < m.sum():
            aucs.append(roc_auc_score(Y[m, g], P[m, g]))
    return float(np.mean(aucs)), aucs


def train():
    from sklearn.model_selection import GroupKFold
    X, Y, groups, skill, _ = load(D_TR)
    print(f"学習の画像 {len(X)} 枚、初期状態 {len(set(groups.tolist()))} 種、やろうとした目標が満たされていた枚数 "
          f"{[int(Y[skill == g, g].sum()) for g in range(10)]}", flush=True)
    folds = list(GroupKFold(n_splits=5).split(X, groups=groups))
    res = []
    for spec in SPECS:
        F = feat(X, spec)
        for C in (0.01, 0.05, 0.2):
            P = np.zeros(Y.shape)
            for g in range(Y.shape[1]):
                for tr, te in folds:
                    if Y[tr, g].all() or not Y[tr, g].any():
                        P[te, g] = Y[tr, g].mean()
                        continue
                    clf, mu, sd = fit(F[tr], Y[tr, g], C)
                    P[te, g] = clf.predict_proba((F[te] - mu) / sd)[:, 1]
            a, _ = attempted_auc(P, Y, skill)
            res.append((a, spec, C))
            print(f"  {'+'.join(spec):28s} C={C:<5} やろうとした目標の AUC {a:.4f}", flush=True)
    a, spec, C = max(res, key=lambda r: r[0])
    F = feat(X, spec)
    W, b, MU, SD = [], [], [], []
    for g in range(Y.shape[1]):
        clf, mu, sd = fit(F, Y[:, g], C)
        W.append(clf.coef_[0]); b.append(clf.intercept_[0]); MU.append(mu); SD.append(sd)
    np.savez(HEADS2, W=np.array(W), b=np.array(b), mu=np.array(MU), sd=np.array(SD), spec=np.array(spec), C=C, cv=a)
    print(f"→ 第 2 版: {'+'.join(spec)} C={C}(交差検証 {a:.4f})→ {HEADS2}")
    return {"spec": "+".join(spec), "C": C, "cv_attempted_auc": round(a, 4), "n_train": int(len(X))}


def predict_matrix(X, heads):
    spec = [str(s) for s in heads["spec"]] if "spec" in heads else [NAMES[int(heads["feat"])]]
    F = feat(X, spec)
    logit = np.stack([((F - heads["mu"][g]) / heads["sd"][g] * heads["W"][g]).sum(1) + heads["b"][g] for g in range(len(heads["W"]))], 1)
    return 1 / (1 + np.exp(-logit))


def evaluate():
    X, Y, groups, skill, imgs = load(D_VA)
    v2 = dict(np.load(HEADS2, allow_pickle=True))
    heads = [("第 2 版" + TAG, v2)]
    if not TAG:  # 第 1 版は 4B の特徴でしか使えない
        heads.insert(0, ("第 1 版(300 枚・8 層目末尾)", dict(np.load(os.path.join(ROOT, "models", "jev_heads_v1.npz"), allow_pickle=True))))
    out = {}
    for name, h in heads:
        P = predict_matrix(X, h)
        a, aucs = attempted_auc(P, Y, skill)
        p_att = P[np.arange(len(P)), skill]
        y_att = Y[np.arange(len(Y)), skill]
        row = {"attempted_auc": round(a, 4), "per_goal": [round(x, 3) for x in aucs]}
        for tau in (0.5, 0.8):
            acc = p_att >= tau
            row[f"tau{tau}"] = {"precision": round(float(y_att[acc].mean()), 3), "recall": round(float(acc[y_att].mean()), 3),
                                "accepted": int(acc.sum())}
        out[name] = row
        print(f"{name}: やろうとした目標の AUC {a:.4f} / τ=0.5 精度 {row['tau0.5']['precision']}・拾えた {row['tau0.5']['recall']}"
              f" / τ=0.8 精度 {row['tau0.8']['precision']}・拾えた {row['tau0.8']['recall']}", flush=True)
    print(f"評価の画像 {len(X)} 枚(初期状態 0〜9、学習に不使用)、やろうとした目標が満たされていた {int(Y[np.arange(len(Y)), skill].sum())} 枚")
    json.dump(out, open(os.path.join(D_VA, f"compare_v1_v2{TAG}.json"), "w"), ensure_ascii=False, indent=1)
    return {k: v["attempted_auc"] for k, v in out.items()}


if __name__ == "__main__":
    cmd = sys.argv[1]
    with Timer("judge", f"改造 Jev 第 2 版{TAG}: {cmd}") as tm:
        r = {"build": build, "train": train, "eval": evaluate}[cmd]()
        if isinstance(r, dict):
            tm.info.update({k: v for k, v in r.items() if not isinstance(v, (dict, list))})
