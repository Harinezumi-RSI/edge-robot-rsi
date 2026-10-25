"""頭の確認を改造 Jev にする: 「ひとこと + 手順」を Qwen に読ませ、途中の層の状態に「手順が合っているか」の線形の頭を付ける。
.venv-mlx で実行。

  python kitchen/plan_jev.py train      # 合成の発話 60 本 + 調整用 30 本 × (正しい手順・崩した手順)で学習、層は交差検証で選ぶ
  python kitchen/plan_jev.py test2      # 新しい評価用 30 本(1 回だけ): 生成の手順の当たり外れを、学習した頭とゼロショットで比べる

崩した手順: 1 つ取り替え・余計な動作を足す・1 つ抜く・空にする・(頼みでないのに)動かす・順番を入れ替える(正解の一覧に無ければ)。
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from kitchen import brain  # noqa: E402
from kitchen.runlog import Timer  # noqa: E402
from kitchen.skills import JA  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEADS = os.path.join(ROOT, "models", "plan_jev.npz")
LAYERS = (8, 16, 24)


def prompt_for(utterance, plan_ids):
    steps = " → ".join(JA[i] for i in plan_ids) if plan_ids else "何もしない"
    return [{"role": "system", "content": brain.CHECK_SYS},
            {"role": "user", "content": f"ひとこと:「{utterance}」\nロボットの手順: {steps}\nこの手順はその人の望みにちょうど合っていますか?"}]


def text_features(messages, layers=LAYERS):
    """文章だけを読ませ、指定した層の最後のトークンの状態を返す([層の数, 2560])。最も深い層で打ち切る。"""
    import mlx.core as mx
    model, processor = brain._load()
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    ids = mx.array([tok.encode(prompt, add_special_tokens=False)])
    lm = model.language_model.model
    full = lm.layers
    try:
        lm.layers = full[: max(layers) + 1]
        out = model.language_model(ids, capture_layer_ids=list(layers), return_hidden=True, skip_logits=True)
    finally:
        lm.layers = full
    f = mx.stack([h[0, -1].astype(mx.float32) for h in out.hidden_states[: len(layers)]])
    mx.eval(f)
    return np.array(f, dtype=np.float32)


def corrupt(ok_plans, rng):
    """崩した手順(正解の一覧に入らないもの)をいくつか作る。"""
    gold = ok_plans[0]
    cands = set()
    skills = list(JA)
    if gold:
        for _ in range(3):
            p = list(gold)
            p[rng.randrange(len(p))] = rng.choice(skills)
            cands.add(tuple(p))
        cands.add(tuple(gold + [rng.choice(skills)]))
        if len(gold) >= 2:
            cands.add(tuple(gold[:-1]))
            cands.add(tuple(reversed(gold)))
        cands.add(())
    else:
        for _ in range(3):
            cands.add((rng.choice(skills),))
    return [list(c) for c in cands if list(c) not in ok_plans and all(len(set(c)) == len(c) for _ in [0])]


def build_pairs(files, seed=0):
    rng = random.Random(seed)
    pairs = []
    for f in files:
        for gi, line in enumerate(open(os.path.join(ROOT, f), encoding="utf-8")):
            d = json.loads(line)
            g = f"{os.path.basename(f)}:{gi}"
            for p in d["ok"]:
                pairs.append((d["text"], p, 1, g))
            for p in corrupt(d["ok"], rng):
                pairs.append((d["text"], p, 0, g))
    return pairs


def train():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    pairs = build_pairs(["data/utterances_train_synth.jsonl", "data/utterances.jsonl"])
    t0 = time.time()
    X = np.stack([text_features(prompt_for(u, p)) for u, p, _, _ in pairs])
    y = np.array([lab for _, _, lab, _ in pairs])
    groups = np.array([g for _, _, _, g in pairs])
    print(f"組 {len(pairs)}(正 {y.sum()}・誤 {len(y) - y.sum()})、特徴 {time.time() - t0:.0f} 秒", flush=True)
    best = None
    for li, L in enumerate(LAYERS):
        P = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            mu, sd = X[tr, li].mean(0), X[tr, li].std(0) + 1e-6
            clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit((X[tr, li] - mu) / sd, y[tr])
            P[te] = clf.predict_proba((X[te, li] - mu) / sd)[:, 1]
        auc = roc_auc_score(y, P)
        print(f"  {L} 層目: 交差検証 AUC {auc:.3f}", flush=True)
        if best is None or auc > best[0]:
            best = (auc, li, L, P)
    auc, li, L, P = best
    mu, sd = X[:, li].mean(0), X[:, li].std(0) + 1e-6
    clf = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced").fit((X[:, li] - mu) / sd, y)
    os.makedirs(os.path.dirname(HEADS), exist_ok=True)
    np.savez(HEADS, W=clf.coef_[0], b=clf.intercept_[0], mu=mu, sd=sd, layer=L, cv_auc=auc)
    # 聞き返しのしきい値も学習データの中(交差検証の予測)だけで決める: 採用した手順の 9 割以上が正しい一番小さい値
    taus = sorted(set(np.round(P, 3)))
    tau = next((t for t in taus if (y[P >= t].mean() if (P >= t).any() else 0) >= 0.9), 0.5)
    np.savez(HEADS, W=clf.coef_[0], b=clf.intercept_[0], mu=mu, sd=sd, layer=L, cv_auc=auc, tau=tau)
    print(f"→ {L} 層目を使う(交差検証 AUC {auc:.3f})、聞き返しのしきい値 {tau:.3f}、保存 {HEADS}")
    return {"pairs": len(pairs), "layer": L, "cv_auc": round(float(auc), 3), "tau": float(tau)}


_h = None


def p_correct(utterance, plan_ids):
    global _h
    if _h is None:
        _h = dict(np.load(HEADS))
    L = int(_h["layer"])
    f = text_features(prompt_for(utterance, plan_ids), layers=(L,))[0]
    z = float(((f - _h["mu"]) / _h["sd"] * _h["W"]).sum() + _h["b"])
    return 1 / (1 + np.exp(-z))


def test2():
    from sklearn.metrics import roc_auc_score
    h = dict(np.load(HEADS))
    tau_jev, tau_zero = float(h["tau"]), 0.15
    rows = []
    for line in open(os.path.join(ROOT, "data/utterances_test2.jsonl"), encoding="utf-8"):
        d = json.loads(line)
        t = time.time()
        pl = brain.plan(d["text"])["plan"]
        t_plan = time.time() - t
        r = {"id": d["id"], "kind": d["kind"], "plan": pl, "hit": pl in d["ok"], "sec_plan": round(t_plan, 2)}
        if pl:
            t = time.time(); r["p_jev"] = round(p_correct(d["text"], pl), 4); r["sec_jev"] = round(time.time() - t, 3)
            t = time.time(); r["p_zero"] = round(brain.plan_ok(d["text"], pl), 4); r["sec_zero"] = round(time.time() - t, 3)
        rows.append(r)
        print(r, flush=True)
    act = [r for r in rows if r["plan"]]
    out = {"plan_acc": int(sum(r["hit"] for r in rows)), "n": len(rows), "n_plans": len(act)}
    for name, key, tau in (("改造 Jev", "p_jev", tau_jev), ("ゼロショット", "p_zero", tau_zero)):
        pos = [r[key] for r in act if r["hit"]]
        neg = [r[key] for r in act if not r["hit"]]
        auc = roc_auc_score([1] * len(pos) + [0] * len(neg), pos + neg) if pos and neg else float("nan")
        wrong_acted = sum((not r["hit"]) and r[key] >= tau for r in act)
        unnecessary = sum(r["hit"] and r[key] < tau for r in act)
        caught = sum((not r["hit"]) and r[key] < tau for r in act)
        out[name] = {"auc": round(float(auc), 3), "tau": float(tau), "wrong_acted": int(wrong_acted), "caught": int(caught),
                     "unnecessary_ask": int(unnecessary),
                     "sec": round(float(np.median([r["sec_jev" if key == "p_jev" else "sec_zero"] for r in act])), 3)}
        print(f"{name}: AUC {auc:.3f}(τ={tau:.3f})間違ったまま動く {wrong_acted}・止めた {caught}・不要な聞き返し {unnecessary}", flush=True)
    print(f"手順づくり(生成): {out['plan_acc']}/{out['n']}")
    json.dump({"summary": out, "rows": rows}, open(os.path.join(ROOT, "outputs/plan_jev_test2.json"), "w"), ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    with Timer("judge", f"頭の確認の改造 Jev: {cmd}") as tm:
        res = train() if cmd == "train" else test2()
        tm.info.update({k: v for k, v in res.items() if not isinstance(v, dict)})
