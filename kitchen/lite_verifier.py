"""本番で候補を選ぶ判定器(LLM の検証器つき best-of-N と同じ形)の下調べ。.venv で実行。

  python kitchen/lite_verifier.py build --judged outputs/lite/x1o/practice_{A,B,C}_judged.jsonl --demos data/lite_rsi/x1o/demos --out data/lite/x1o_verifier.npz
  python kitchen/lite_verifier.py train --data data/lite/x1o_verifier.npz --out models/lite/verifier_x1o.npz

自己進化の練習(揺らぎつきの試行、改造 Jev の採点つき)を使う。1 サンプル = 塊の始まりの場面(描き直し)と、そこから実際に動かした 4 step、
答え = その試行が最後にうまくいったか(改造 Jev の採点 ≥ 0.8。シミュレータの正解は比べるためだけ)。
train は 2 つの頭を比べる: 場面だけ(V)と、場面 + 動かした塊(Q)。Q の当たり具合が V より上がらなければ、塊の候補を選んでも意味がない
(揺らぎの向きで成否が決まっていない)→ 本番には入れない。画像の特徴は土台(DINOv2-small)の 2 枚それぞれの平均(768 次元)。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

TAU = 0.8
EXEC = 4


def build(a):
    from kitchen.chain_env import KitchenChainEnv
    from kitchen.lite_feats import extract, obs_state
    from kitchen.record import restore_obs
    env = KitchenChainEnv(start_task=1, init_index=0)
    env.reset()
    X, S, F, C, Y, Ytrue, E, K = [], [], [], [], [], [], [], []
    t0, ei = time.time(), 0
    for path in a.judged:
        for n_row, r in enumerate(map(json.loads, open(path))):
            if a.limit and n_row >= a.limit:
                break
            z = np.load(os.path.join(a.demos, r["demo"]))
            states, actions, skill = z["states"], z["actions"], int(z["skill"])
            fix = z["fixtures"] if "fixtures" in z else None
            y, yt = float(r["p_jev"] >= TAU), float(bool(r["labels"][str(skill)]))
            for t in range(0, len(actions) - 1, a.every):
                obs = restore_obs(env, states[t], fix)
                f, _, g = extract("dino", obs, skill)
                g = np.asarray(g, np.float32).reshape(2, -1, g.shape[-1]).mean(1).reshape(-1)  # 2 枚それぞれの平均
                ch = actions[t:t + EXEC]
                if len(ch) < EXEC:
                    ch = np.concatenate([ch, np.repeat(ch[-1:], EXEC - len(ch), 0)])
                X.append(g.astype(np.float16)); S.append(obs_state(obs)); F.append(np.asarray(f, np.float32).reshape(-1))
                C.append(ch.reshape(-1).astype(np.float32)); Y.append(y); Ytrue.append(yt); E.append(ei); K.append(skill)
            ei += 1
            if ei % 100 == 0:
                print(f"  試行 {ei}・サンプル {len(X)}・{(time.time() - t0) / 60:.1f} 分", flush=True)
    np.savez(a.out, X=np.stack(X), S=np.stack(S), F=np.stack(F), C=np.stack(C), Y=np.array(Y), Ytrue=np.array(Ytrue), E=np.array(E), K=np.array(K))
    print(f"保存: {a.out}(試行 {ei}・サンプル {len(X)}・成功の割合 {np.mean(Y):.2f}・{(time.time() - t0) / 60:.1f} 分)")


def auc(y, p):
    o = np.argsort(p)
    r = np.empty(len(p))
    r[o] = np.arange(1, len(p) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def train(a):
    import torch
    import torch.nn as nn
    z = np.load(a.data)
    X, S, F, C, Y, E = (z[k].astype(np.float32) for k in ("X", "S", "F", "C", "Y", "E"))
    rng = np.random.default_rng(0)
    ep = np.unique(E)
    va_ep = set(rng.choice(ep, size=max(1, len(ep) // 5), replace=False).tolist())
    va = np.array([e in va_ep for e in E])
    stat = {}

    def norm(M, key):
        mu, sd = M[~va].mean(0), M[~va].std(0) + 1e-6
        stat[key] = (mu, sd)
        return (M - mu) / sd

    Xn, Sn, Cn = norm(X, "X"), norm(S, "S"), norm(C, "C")
    out = {}
    for name, parts in (("V", [Xn, Sn, F]), ("Q", [Xn, Sn, F, Cn])):
        Z = torch.tensor(np.concatenate(parts, 1))
        y = torch.tensor(Y)
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(Z.shape[1], 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        tr_idx = np.where(~va)[0]
        best = (-1.0, None)
        for epoch in range(a.epochs):
            net.train()
            rng.shuffle(tr_idx)
            for i in range(0, len(tr_idx), 512):
                b = tr_idx[i:i + 512]
                loss = nn.functional.binary_cross_entropy_with_logits(net(Z[b]).squeeze(1), y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                p = net(Z[va]).squeeze(1).numpy()
            s = auc(Y[va], p)
            if s > best[0]:
                best = (s, {k: v.detach().numpy().copy() for k, v in net.state_dict().items()})
        out[name] = best
        print(f"{name}(場面{' + 動かした塊' if name == 'Q' else 'だけ'}): 検証の AUC {best[0]:.3f}", flush=True)
    gain = out["Q"][0] - out["V"][0]
    print(f"塊を見た上積み: {gain:+.3f}(検証の試行 {len(va_ep)}・サンプル {va.sum()})")
    np.savez(a.out, auc_v=out["V"][0], auc_q=out["Q"][0], **{f"q_{k}": v for k, v in out["Q"][1].items()},
             **{f"mu_{k}": v[0] for k, v in stat.items()}, **{f"sd_{k}": v[1] for k, v in stat.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "train"])
    ap.add_argument("--judged", nargs="*", default=[])
    ap.add_argument("--demos", default="")
    ap.add_argument("--every", type=int, default=8, help="塊の始まりを何 step おきに取るか")
    ap.add_argument("--data", default="")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0, help="試し: 1 ファイルあたりの試行の数")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a) if a.mode == "build" else train(a)


if __name__ == "__main__":
    main()
