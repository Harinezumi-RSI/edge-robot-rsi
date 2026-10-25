"""自己進化の学び直しを「いまの手から小さく動かす」。.venv で実行。

  python kitchen/lite_finetune.py --init models/lite/c1o.npz --data <復習のデータ...> --hits <当たりのデータ...> --out models/lite/x3o.npz

LLM の RL が参照モデルから離れすぎないように学ぶのと同じく、いまの頭の重みと正規化の値をそのまま使い、
復習(元の学習データ)と当たり(自己進化の練習で改造 Jev が選んだもの)を混ぜて、小さな学習率で少しだけ学ぶ。
lite_train.py はまっさらな頭から学び直すので、学習の種の揺れで動作が丸ごと入れ替わる(29 周目)。x1o・x2o で狙っていない動作が
下がったのはこの揺れの分もあると見て、それを避ける。出力の形は lite_train.py と同じ(lite_batch.py・demo.py でそのまま使える)。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from kitchen.lite_common import state_feats  # noqa: E402
from kitchen.lite_train import Head, load  # noqa: E402
from kitchen.runlog import Timer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="出発点の頭(この重みと正規化の値を使う)")
    ap.add_argument("--data", nargs="+", required=True, help="復習のデータ(出発点の頭の学習データ)")
    ap.add_argument("--hits", nargs="+", required=True, help="自己進化の当たりのデータ")
    ap.add_argument("--hit-weight", type=float, default=3.0, help="当たりのサンプルの重み(復習に対して)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    h = dict(np.load(a.init, allow_pickle=True))
    assert str(h["arch"]) == "grid", "grid の頭だけ"
    F0, G0, S0, A0, K0, E0, names, backend = load(a.data)
    F1, G1, S1, A1, K1, E1, names1, backend1 = load(a.hits)
    assert names == names1 and backend == backend1 == str(h["backend"])
    F, G, S8, A, K = (np.concatenate(x) for x in ((F0, F1), (G0, G1), (S0, S1), (A0, A1), (K0, K1)))
    is_hit = np.r_[np.zeros(len(F0), bool), np.ones(len(F1), bool)]
    cnt = np.bincount(K, minlength=10).astype(np.float64)
    w = (1.0 / cnt[K]) * np.where(is_hit, a.hit_weight, 1.0)  # 動作ごとにそろえ、当たりを重く
    Sn = ((state_feats(S8) - h["s_mu"]) / h["s_sd"]).astype(np.float32)
    Gn = ((G.astype(np.float32) - h["g_mu"]) / h["g_sd"]).astype(np.float16)
    spec = [str(n) for n in h["feat_names"]]
    Ln = ((np.concatenate([F[:, names.index(n)] for n in spec], axis=1) - h["l_mu"]) / h["l_sd"]).astype(np.float32)
    Y = ((A - h["a_mu"]) / h["a_sd"]).reshape(len(A), -1).astype(np.float32)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    m = Head("grid", Ln.shape[1], Gn.shape[1], Gn.shape[2])
    with torch.no_grad():  # 出発点の重みを入れる
        m.P.weight.copy_(torch.from_numpy(h["P_W"].T)); m.P.bias.copy_(torch.from_numpy(h["P_b"]))
        m.L.weight.copy_(torch.from_numpy(h["L_W"].T)); m.L.bias.copy_(torch.from_numpy(h["L_b"]))
        lin = [l for l in m.net if isinstance(l, torch.nn.Linear)]
        for i, l in enumerate(lin):
            l.weight.copy_(torch.from_numpy(h[f"W{i}"].T)); l.bias.copy_(torch.from_numpy(h[f"b{i}"]))
    T = lambda x, i: torch.from_numpy(np.asarray(x[i], dtype=np.float32))  # noqa: E731
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=1e-4)
    n = len(F)
    print(f"復習 {len(F0)}・当たり {len(F1)}(重み ×{a.hit_weight})・学習率 {a.lr}・{a.epochs} 周", flush=True)

    def loss_on(idx):
        m.eval()
        with torch.no_grad():
            return sum(float(((m(T(Ln, idx[j:j + 512]), T(Sn, idx[j:j + 512]), T(Gn, idx[j:j + 512])) - T(Y, idx[j:j + 512])) ** 2).sum())
                       for j in range(0, len(idx), 512)) / (len(idx) * Y.shape[1])

    hits_idx, rep_idx = np.where(is_hit)[0], np.where(~is_hit)[0][::20]
    print(f"  出発点: 当たりの損失 {loss_on(hits_idx):.4f}・復習の損失 {loss_on(rep_idx):.4f}", flush=True)
    with Timer("train", f"軽い手を小さく学び直す {os.path.basename(a.out)}(出発点 {os.path.basename(a.init)})", samples=int(n), hits=int(len(F1)),
               lr=a.lr, epochs=a.epochs, hit_weight=a.hit_weight) as tm:
        for ep in range(a.epochs):
            m.train()
            idx = rng.choice(n, size=n, replace=True, p=w / w.sum())
            for i in range(0, n, 256):
                b = idx[i:i + 256]
                loss = ((m(T(Ln, b), T(Sn, b), T(Gn, b)) - T(Y, b)) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            print(f"  {ep + 1} 周: 当たりの損失 {loss_on(hits_idx):.4f}・復習の損失 {loss_on(rep_idx):.4f}", flush=True)
        out = {k: v for k, v in h.items() if not (k.startswith("W") or k.startswith("b") and k[1:].isdigit())}
        out.update({f"W{i}": l.weight.detach().numpy().T.astype(np.float32) for i, l in enumerate(lin)})
        out.update({f"b{i}": l.bias.detach().numpy().astype(np.float32) for i, l in enumerate(lin)})
        out.update(P_W=m.P.weight.detach().numpy().T.astype(np.float32), P_b=m.P.bias.detach().numpy().astype(np.float32),
                   L_W=m.L.weight.detach().numpy().T.astype(np.float32), L_b=m.L.bias.detach().numpy().astype(np.float32),
                   data=np.array(a.data + a.hits), init=a.init, lr=a.lr, epochs=a.epochs, hit_weight=a.hit_weight)
        np.savez(a.out, **out)
        tm.info.update(train_steps=int(a.epochs * n / 256))
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
