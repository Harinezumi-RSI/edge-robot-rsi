"""軽い手の行動の頭を学習する(.venv で実行、数分、CPU)。

  python kitchen/lite_train.py --data data/lite/r0.npz [data/lite/xxx.npz ...] --out models/lite_head.npz

入力: 土台の特徴(lite_feats.py: 要約 F と 4×4 マス G)+ 手先の状態(位置・回転行列の 2 列・指 = 11 次元)。
出力: 8 step 分の行動(7 次元 × 8)。損失は正規化した行動の平均二乗誤差。動作ごとのサンプル数の偏りは重みでならす。
作りの候補:
  pool : 要約の特徴 + 状態 → 2 層の MLP
  grid : マスごとに共通の射影(1024→64)で位置を残す + 最後のトークン(指示文)を 128 に射影 + 状態 → 2 層の MLP
「試行ごとに分けた検証の損失」で作りを選ぶ(同じ試行の似たサンプルが学習と検証にまたがらない)。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from kitchen.lite_common import CHUNK, GRID_LAYER, NAMES, state_feats  # noqa: E402

BACKEND_UPTO = {"qwen08": GRID_LAYER}  # マスを取る層まで計算する(言語モデルを使わない土台は 0)
from kitchen.runlog import Timer  # noqa: E402

CANDS = [("pool", ["L12_末尾", "L12_画像平均"]), ("pool", ["L6_画像平均", "L12_画像平均", "L18_末尾"]),
         ("grid", ["L12_末尾"]), ("grid", ["L18_末尾"]), ("grid", ["指示_one-hot"])]


def load(paths):
    zs = [np.load(p, allow_pickle=True) for p in paths]
    off, E = 0, []
    for z in zs:
        E.append(z["E"] + off)
        off += int(z["E"].max()) + 1
    cat = lambda k: np.concatenate([z[k] for z in zs])  # noqa: E731
    names = {tuple(str(n) for n in z["F_names"]) if "F_names" in z else tuple(NAMES) for z in zs}
    backends = {str(z["backend"]) if "backend" in z else "qwen08" for z in zs}
    assert len(names) == 1 and len(backends) == 1, f"土台の違うデータを混ぜている: {backends} {names}"
    return cat("F"), cat("G"), cat("S"), cat("A"), cat("K"), np.concatenate(E), list(names.pop()), backends.pop()


class Head(nn.Module):
    def __init__(self, arch, d_l, n_cells, d_g=1024, d_s=11, d_out=CHUNK * 7, h=512, p=0.1, d_p=64, d_lp=128):
        super().__init__()
        self.arch = arch
        if arch == "grid":
            self.P, self.L = nn.Linear(d_g, d_p), nn.Linear(d_l, d_lp)
            d_in = n_cells * d_p + d_lp + d_s
        else:
            d_in = d_l + d_s
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, d_out))

    def forward(self, l, s, g=None):
        parts = [torch.relu(self.P(g)).flatten(1), torch.relu(self.L(l))] if self.arch == "grid" else [l]
        return self.net(torch.cat(parts + [s], 1))


def batches(n, w, rng, bs=256):
    idx = rng.choice(n, size=n, replace=True, p=w / w.sum())
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def fit(arch, L, S, G, Y, w, tr, va=None, epochs=40, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Head(arch, L.shape[1], G.shape[1], G.shape[2])
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    T = lambda a, i: torch.from_numpy(np.asarray(a[i], dtype=np.float32))  # noqa: E731
    best = (float("inf"), epochs - 1)
    for ep in range(epochs):
        m.train()
        for b in batches(len(tr), w[tr], rng):
            i = tr[b]
            loss = ((m(T(L, i), T(S, i), T(G, i) if arch == "grid" else None) - T(Y, i)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if va is not None:
            m.eval()
            with torch.no_grad():
                v = sum(float(((m(T(L, va[j:j + 512]), T(S, va[j:j + 512]), T(G, va[j:j + 512]) if arch == "grid" else None)
                                 - T(Y, va[j:j + 512])) ** 2).sum()) for j in range(0, len(va), 512)) / (len(va) * Y.shape[1])
            if v < best[0]:
                best = (v, ep)
    return m, best


def standardize(X, idx):
    mu = np.asarray(X[idx], dtype=np.float32).reshape(-1, X.shape[-1]).mean(0)
    sd = np.asarray(X[idx], dtype=np.float32).reshape(-1, X.shape[-1]).std(0) + 1e-6
    return mu, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", default="models/lite_head.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--arch", default="auto", choices=["auto", "grid", "pool"])
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="頭の初期値と順番の乱数(検証の分け方は固定)。アンサンブル用に別の種で作る")
    a = ap.parse_args()
    torch.set_num_threads(4)
    F, G, S8, A, K, E, f_names, backend = load(a.data)
    S = state_feats(S8)
    a_mu, a_sd = A.reshape(-1, 7).mean(0), A.reshape(-1, 7).std(0) + 1e-6
    Y = ((A - a_mu) / a_sd).reshape(len(A), -1).astype(np.float32)
    cnt = np.bincount(K, minlength=10).astype(np.float64)
    w = np.ones(len(K)) if a.no_balance else 1.0 / cnt[K]
    rng = np.random.default_rng(0)
    eps = np.unique(E)
    val_eps = set(rng.choice(eps, size=max(1, len(eps) // 10), replace=False).tolist())
    va_mask = np.array([e in val_eps for e in E])
    tr, va = np.where(~va_mask)[0], np.where(va_mask)[0]
    print(f"サンプル {len(F)}・試行 {len(eps)}(検証 {len(val_eps)} 試行)・動作ごと {cnt.astype(int).tolist()}", flush=True)
    s_mu, s_sd = standardize(S, tr)
    Sn = ((S - s_mu) / s_sd).astype(np.float32)
    g_mu, g_sd = standardize(G, tr[::4])
    Gn = ((G.astype(np.float32) - g_mu) / g_sd).astype(np.float16)
    cands = [c for c in CANDS if a.arch in ("auto", c[0]) and all(n in f_names for n in c[1])]
    print(f"土台 {backend}・特徴 {f_names}・マス {G.shape[1:]}・候補 {len(cands)}", flush=True)
    with Timer("train", f"軽い手の行動の頭 {os.path.basename(a.out)}", samples=int(len(F)), episodes_data=int(len(eps))) as tm:
        res = []
        for arch, spec in cands:
            Lr = np.concatenate([F[:, f_names.index(n)] for n in spec], axis=1).astype(np.float32)
            l_mu, l_sd = standardize(Lr, tr)
            Ln = (Lr - l_mu) / l_sd
            t = time.time()
            _, (v, ep) = fit(arch, Ln, Sn, Gn, Y, w, tr, va, epochs=a.epochs, seed=a.seed)
            res.append((v, arch, spec, ep))
            print(f"  {arch} {'+'.join(spec):36s} 検証の損失 {v:.4f}(最良 {ep + 1} 周目、{time.time() - t:.0f} 秒)", flush=True)
        v, arch, spec, ep = min(res, key=lambda r: r[0])
        Lr = np.concatenate([F[:, f_names.index(n)] for n in spec], axis=1).astype(np.float32)
        allidx = np.arange(len(F))
        l_mu, l_sd = standardize(Lr, allidx)
        m, _ = fit(arch, (Lr - l_mu) / l_sd, Sn, Gn, Y, w, allidx, epochs=ep + 1, seed=a.seed)
        lin = [l for l in m.net if isinstance(l, nn.Linear)]
        out = {f"W{i}": l.weight.detach().numpy().T.astype(np.float32) for i, l in enumerate(lin)}
        out.update({f"b{i}": l.bias.detach().numpy().astype(np.float32) for i, l in enumerate(lin)})
        if arch == "grid":
            out.update(P_W=m.P.weight.detach().numpy().T.astype(np.float32), P_b=m.P.bias.detach().numpy().astype(np.float32),
                       L_W=m.L.weight.detach().numpy().T.astype(np.float32), L_b=m.L.bias.detach().numpy().astype(np.float32),
                       g_mu=g_mu, g_sd=g_sd)
        upto = max([int(n.split("_")[0][1:]) for n in spec if n.startswith("L")] + ([BACKEND_UPTO.get(backend, 0)] if arch == "grid" else []) + [0])
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        np.savez(a.out, **out, arch=arch, n_layers=len(lin), s_mu=s_mu, s_sd=s_sd, l_mu=l_mu, l_sd=l_sd,
                 a_mu=a_mu.astype(np.float32), a_sd=a_sd.astype(np.float32), chunk=CHUNK, feat_names=np.array(spec), upto=upto,
                 val_loss=v, data=np.array(a.data), backend=backend, seed=a.seed)
        size = os.path.getsize(a.out) / 1e6
        print(f"→ {backend} {arch} {'+'.join(spec)}(検証の損失 {v:.4f}、{ep + 1} 周)、{upto} 層目まで、頭 {size:.1f} MB → {a.out}")
        tm.info.update(backend=backend, arch=arch, spec="+".join(spec), val_loss=round(v, 4), head_mb=round(size, 1), train_steps=0)


if __name__ == "__main__":
    main()
