"""軽い手の共通部品(.venv と .venv-mlx の両方から読む。numpy だけ)。

特徴: 0.8B の 6/12/18 層目の「最後のトークン」「画像トークンの平均」と、12 層目の画像トークンを 4×4 のマスに平均したもの
(画像 1 枚 8×8 トークン → 16 マス、正面とグリッパで 32 マス)。平均だけだと物の位置が消えるのでマスも持つ。
"""
import numpy as np

LAYERS = (6, 12, 18)
NAMES = [f"L{L}_{p}" for L in LAYERS for p in ("末尾", "画像平均")]
GRID_LAYER = 12
GRID = 4
CHUNK = 8


def rodrigues(rv):
    th = np.linalg.norm(rv, axis=-1, keepdims=True)
    k = rv / np.maximum(th, 1e-8)
    K = np.zeros(rv.shape[:-1] + (3, 3), dtype=np.float32)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return np.eye(3, dtype=np.float32) + s * K + (1 - c) * (K @ K)


def state_feats(S):
    """[N, 8](位置・回転ベクトル・指 2 本)→ [N, 11](位置・回転行列の 2 列・指)。
    下向きの手先は回転角が π 付近で、回転ベクトルの向きが跳ぶ(同じ姿勢が正反対の値になる)ので行列にする。"""
    S = np.asarray(S, dtype=np.float32)
    Rm = rodrigues(S[:, 3:6])
    return np.concatenate([S[:, :3], Rm[:, :, 0], Rm[:, :, 1], S[:, 6:8]], axis=1).astype(np.float32)


def pool_grid(tokens, h, w, g=GRID):
    """画像 1 枚の [h*w, D](行ごとの並び)→ [g*g, D]"""
    D = tokens.shape[-1]
    return tokens.reshape(g, h // g, g, w // g, D).mean(axis=(1, 3)).reshape(g * g, D)


def head_forward_batch(h, feats, grid, state8):
    """h: 頭(npz の中身)・feats: {名前: [N, d]}・grid: [N, マス, 次元] か None・state8: [N, 8] → 行動 [N, CHUNK, 7]"""
    state8 = np.asarray(state8, np.float32)
    N = len(state8)
    s = (state_feats(state8) - h["s_mu"]) / h["s_sd"]
    l = (np.concatenate([np.asarray(feats[str(n)], np.float32).reshape(N, -1) for n in h["feat_names"]], axis=1) - h["l_mu"]) / h["l_sd"]
    if str(h["arch"]) == "grid":
        g = (np.asarray(grid, np.float32) - h["g_mu"]) / h["g_sd"]
        parts = [np.maximum(g @ h["P_W"] + h["P_b"], 0).reshape(N, -1), np.maximum(l @ h["L_W"] + h["L_b"], 0)]
    else:
        parts = [l]
    x = np.concatenate(parts + [s], 1)
    n = int(h["n_layers"])
    for i in range(n):
        x = x @ h[f"W{i}"] + h[f"b{i}"]
        if i < n - 1:
            x = np.maximum(x, 0)
    return x.reshape(N, -1, 7) * h["a_sd"] + h["a_mu"]


def head_forward(h, feats, grid, state8):
    """1 つ分: feats: {名前: [d]}・grid: [マス, 次元] か None・state8: [8] → 行動 [CHUNK, 7]"""
    return head_forward_batch(h, {k: np.asarray(v)[None] for k, v in feats.items()}, None if grid is None else np.asarray(grid)[None],
                              np.asarray(state8, np.float32)[None])[0]
