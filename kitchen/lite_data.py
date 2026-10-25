"""軽い手の学習データ: 成功した試行(状態と行動の記録)を描き直し、軽い手のサーバで特徴にする。.venv で実行。

  python kitchen/lite_data.py --demos data/rsi/r0/demos --every 4 --out data/lite/r0.npz [--accept-list 採用一覧.txt] [--backend qwen08|qwen08v|dino]

1 サンプル = その step の正面とグリッパの画像 2 枚(256×256、上下左右を直した向き)から土台ごとの特徴(F と G、lite_feats.py)、手先の状態 8 次元、
その step から 8 step 分の行動(試行の終わりを越える分は最後の行動で埋める)。
既定はシミュレータの成功(sim_ok)の試行 = 先生(SmolVLA)の成功を写す。--accept-list があればその一覧だけ(自分の採点で選んだもの)。
"""
import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from kitchen.chain_env import KitchenChainEnv  # noqa: E402
from kitchen.lite_common import CHUNK  # noqa: E402
from kitchen.lite_feats import BACKENDS, b64png, extract, obs_images, obs_state, post  # noqa: E402,F401
from kitchen.record import restore_obs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", nargs="+", required=True)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--accept-list", default="")
    ap.add_argument("--backend", default="qwen08", choices=BACKENDS)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = []
    if a.accept_list:
        files = [l.strip().split("\t")[0] for l in open(a.accept_list) if l.strip()]
    else:
        for d in a.demos:
            for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
                z = np.load(f)
                if "sim_ok" not in z or bool(z["sim_ok"]):
                    files.append(f)
    env = KitchenChainEnv(start_task=1, init_index=0)
    env.reset()
    F, G, S, A, K, E = [], [], [], [], [], []
    t0 = time.time()
    for ei, f in enumerate(files):
        z = np.load(f)
        states, actions, skill = z["states"], z["actions"], int(z["skill"])
        fix = z["fixtures"] if "fixtures" in z else None  # 備え付けの置き場所(記録にあれば戻す。古い記録には無い)
        for t in range(0, len(actions), a.every):
            obs = restore_obs(env, states[t], fix)
            fv, names, g = extract(a.backend, obs, skill)
            F.append(fv)
            G.append(g)
            S.append(obs_state(obs))
            ch = actions[t:t + CHUNK]
            if len(ch) < CHUNK:
                ch = np.concatenate([ch, np.repeat(ch[-1:], CHUNK - len(ch), 0)])
            A.append(ch.astype(np.float32))
            K.append(skill)
            E.append(ei)
        if ei % 20 == 0:
            print(f"  試行 {ei + 1}/{len(files)}・サンプル {len(F)}・{(time.time() - t0) / max(1, len(F)):.3f} 秒/枚", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez(a.out, F=np.stack(F), G=np.stack(G), S=np.stack(S), A=np.stack(A), K=np.array(K), E=np.array(E), files=np.array(files),
             F_names=np.array(names), backend=a.backend)
    print(f"保存: {a.out}(試行 {len(files)}・サンプル {len(F)}・{(time.time() - t0) / 60:.1f} 分)")


if __name__ == "__main__":
    main()
