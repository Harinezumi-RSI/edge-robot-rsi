"""いまある軽い手の学習データの場面を、別の土台の特徴で取り直す(行動の答え S・A・K・E はそのまま)。.venv で実行。

  python kitchen/lite_refeat.py --src data/lite/r0_dino.npz --every 4 --backend dino8 --out data/lite/r0_dino8.npz
  python kitchen/lite_refeat.py --src data/lite/d1_dagger.npz --every 8 --backend dino8 --out data/lite/d1_dagger_dino8.npz
  python kitchen/lite_refeat.py --src data/lite/d2_dagger.npz --every 8 --judged outputs/lite/d2/practice_judged.jsonl --every-ok 32 --backend dino8 ...

場面の並びは作ったときと同じ規則で作り直し(試行のファイル順 × t = 0, every, 2·every, …)、手先の状態が元のデータと一致するかで確かめる。
先生にもう一度答えさせないので、特徴の取り直しは数分で済む。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from kitchen.chain_env import KitchenChainEnv  # noqa: E402
from kitchen.lite_feats import BACKENDS, extract_batch, obs_state  # noqa: E402
from kitchen.record import restore_obs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--every", type=int, required=True)
    ap.add_argument("--judged", default="")
    ap.add_argument("--every-ok", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--backend", required=True, choices=BACKENDS)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.src, allow_pickle=True)
    files = [str(f) for f in z["files"]]
    ok = {r["demo"] for r in map(json.loads, open(a.judged)) if r.get("p_jev", 0) >= a.tau} if a.judged else set()
    jobs = []
    for ei, f in enumerate(files):
        d = np.load(f)
        step = a.every_ok if os.path.basename(f) in ok else a.every
        fix = d["fixtures"] if "fixtures" in d else None
        jobs += [(ei, int(d["skill"]), d["states"][t], fix) for t in range(0, len(d["actions"]), step)]
    assert len(jobs) == len(z["K"]), f"場面の数が合わない: 作り直し {len(jobs)} vs 元 {len(z['K'])}"
    assert all(j[0] == e and j[1] == k for j, e, k in zip(jobs, z["E"], z["K"])), "場面の並びが合わない"
    env = KitchenChainEnv(start_task=1, init_index=0)
    env.reset()
    G, S, t0 = [], [], time.time()
    for i in range(0, len(jobs), a.batch):
        part = jobs[i:i + a.batch]
        obs = [restore_obs(env, st, fx) for _, _, st, fx in part]
        F, names, g = extract_batch(a.backend, obs, [k for _, k, _, _ in part])
        G.append(g)
        S += [obs_state(o) for o in obs]
        if (i // a.batch) % 100 == 0:
            print(f"  場面 {i + len(part)}/{len(jobs)}・{(time.time() - t0) / (i + len(part)):.3f} 秒/場面", flush=True)
    S = np.stack(S)
    gap = float(np.abs(S - z["S"]).max())
    print(f"手先の状態の最大のずれ(元のデータと) {gap:.2e}")
    assert gap < 1e-3, "場面が元のデータと違う"
    oh = np.zeros((len(jobs), 1, 10), np.float16)
    oh[np.arange(len(jobs)), 0, z["K"]] = 1
    np.savez(a.out, F=oh, G=np.concatenate(G), S=z["S"], A=z["A"], K=z["K"], E=z["E"], files=z["files"], F_names=np.array(names), backend=a.backend)
    print(f"保存: {a.out}(場面 {len(jobs)}・{(time.time() - t0) / 60:.1f} 分)")


if __name__ == "__main__":
    main()
