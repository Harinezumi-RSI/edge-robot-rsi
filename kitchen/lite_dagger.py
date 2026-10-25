"""軽い手の DAgger: 軽い手が自分で動いて行った場面に、先生(SmolVLA)がその場面からの正しい 8 step をつける。.venv で実行。

  python kitchen/lite_dagger.py --demos data/lite_rsi/d1/demos --every 6 [--teacher-delta models/rsi/xxx.safetensors] [--backend qwen08] --out data/lite/d1.npz

入力の試行(lite_eval.py --record の記録)は成功も失敗も全部使う。外れていく途中の場面こそ、先生の答えが要る。
--judged(改造 Jev の採点つきの練習の記録)を渡すと、Jev が成功と見た試行は --every-ok おき(既定 32)、それ以外は --every おきに答えをつける
(答えつけが 1 周の半分を占めたため。シミュレータの正解は使わない)。
1 サンプル = その場面の描き直し → 軽い手の特徴(F・G)・手先の状態・先生の行動の塊。形は lite_data.py と同じ(lite_train.py にそのまま渡せる)。
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
from kitchen.lite_feats import BACKENDS, extract, obs_state  # noqa: E402
from kitchen.record import restore_obs  # noqa: E402
from kitchen.runlog import Timer  # noqa: E402
from kitchen.skills import INSTRUCTION  # noqa: E402
from kitchen.vla import VLA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", nargs="+", required=True)
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--teacher-delta", default="")
    ap.add_argument("--teacher", default="smolvla", choices=("smolvla", "oft"), help="oft は kitchen/oft_server.py に聞く")
    ap.add_argument("--oft-url", default="http://127.0.0.1:8770")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--backend", default="qwen08", choices=BACKENDS)
    ap.add_argument("--judged", default="", help="practice_judged.jsonl(p_jev と demo の名前)")
    ap.add_argument("--every-ok", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = sorted(f for d in a.demos for f in glob.glob(os.path.join(d, "*.npz")))
    if a.teacher == "oft":
        from kitchen.oft_teacher import OFTTeacher
        vla = OFTTeacher(a.oft_url)
    else:
        vla = VLA()
    if a.teacher_delta:
        from safetensors.torch import load_file
        vla.policy.load_state_dict(load_file(a.teacher_delta), strict=False)
    env = KitchenChainEnv(start_task=1, init_index=0)
    env.reset()
    F, G, S, A, K, E = [], [], [], [], [], []
    t0 = time.time()
    with Timer("label", f"軽い手の DAgger: 先生が答えをつける {os.path.basename(a.out)}", trials=len(files),
               teacher=a.teacher if a.teacher == "oft" else (os.path.basename(a.teacher_delta) or "base")) as tm:
        ok = set()
        if a.judged:
            import json
            ok = {r["demo"] for r in map(json.loads, open(a.judged)) if r.get("p_jev", 0) >= a.tau}
        jobs = []
        for ei, f in enumerate(files):
            z = np.load(f)
            step = a.every_ok if os.path.basename(f) in ok else a.every
            fix = z["fixtures"] if "fixtures" in z else None
            jobs += [(ei, int(z["skill"]), z["states"][t], fix) for t in range(0, len(z["actions"]), step)]
        print(f"答えをつける場面 {len(jobs)}(Jev が成功と見た試行 {len(ok)} 本は {a.every_ok} step おき)", flush=True)
        for j in range(0, len(jobs), a.batch):
            part = jobs[j:j + a.batch]
            obs = [restore_obs(env, st, fx) for _, _, st, fx in part]
            ch = vla.chunk_batch(obs, [INSTRUCTION[k] for _, k, _, _ in part], n=CHUNK, seed=j)
            for (ei, k, _, _), o, c in zip(part, obs, ch):
                fv, names, g = extract(a.backend, o, k)
                F.append(fv)
                G.append(g)
                S.append(obs_state(o))
                A.append(np.clip(c, -1, 1).astype(np.float32))
                K.append(k)
                E.append(ei)
            if (j // a.batch) % 50 == 0:
                print(f"  場面 {len(F)}/{len(jobs)}・{(time.time() - t0) / max(1, len(F)):.3f} 秒/場面", flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        np.savez(a.out, F=np.stack(F), G=np.stack(G), S=np.stack(S), A=np.stack(A), K=np.array(K), E=np.array(E), files=np.array(files),
                 F_names=np.array(names), backend=a.backend)
        tm.info.update(samples=len(F))
    print(f"保存: {a.out}(試行 {len(files)}・場面 {len(F)}・{(time.time() - t0) / 60:.1f} 分)")


if __name__ == "__main__":
    main()
