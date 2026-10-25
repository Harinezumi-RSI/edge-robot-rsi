"""手(SmolVLA)の推論 1 回にかかる時間を、改造の条件ごとに測る(.venv で実行、他の重い処理が無いときに)。

  python kitchen/profile_vla.py [--reps 10]

条件: ノイズ除去の反復回数(num_steps 10 → 1)× 数値の精度(読み込んだまま / bf16 / fp16)。
推論 1 回 = 画像 2 枚と指示文の読み込み(VLM 16 層)+ 行動 50 step 分の生成(反復ごとに小さな専門家ネット)。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from kitchen.chain_env import KitchenChainEnv  # noqa: E402
from kitchen.skills import INSTRUCTION  # noqa: E402
from kitchen.vla import VLA  # noqa: E402


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def time_infer(vla, obs, instr, reps):
    ts = []
    for _ in range(reps + 2):
        vla.reset()  # 行動のキューを空にして、毎回フルに推論させる
        sync()
        t = time.perf_counter()
        vla.act(obs, instr)
        sync()
        ts.append(time.perf_counter() - t)
    ts = sorted(ts[2:])  # 最初の 2 回はウォームアップ
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="outputs/profile_vla.json")
    a = ap.parse_args()
    env = KitchenChainEnv(start_task=1, init_index=0)
    obs, _ = env.reset()
    vla = VLA()
    instr = INSTRUCTION[1]
    res = []
    for dtype in ("as_loaded", "bf16", "fp16"):  # 半精度は autocast(重みは読み込んだまま)
        if dtype == "bf16":
            vla.set_dtype(torch.bfloat16)
        elif dtype == "fp16":
            vla.set_dtype(torch.float16)
        for k in (10, 5, 3, 2, 1):
            vla.policy.config.num_steps = k
            try:
                t = time_infer(vla, obs, instr, a.reps)
                res.append({"dtype": dtype, "num_steps": k, "sec": round(t, 4)})
                print(f"{dtype:9s} num_steps {k:2d}: {t * 1000:7.1f} ms/推論", flush=True)
            except Exception as e:  # 精度を下げて動かない組み合わせは記録だけ残す
                res.append({"dtype": dtype, "num_steps": k, "error": repr(e)[:200]})
                print(f"{dtype:9s} num_steps {k:2d}: 失敗 {e!r}"[:200], flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
