"""改造 Jev(画像 → 10 個の目標それぞれの達成確率)の学習データを集める(.venv で実行)。

  python kitchen/collect_judge_data.py --skills 0-9 --inits 10-19 --out data/judge

VLA に各動作をやらせ、every ステップごとに画像と 10 個の目標の正解(シミュレータの判定)を保存する。
動作のあとはアームをホームへ戻す動きの間も撮る(実行時の確認はアームをどけた後に撮るので、その見え方も入れる)。
1 行 = 画像 1 枚: {"img", "labels": {"0": bool, ...}, "skill", "init", "t", "phase"}。
物差し(評価用の初期状態 0〜9)と重ならないよう、既定では初期状態 10 以降を使う。
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imageio.v2 as imageio  # noqa: E402

from kitchen.bench import mac_lock, parse_range  # noqa: E402
from kitchen.chain_env import KitchenChainEnv  # noqa: E402
from kitchen.run_chain import go_home, run_skill  # noqa: E402
from kitchen.skills import INSTRUCTION  # noqa: E402
from kitchen.vla import VLA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", default="0-9")
    ap.add_argument("--inits", default="10-19")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out", default="data/judge")
    ap.add_argument("--n-action-steps", type=int, default=-1,
                    help="-1: 速さの比較(outputs/bench/speed_*.jsonl)で成功が落ちていなければ 5、落ちていれば 1")
    a = ap.parse_args()
    if a.n_action_steps < 0:
        def ok(name):
            p = os.path.join("outputs/bench", f"speed_{name}.jsonl")
            return sum(json.loads(l)["all_ok"] for l in open(p)) if os.path.exists(p) else None
        base, n5 = ok("base"), ok("n5")
        a.n_action_steps = 5 if base is not None and n5 is not None and n5 >= base - 1 else 1
        print(f"n_action_steps = {a.n_action_steps}(成功 base {base} / n5 {n5})", flush=True)
    os.makedirs(os.path.join(a.out, "img"), exist_ok=True)
    index = os.path.join(a.out, "index.jsonl")
    done = set()
    if os.path.exists(index):
        done = {(r["skill"], r["init"]) for r in map(json.loads, open(index))}
    vla = VLA(n_action_steps=a.n_action_steps if a.n_action_steps > 1 else None)
    for skill in parse_range(a.skills):
        env = None
        for init in parse_range(a.inits):
            if (skill, init) in done:
                continue
            if env is None:
                env = KitchenChainEnv(start_task=skill, init_index=init)
            env.init_state_id = init
            rows = []

            def snap(e, s, t, phase):
                name = f"s{skill}_i{init}_{phase}_t{t:03d}.jpg"
                imageio.imwrite(os.path.join(a.out, "img", name), e.hires(width=640, height=480), quality=90)
                rows.append({"img": name, "labels": {str(g): e.goal_reached(g) for g in INSTRUCTION},
                             "skill": skill, "init": init, "t": t, "phase": phase})

            with mac_lock():
                import numpy as np
                import torch
                torch.manual_seed(init)
                np.random.seed(init)
                obs, _ = env.reset()
                home = (obs["robot_state"]["eef"]["pos"].copy(), obs["robot_state"]["eef"]["quat"].copy())
                snap(env, skill, 0, "start")
                obs, ok, n = run_skill(env, vla, obs, skill,
                                       on_frame=lambda e, s, t: t % a.every == 0 and snap(e, s, t, "act"))
                obs, _ = go_home(env, obs, home, on_frame=lambda e, s, t: t % a.every == 0 and snap(e, s, t, "home"))
                snap(env, skill, n, "end")
            with open(index, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(f"skill {skill} init {init}: {'成功' if ok else '失敗'} {n} step, 画像 {len(rows)} 枚", flush=True)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
