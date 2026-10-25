"""ホームへ戻す動きが、できた状態を崩すかを測る(VLA を使わない、シミュレータだけ。.venv で実行)。

  python kitchen/test_homing.py outputs/rsi/r0/practice.jsonl data/rsi/r0/demos --release 0 12

練習の記録(状態と行動)から成功の瞬間の状態に戻し、戻し方ごとに「戻した後も目標が満たされているか」を数える。
同じ状態から始めるので戻し方どうしは完全な対応比較になる。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from lerobot.envs.libero import get_task_init_states  # noqa: E402

from kitchen.chain_env import KitchenChainEnv  # noqa: E402
from kitchen.record import restore_obs  # noqa: E402
from kitchen.run_chain import go_home  # noqa: E402
from kitchen.runlog import Timer  # noqa: E402


def main():
    runs, demo_dir = sys.argv[1], sys.argv[2]
    variants = [int(x) for x in sys.argv[sys.argv.index("--release") + 1:]]
    rows = [json.loads(l) for l in open(runs) if json.loads(l)["ok"]]
    env = KitchenChainEnv(start_task=1, init_index=0)
    env.reset()
    res = {v: [] for v in variants}
    with Timer("eval", "ホームへ戻す動きが崩すか(シミュレータだけ、成功の瞬間から)", episodes=len(rows) * len(variants),
               variants=variants) as tm:
        for r in rows:
            z = np.load(os.path.join(demo_dir, r["demo"]))
            states, actions = z["states"], z["actions"]
            env._init_states = get_task_init_states(env._suite, r["skill"])
            env.init_state_id = r["init"]
            obs, _ = env.reset()  # 初期姿勢(ホーム)を得るため
            home = (obs["robot_state"]["eef"]["pos"].copy(), obs["robot_state"]["eef"]["quat"].copy())
            env.set_goal(r["skill"])
            for v in variants:
                restore_obs(env, states[-1])
                env._env.env.robots[0].controller.update(force=True)
                env._env.env.robots[0].controller.reset_goal()
                obs, _, _, _, info = env.step(actions[-1])  # 最後の行動で成功の瞬間へ
                before = env.goal_reached(r["skill"])
                obs, _ = go_home(env, obs, home, release_steps=v)
                res[v].append({"skill": r["skill"], "init": r["init"], "before": before, "after": env.goal_reached(r["skill"])})
        for v in variants:
            rr = [x for x in res[v] if x["before"]]
            tm.info[f"release{v}_broken"] = sum(not x["after"] for x in rr)
            tm.info[f"release{v}_n"] = len(rr)
    for v in variants:
        rr = [x for x in res[v] if x["before"]]
        by = {}
        for x in rr:
            by.setdefault(x["skill"], []).append(not x["after"])
        print(f"release {v:2d} step: 成功の瞬間 {len(rr)} 本のうち戻して崩れた {sum(not x['after'] for x in rr)}  "
              + " ".join(f"{s}:{sum(b)}/{len(b)}" for s, b in sorted(by.items())))
    a, b = variants[0], variants[-1]
    keys = [(x["skill"], x["init"]) for x in res[a]]
    da = {(x["skill"], x["init"]): x for x in res[a]}
    db = {(x["skill"], x["init"]): x for x in res[b]}
    only_a = sum(da[k]["before"] and db[k]["before"] and (not da[k]["after"]) and db[k]["after"] for k in keys)
    only_b = sum(da[k]["before"] and db[k]["before"] and da[k]["after"] and (not db[k]["after"]) for k in keys)
    print(f"対応: release {a} だけ崩れた {only_a} / release {b} だけ崩れた {only_b}")
    json.dump({str(v): res[v] for v in variants}, open(runs.replace(".jsonl", "_homing_test.json"), "w"))


if __name__ == "__main__":
    main()
