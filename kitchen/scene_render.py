"""見てから察するの評価用: 何もしていない初期の場面を描く(初期状態 0〜4、評価専用)。.venv で実行。

  python kitchen/scene_render.py → data/scene/init_i{i}.jpg と data/scene/init.jsonl(10 目標の正解)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imageio.v2 as imageio  # noqa: E402
from lerobot.envs.libero import get_task_init_states  # noqa: E402

from kitchen.chain_env import KitchenChainEnv  # noqa: E402
from kitchen.skills import INSTRUCTION  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    out = os.path.join(ROOT, "data/scene")
    os.makedirs(out, exist_ok=True)
    env = KitchenChainEnv(start_task=1, init_index=0)
    rows = []
    for i in range(5):
        env._init_states = get_task_init_states(env._suite, 0)
        env.init_state_id = i
        env.reset()
        name = f"init_i{i}.jpg"
        imageio.imwrite(os.path.join(out, name), env.hires(width=640, height=480), quality=92)
        rows.append({"img": name, "init": i, "labels": {str(g): env.goal_reached(g) for g in INSTRUCTION}})
    with open(os.path.join(out, "init.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print([sum(r["labels"].values()) for r in rows], "個の目標が最初から満たされている")
