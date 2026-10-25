"""手順(動作の番号の列)を一つの台所で続けて実行し、動作ごとの成否を記録する。

  python kitchen/run_chain.py --plan 1,7 --init 0 [--frames outputs/frames/x]

成否はシミュレータの目標判定(bddl の goal)で測る(評価の正解)。視覚モデルによる確認は別の層で足す。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402

from kitchen.chain_env import KitchenChainEnv  # noqa: E402
from kitchen.skills import INSTRUCTION  # noqa: E402
from kitchen.vla import VLA  # noqa: E402

MAX_STEPS = 300  # LeRobot の LIBERO-Goal の上限と同じ


def run_skill(env, vla, obs, skill, max_steps=MAX_STEPS, on_frame=None):
    env.set_goal(skill)
    vla.reset()
    for t in range(1, max_steps + 1):
        obs, _, _, _, info = env.step(vla.act(obs, INSTRUCTION[skill]))
        if on_frame:
            on_frame(env, skill, t)
        if info["is_success"]:
            return obs, True, t
    return obs, False, max_steps


def _quat_err(q_from, q_to):
    """q_from から q_to への回転(軸×角、xyzw)。"""
    from scipy.spatial.transform import Rotation as R
    return (R.from_quat(q_to) * R.from_quat(q_from).inv()).as_rotvec()


def go_home(env, obs, home, on_frame=None, max_steps=80, release_steps=0):
    """手先を最初の姿勢へ戻す(手続きの小さな動作: 持ち上げる → 水平に戻る → 下ろす、グリッパは開く)。
    VLA は各動作をホームの姿勢から始める手本で学習しているので、次の動作を学習時と同じ始め方にそろえる。
    release_steps: 動く前にその場でグリッパを開いて待つ step 数(つかんだまま持ち上げて物を崩さないため)。"""
    pos0, quat0 = home
    for t in range(release_steps):
        obs, _, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1.0], dtype=np.float32))
        if on_frame:
            on_frame(env, -1, t)
    lift = max(obs["robot_state"]["eef"]["pos"][2], pos0[2]) + 0.05
    for t in range(1, max_steps + 1):
        pos, quat = obs["robot_state"]["eef"]["pos"], obs["robot_state"]["eef"]["quat"]
        far = np.linalg.norm((pos - pos0)[:2]) > 0.02
        if far and pos[2] < lift - 0.01:
            target = np.array([pos[0], pos[1], lift])      # 1) その場で持ち上げる
        elif far:
            target = np.array([pos0[0], pos0[1], lift])    # 2) 高いまま水平に戻る
        else:
            target = pos0                                  # 3) 最初の高さへ下ろす
        dp = np.clip((target - pos) / 0.05 * 1.5, -1, 1)
        dr = np.clip(_quat_err(quat, quat0) / 0.5 * 1.5, -1, 1)
        obs, _, _, _, _ = env.step(np.concatenate([dp, dr, [-1.0]]))
        if on_frame:
            on_frame(env, -1, t)
        if not far and np.linalg.norm(pos - pos0) < 0.01 and np.linalg.norm(_quat_err(quat, quat0)) < 0.05:
            break
    return obs, t


def snapshot(env, path):
    """動作の終わりの画像と、10 個の目標それぞれの正解(シミュレータの判定)を保存する(確認係の物差し)。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.imwrite(path + ".png", env.hires(width=640, height=480))
    json.dump({str(s): env.goal_reached(s) for s in INSTRUCTION}, open(path + ".json", "w"))


def run_plan(env, vla, plan, on_frame=None, home=False, snap_prefix="", seed=None):
    """seed: VLA の行動生成は毎回ノイズを引く(flow matching)ので、比較するときは試行ごとに固定する。"""
    if seed is not None:
        import torch
        torch.manual_seed(seed)
        np.random.seed(seed)
    obs, _ = env.reset()
    home_pose = (obs["robot_state"]["eef"]["pos"].copy(), obs["robot_state"]["eef"]["quat"].copy())
    steps = []
    for k, skill in enumerate(plan):
        t0 = time.time()
        if home and k > 0:
            obs, _ = go_home(env, obs, home_pose, on_frame=on_frame)
        obs, ok, n = run_skill(env, vla, obs, skill, on_frame=on_frame)
        if snap_prefix:
            snapshot(env, f"{snap_prefix}_k{k}_s{skill}")
        steps.append({"skill": skill, "ok": ok, "steps": n, "sec": round(time.time() - t0, 1),
                      # 前の動作の結果が後の動作で崩れていないか(例: 器をしまった後に火をつけて器が動いた)
                      "kept": [env.goal_reached(s) for s in plan[:k + 1]]})
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--frames", default="")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--n-action-steps", type=int, default=0)
    ap.add_argument("--home", action="store_true", help="動作の間で手先を最初の姿勢へ戻す")
    a = ap.parse_args()
    plan = [int(x) for x in a.plan.split(",")]
    env = KitchenChainEnv(start_task=plan[0], init_index=a.init)
    vla = VLA(n_action_steps=a.n_action_steps or None)
    writer = None
    if a.frames:
        os.makedirs(os.path.dirname(a.frames) or ".", exist_ok=True)
        writer = imageio.get_writer(a.frames + ".mp4", fps=20 // a.stride, quality=8)

    def on_frame(e, skill, t):
        if writer and t % a.stride == 0:
            writer.append_data(e.hires(width=640, height=480))

    t0 = time.time()
    steps = run_plan(env, vla, plan, on_frame=on_frame if writer else None, home=a.home, seed=a.init)
    if writer:
        writer.close()
    print(json.dumps({"plan": plan, "init": a.init, "steps": steps, "sec": round(time.time() - t0, 1),
                      "n_action_steps": vla.n_action_steps}, ensure_ascii=False))


if __name__ == "__main__":
    main()
