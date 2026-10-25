"""軽い手で動かす(評価・練習)。.venv で実行、手は lite_server.py に問い合わせる。

  python kitchen/lite_eval.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --out outputs/lite/val.jsonl
  python kitchen/lite_eval.py --skills ... --inits 10-39 --record data/lite_rsi/r1/demos --snap data/lite_rsi/r1/snaps --out ...  # 練習

--exec step ごとに手に聞いて(既定 4)、返ってきた 8 step 分の行動の先頭 --exec 個を実行する。--head で頭を選ぶ(既定 models/lite_head.npz)。
特徴は頭に記録された土台(lite_feats.py)で取り出し、頭の計算はこのプロセスの numpy で行う。
--snap: 試行の後にホームへ戻して画像と 10 個の正解を保存(改造 Jev の採点用)。--record: 状態と行動を全部記録(採点で選ぶ)。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
from lerobot.envs.libero import get_task_init_states  # noqa: E402

from kitchen.bench import mac_lock, parse_range  # noqa: E402
from kitchen.chain_env import KitchenChainEnv  # noqa: E402
from kitchen.lite_common import head_forward  # noqa: E402
from kitchen.lite_feats import extract, obs_state  # noqa: E402
from kitchen.record import sim_state  # noqa: E402
from kitchen.run_chain import go_home  # noqa: E402
from kitchen.runlog import Timer  # noqa: E402
from kitchen.skills import INSTRUCTION  # noqa: E402


def load_head(path):
    h = dict(np.load(path or "models/lite_head.npz", allow_pickle=True))
    h["backend"] = str(h.get("backend", "qwen08"))
    return h


def head_act(h, obs, skill):
    F, names, G = extract(h["backend"], obs, skill, upto=int(h["upto"]) or None)
    return head_forward(h, {n: F[i] for i, n in enumerate(names)}, G, obs_state(obs))


def episode(env, skill, init, sigma, seed, record=False, max_steps=300, head=None, n_exec=4):
    env._init_states = get_task_init_states(env._suite, skill)
    env.init_state_id = init
    obs, _ = env.reset()
    home = (obs["robot_state"]["eef"]["pos"].copy(), obs["robot_state"]["eef"]["quat"].copy())
    env.set_goal(skill)
    if sigma > 0:
        obs = env.perturb_robot(sigma, np.random.default_rng(seed))
    states, actions, ok, t_inf, n_inf = [], [], False, 0.0, 0
    queue = []
    for t in range(max_steps):
        if not queue:
            ts = time.perf_counter()
            queue = list(head_act(head, obs, skill)[:n_exec].astype(np.float32))
            t_inf += time.perf_counter() - ts
            n_inf += 1
        act = np.clip(queue.pop(0), -1, 1)
        if record:
            states.append(sim_state(env))
            actions.append(act)
        obs, _, _, _, info = env.step(act)
        if info["is_success"]:
            ok = True
            break
    return obs, home, {"ok": ok, "steps": t + 1, "sec_per_inference": round(t_inf / max(1, n_inf), 4)}, states, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", nargs="+", type=int, required=True)
    ap.add_argument("--inits", default="0-9")
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--record", default="")
    ap.add_argument("--snap", default="")
    ap.add_argument("--tag", default="lite")
    ap.add_argument("--name", default="")
    ap.add_argument("--kind", default="eval")
    ap.add_argument("--head", default="")
    ap.add_argument("--exec", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    head = load_head(a.head)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    for d in (a.record, a.snap):
        if d:
            os.makedirs(d, exist_ok=True)
    done = {(r["skill"], r["init"]) for r in map(json.loads, open(a.out))} if os.path.exists(a.out) else set()
    env = KitchenChainEnv(start_task=1, init_index=0)
    n_ok = n = 0
    t0 = time.time()
    with Timer(a.kind, a.name or f"軽い手 {os.path.basename(a.out)}", skills=a.skills, inits=a.inits, sigma=a.sigma,
               head=os.path.basename(a.head or "lite_head.npz"), backend=head["backend"], n_exec=a.exec) as tm:
        for s in a.skills:
            todo = [i for i in parse_range(a.inits) if (s, i) not in done]
            if not todo:
                continue
            with mac_lock():  # 鍵は動作ごと(最大 10 試行)に取る。試行ごとだと待ちの 30 秒刻みで遅い
                for i in todo:
                    seed = 10_000 * s + i + a.seed_offset
                    te = time.time()
                    obs, home, r, states, actions = episode(env, s, i, a.sigma, seed, record=bool(a.record),
                                                            max_steps=a.max_steps, head=head, n_exec=a.exec)
                    row = {"skill": s, "init": i, "sigma": a.sigma, "seed": seed, **r, "sec": round(time.time() - te, 1)}
                    if a.snap:
                        go_home(env, obs, home)
                        name = f"s{s}_i{i}_{a.tag}.jpg"
                        imageio.imwrite(os.path.join(a.snap, name), env.hires(width=640, height=480), quality=92)
                        row["img"] = name
                        row["labels"] = {str(g): env.goal_reached(g) for g in INSTRUCTION}
                    if a.record:
                        name = f"s{s}_i{i}_{a.tag}.npz"
                        np.savez_compressed(os.path.join(a.record, name), states=np.array(states, dtype=np.float32),
                                            actions=np.array(actions, dtype=np.float32), skill=s, init=i, sim_ok=r["ok"])
                        row["demo"] = name
                    with open(a.out, "a") as f:
                        f.write(json.dumps(row) + "\n")
                    n_ok += r["ok"]
                    n += 1
                    print(f"動作 {s} 初期状態 {i}: {'○' if r['ok'] else '×'} {r['steps']} step {row['sec']} 秒(推論 {r['sec_per_inference']} 秒/回)", flush=True)
        tm.info.update(episodes=n, ok=n_ok)
    print(f"合計 {n_ok}/{n}  {time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
