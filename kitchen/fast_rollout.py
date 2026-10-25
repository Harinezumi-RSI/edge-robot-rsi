"""台所を N 個同時に動かし、VLA の推論を 1 回にまとめる実行器(評価・データ集めを速く。.venv で実行)。

  python kitchen/fast_rollout.py --skills 1 7 3 --inits 0-9 --n-envs 8 --out outputs/fast/base.jsonl
  python kitchen/fast_rollout.py --skills 1 --inits 0-9 --sigma 0.1 --out outputs/fast/ris01.jsonl        # ミニ RIS
  python kitchen/fast_rollout.py --skills 1 --inits 10-49 --sigma 0.1 --home-first --record data/ris_demos  # 戻す手本

試行は N 個ずつの「波」で回す。波の中は同じ動作でそろえる(長さが近く、待ちが減る)。
乱数: 波ごとに torch の種を固定(同じ並びなら何度回しても同じ結果)、初期姿勢のずれは試行ごとの種。
--home-first: ずらした姿勢から手続きでホームへ戻してから VLA に渡す(戻す手本づくり・上限の確認用)。
--record: 成功した試行の「状態と行動」を保存(画像は持たない。戻す区間の行動も含む)。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from lerobot.envs.libero import get_task_init_states  # noqa: E402

from kitchen.bench import mac_lock, parse_range  # noqa: E402
from kitchen.chain_env import KitchenChainEnv, fixture_seed  # noqa: E402
from kitchen.record import sim_state  # noqa: E402
from kitchen.run_chain import go_home  # noqa: E402
from kitchen.runlog import Timer  # noqa: E402
from kitchen.skills import INSTRUCTION  # noqa: E402
from kitchen.vla import VLA  # noqa: E402

MAX_STEPS = 300


class BatchRunner:
    def __init__(self, vla, n_envs):
        self.vla = vla
        self.envs = [KitchenChainEnv(start_task=1, init_index=0) for _ in range(n_envs)]
        suite = self.envs[0]._suite
        self.inits = {s: get_task_init_states(suite, s) for s in INSTRUCTION}

    def _start(self, env, ep):
        env._init_states = self.inits[ep["skill"]]
        env.init_state_id = ep["init"]
        obs, _ = env.reset()
        home = (obs["robot_state"]["eef"]["pos"].copy(), obs["robot_state"]["eef"]["quat"].copy())
        env.set_goal(ep["skill"])
        self._home = getattr(self, "_home", {})
        self._home[id(env)] = home
        rec = {"states": [], "actions": [], "fixtures": env.place_fixtures(fixture_seed(ep["skill"], ep["init"]))}
        if ep.get("sigma", 0) > 0:
            obs = env.perturb_robot(ep["sigma"], np.random.default_rng(ep["seed"]))
        if ep.get("home_first"):
            obs = self._home_recorded(env, obs, home, rec if ep.get("record") else None)
        return obs, rec

    def _home_recorded(self, env, obs, home, rec):
        """go_home と同じ動き。記録するときは各 step の状態と行動を残す。"""
        from kitchen.run_chain import _quat_err
        pos0, quat0 = home
        lift = max(obs["robot_state"]["eef"]["pos"][2], pos0[2]) + 0.05
        for t in range(80):
            pos, quat = obs["robot_state"]["eef"]["pos"], obs["robot_state"]["eef"]["quat"]
            far = np.linalg.norm((pos - pos0)[:2]) > 0.02
            if far and pos[2] < lift - 0.01:
                target = np.array([pos[0], pos[1], lift])
            elif far:
                target = np.array([pos0[0], pos0[1], lift])
            else:
                target = pos0
            if not far and np.linalg.norm(pos - pos0) < 0.01 and np.linalg.norm(_quat_err(quat, quat0)) < 0.05:
                break
            act = np.concatenate([np.clip((target - pos) / 0.05 * 1.5, -1, 1),
                                  np.clip(_quat_err(quat, quat0) / 0.5 * 1.5, -1, 1), [-1.0]]).astype(np.float32)
            if rec is not None:
                rec["states"].append(sim_state(env))
                rec["actions"].append(act)
            obs, _, _, _, _ = env.step(act)
        return obs

    def snapshot_after(self, env, ep, path_prefix):
        """試行の後、アームをホームへ戻してから(確認係が見るのと同じ見え方)画像と 10 個の目標の正解を保存する。"""
        import imageio.v2 as imageio
        obs = env._format_raw_obs(env._env.env._get_observations(force_update=True))
        self._home_recorded(env, obs, self._home[id(env)], None)
        imageio.imwrite(path_prefix + ".jpg", env.hires(width=640, height=480), quality=92)
        labels = {str(g): env.goal_reached(g) for g in INSTRUCTION}
        return labels

    def run_wave(self, wave, seed, max_steps=MAX_STEPS, snap_dir=""):
        torch.manual_seed(seed)
        envs = self.envs[: len(wave)]
        obs, recs = [], []
        for env, ep in zip(envs, wave):
            o, r = self._start(env, ep)
            obs.append(o)
            recs.append(r)
        self.vla.reset()
        n = len(wave)
        done, ok, steps = [False] * n, [False] * n, [0] * n
        instr = [INSTRUCTION[ep["skill"]] for ep in wave]
        t_inf = 0.0
        for t in range(max_steps):
            ts = time.perf_counter()
            acts = self.vla.act_batch(obs, instr, active=[not d for d in done])
            t_inf += time.perf_counter() - ts
            for i, (env, ep) in enumerate(zip(envs, wave)):
                if done[i]:
                    continue
                if ep.get("record"):
                    recs[i]["states"].append(sim_state(env))
                    recs[i]["actions"].append(acts[i].astype(np.float32))
                obs[i], _, _, _, info = env.step(acts[i])
                steps[i] = t + 1
                if info["is_success"]:
                    done[i] = ok[i] = True
            if all(done):
                break
        out = [{"skill": ep["skill"], "init": ep["init"], "sigma": ep.get("sigma", 0.0), "seed": ep["seed"],
                "home_first": bool(ep.get("home_first")), "ok": ok[i], "steps": steps[i]}
               for i, ep in enumerate(wave)]
        if snap_dir:
            os.makedirs(snap_dir, exist_ok=True)
            for i, (env, ep) in enumerate(zip(envs, wave)):
                name = f"s{ep['skill']}_i{ep['init']}_sg{int(ep.get('sigma', 0) * 100):02d}_{ep.get('tag', '')}"
                out[i]["img"] = name + ".jpg"
                out[i]["labels"] = self.snapshot_after(env, ep, os.path.join(snap_dir, name))
        return out, recs, t_inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", nargs="+", type=int, required=True)
    ap.add_argument("--inits", default="0-9")
    ap.add_argument("--sigma", type=float, default=0.0, help="ミニ RIS: 初期姿勢の関節のずれ [rad]")
    ap.add_argument("--home-first", action="store_true")
    ap.add_argument("--record", default="", help="試行の状態と行動を保存する置き場(既定は成功した試行だけ)")
    ap.add_argument("--record-all", action="store_true",
                    help="成否にかかわらず全部保存(ミニ RSI: 成否は後で改造 Jev が決める。シミュレータの正解は記録用に残すだけ)")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-action-steps", type=int, default=0)
    ap.add_argument("--teacher", default="smolvla", choices=("smolvla", "oft"), help="oft は kitchen/oft_server.py に聞く")
    ap.add_argument("--oft-url", default="http://127.0.0.1:8770")
    ap.add_argument("--num-steps", type=int, default=0)
    ap.add_argument("--delta", default="", help="追加学習した差分の重み")
    ap.add_argument("--snap", default="", help="試行の後(ホームへ戻してから)の画像と 10 個の正解を保存する置き場")
    ap.add_argument("--tag", default="", help="画像の名前に付ける印(条件の区別)")
    ap.add_argument("--seed-offset", type=int, default=0, help="種をずらす(ミニ RSI の周ごとに別の試行にする)")
    ap.add_argument("--kind", default="eval", help="記録の種類(eval / inner など)")
    ap.add_argument("--name", default="", help="記録に残す名前")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.teacher == "oft":
        from kitchen.oft_teacher import OFTTeacher
        vla = OFTTeacher(a.oft_url, exec_k=a.n_action_steps or 8)
        num_steps = None
    else:
        vla = VLA(n_action_steps=a.n_action_steps or None)
        if a.num_steps:
            vla.policy.config.num_steps = a.num_steps
        num_steps = vla.policy.config.num_steps
    if a.delta:
        from safetensors.torch import load_file
        missing = vla.policy.load_state_dict(load_file(a.delta), strict=False)
        print(f"差分を読み込み: {a.delta}(読み込まなかった重み {len(missing.missing_keys)})", flush=True)
    runner = BatchRunner(vla, a.n_envs)
    eps = [{"skill": s, "init": i, "sigma": a.sigma, "seed": 10_000 * s + i + a.seed_offset, "home_first": a.home_first,
            "record": bool(a.record), "tag": a.tag} for s in a.skills for i in parse_range(a.inits)]
    done = set()
    if os.path.exists(a.out):
        done = {(r["skill"], r["init"]) for r in map(json.loads, open(a.out))}
    eps = [e for e in eps if (e["skill"], e["init"]) not in done]
    if a.record:
        os.makedirs(a.record, exist_ok=True)
    t0 = time.time()
    n_ok = n_all = 0
    timer = Timer(a.kind, a.name or f"実行器 {os.path.basename(a.out)}", skills=a.skills, inits=a.inits, sigma=a.sigma,
                  n_envs=a.n_envs, n_action_steps=vla.n_action_steps, num_steps=num_steps, teacher=a.teacher,
                  delta=os.path.basename(a.delta) if a.delta else "").__enter__()
    for w in range(0, len(eps), a.n_envs):
        wave = eps[w: w + a.n_envs]
        with mac_lock():
            tw = time.time()
            res, recs, t_inf = runner.run_wave(wave, seed=wave[0]["seed"], snap_dir=a.snap)
        for r, rec in zip(res, recs):
            r["fixtures"] = np.round(rec["fixtures"][:, :2], 5).tolist()
            r.update({"teacher": a.teacher, "n_action_steps": vla.n_action_steps, "num_steps": num_steps,
                      "delta": os.path.basename(a.delta) if a.delta else ""})
            if a.record and (r["ok"] or a.record_all):
                name = f"s{r['skill']}_i{r['init']}_sg{int(a.sigma * 100):02d}{'_' + a.tag if a.tag else ''}.npz"
                r["demo"] = name
                np.savez_compressed(os.path.join(a.record, name),
                                    states=np.array(rec["states"], dtype=np.float32),
                                    actions=np.array(rec["actions"], dtype=np.float32), skill=r["skill"], init=r["init"],
                                    sim_ok=r["ok"], fixtures=rec["fixtures"])
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
        n_ok += sum(r["ok"] for r in res)
        n_all += len(res)
        print(f"波 {w // a.n_envs + 1}: 成功 {sum(r['ok'] for r in res)}/{len(res)}  {time.time() - tw:.0f} 秒"
              f"(推論 {t_inf:.0f} 秒)  累計 {n_ok}/{n_all}", flush=True)
    timer.info.update(episodes=n_all, ok=n_ok, waves=-(-n_all // a.n_envs))
    timer.__exit__(None, None, None)
    print(f"合計 {n_ok}/{n_all}  {time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
