"""軽い手を台所 N 個で同時に動かす(評価・練習)。.venv で実行。出力の形は lite_eval.py と同じ。

  python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --head models/lite/l0dino.npz --n-envs 8 --out outputs/lite/x/val.jsonl
  練習: --record dir --snap dir --tag t [--sigma 0.1 --seed-offset n]

空いた台所にはすぐ次の試行を入れる。特徴はまとめて取り出す(DINOv2 は 2N 枚を 1 回の順伝播、Qwen はサーバに 1 本ずつ)。
鍵(同じ Mac のほかの重い処理と共用)はおよそ --slice 秒ごとに返す(新しい試行を入れるのを止め、走っている試行を終えてから返し、取り直す)。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402

from kitchen.bench import mac_lock, parse_range  # noqa: E402
from kitchen.chain_env import KitchenChainEnv, check_init, fixture_seed, init_table  # noqa: E402
from kitchen.lite_common import head_forward_batch  # noqa: E402
from kitchen.lite_eval import load_head  # noqa: E402
from kitchen.lite_feats import extract_batch, obs_state  # noqa: E402
from kitchen.record import sim_state  # noqa: E402
from kitchen.run_chain import go_home  # noqa: E402
from kitchen.runlog import Timer  # noqa: E402
from kitchen.skills import INSTRUCTION  # noqa: E402


# 続けて動く評価の「前の動作」: 狙いの動作と別の物を扱い、邪魔しない組み合わせ(狙い: 前)
PRE = {0: 9, 1: 2, 2: 7, 3: 9, 4: 7, 5: 2, 6: 5, 7: 1, 8: 9, 9: 0}


def home_recorded(env, obs, home, rec):
    """go_home と同じ手続きの動き(fast_rollout と同じ)。rec があれば各 step の状態と行動を残す(戻す動きも学ぶため)。"""
    from kitchen.run_chain import _quat_err
    pos0, quat0 = home
    lift = max(obs["robot_state"]["eef"]["pos"][2], pos0[2]) + 0.05
    for _ in range(80):
        pos, quat = obs["robot_state"]["eef"]["pos"], obs["robot_state"]["eef"]["quat"]
        far = np.linalg.norm((pos - pos0)[:2]) > 0.02
        target = np.array([pos[0], pos[1], lift]) if far and pos[2] < lift - 0.01 else (np.array([pos0[0], pos0[1], lift]) if far else pos0)
        if not far and np.linalg.norm(pos - pos0) < 0.01 and np.linalg.norm(_quat_err(quat, quat0)) < 0.05:
            break
        act = np.concatenate([np.clip((target - pos) / 0.05 * 1.5, -1, 1), np.clip(_quat_err(quat, quat0) / 0.5 * 1.5, -1, 1), [-1.0]]).astype(np.float32)
        if rec is not None:
            rec["states"].append(sim_state(env))
            rec["actions"].append(act)
        obs, _, _, _, _ = env.step(act)
    return obs


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
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--slice", type=float, default=180.0)
    ap.add_argument("--lazy-camera", action="store_true", help="(既定になった。互換のため残す)")
    ap.add_argument("--normal-camera", action="store_true", help="カメラを毎 step 撮る(撮る瞬間が環境の来歴で揺れ、同じ試行でも結果が変わる)")
    ap.add_argument("--home-first", action="store_true", help="ずらした姿勢から、まず手続きでホームへ戻してから手が動く(戻す動きも記録する)")
    ap.add_argument("--chain", action="store_true", help="続けて動く: 狙いの動作の前に、邪魔しない別の動作(PRE)を同じ手でやり、ホームへ戻してから狙いの動作。成否と記録は狙いの動作だけ")
    ap.add_argument("--explore", type=float, default=0.0,
                    help="自己進化の揺らぎ: 塊ごとに N(0, σ) を行動に足す(4 step 同じ向き・グリッパには足さない、試行ごとの種)")
    ap.add_argument("--samples", type=int, default=1, help="同じ動作・初期状態を何回試すか(揺らぎの組。GRPO の組と同じ)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.lazy_camera = not a.normal_camera  # 備え付けを種で置き、手が見る step だけ撮れば、同じ試行は何度でも同じ結果になる(24 周目)
    if a.head.endswith(".json"):  # 動作ごとに頭を選ぶ対応表 {"動作": "頭のパス"}
        hmap = {int(k): v for k, v in json.load(open(a.head)).items()}
        cache = {p: load_head(p) for p in set(hmap.values())}
        by_skill = {k: [cache[p]] for k, p in hmap.items()}
        heads = list(cache.values())
    else:
        heads = [load_head(h) for h in (a.head.split(",") if a.head else [""])]  # 「,」で区切ると頭のアンサンブル(行動を平均)
        by_skill = None
    head = heads[0]
    upto = int(head["upto"]) or None
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for d in (a.record, a.snap):
        if d:
            os.makedirs(d, exist_ok=True)
    done = {(r["skill"], r["init"], r.get("sample", 0)) for r in map(json.loads, open(a.out))} if os.path.exists(a.out) else set()
    pending = [(s, i, k) for s in a.skills for i in parse_range(a.inits) for k in range(a.samples) if (s, i, k) not in done]
    envs = [KitchenChainEnv(start_task=1, init_index=0) for _ in range(max(1, min(a.n_envs, len(pending))))]
    if a.lazy_camera:
        for e in envs:
            e.set_lazy_camera(True)
    inits = {s: init_table(envs[0]._suite, s) for s in INSTRUCTION}  # 100 番以降はどの段階でも使っていない新しい初期状態
    slots = [None] * len(envs)
    n_ok = n = 0
    t0 = time.time()
    t_feat = 0.0
    n_feat = 0

    def start(env, s, i, smp=0):
        check_init(inits[s], i)
        env._init_states = inits[s]
        env.init_state_id = i
        obs, _ = env.reset()
        home = (obs["robot_state"]["eef"]["pos"].copy(), obs["robot_state"]["eef"]["quat"].copy())
        env.set_goal(s)
        fix = env.place_fixtures(fixture_seed(s, i))
        seed = 10_000 * s + i + a.seed_offset
        if a.sigma > 0:
            obs = env.perturb_robot(a.sigma, np.random.default_rng(seed))
        sl = {"skill": s, "init": i, "seed": seed, "obs": obs, "home": home, "t": 0, "queue": [], "states": [], "actions": [],
              "t0": time.time(), "fixtures": fix, "sample": smp, "rng": np.random.default_rng(seed * 1000 + smp + 1)}
        if a.home_first:
            sl["obs"] = home_recorded(env, obs, home, sl if a.record else None)
        if a.chain:
            sl["pre"], sl["pre_ok"] = PRE[s], run_pre(env, sl, PRE[s])
            env.set_goal(s)
            sl["obs"] = home_recorded(env, sl["obs"], home, None)
            if a.lazy_camera:
                sl["obs"] = env.observe()
        return sl

    def run_pre(env, sl, g, max_steps=300):
        """前の動作を同じ手で(1 台ずつ。成否は記録だけ)。"""
        env.set_goal(g)
        obs, q = sl["obs"], []
        for _ in range(max_steps):
            if not q:
                o = env.observe() if a.lazy_camera else obs
                F, names, G = extract_batch(heads[0]["backend"], [o], [g], upto)
                hs = by_skill[g] if by_skill is not None else heads
                q = list(np.mean([head_forward_batch(h, {nm: F[:, j] for j, nm in enumerate(names)}, G, obs_state(o)[None]) for h in hs],
                                 axis=0)[0, :a.exec])
            obs, _, _, _, info = env.step(np.clip(np.asarray(q.pop(0), np.float32), -1, 1))
            if info["is_success"]:
                break
        sl["obs"] = obs
        return bool(env.goal_reached(g))

    def finish(k, ok):
        nonlocal n_ok, n
        sl, env = slots[k], envs[k]
        s, i = sl["skill"], sl["init"]
        sfx = f"_k{sl['sample']}" if a.samples > 1 else ""
        row = {"skill": s, "init": i, **({"sample": sl["sample"], "explore": a.explore} if a.samples > 1 or a.explore > 0 else {}),
               "sigma": a.sigma, "seed": sl["seed"], "ok": ok, "steps": sl["t"], "sec": round(time.time() - sl["t0"], 1),
               "n_envs": len(envs), "lazy_camera": a.lazy_camera, "fixtures": np.round(sl["fixtures"][:, :2], 5).tolist(),
               **({"pre": sl["pre"], "pre_ok": sl["pre_ok"]} if "pre" in sl else {})}
        if a.snap:
            go_home(env, sl["obs"], sl["home"])
            name = f"s{s}_i{i}{sfx}_{a.tag}.jpg"
            imageio.imwrite(os.path.join(a.snap, name), env.hires(width=640, height=480), quality=92)
            row["img"] = name
            row["labels"] = {str(g): env.goal_reached(g) for g in INSTRUCTION}
        if a.record:
            name = f"s{s}_i{i}{sfx}_{a.tag}.npz"
            np.savez_compressed(os.path.join(a.record, name), states=np.array(sl["states"], dtype=np.float32),
                                actions=np.array(sl["actions"], dtype=np.float32), skill=s, init=i, sim_ok=ok, fixtures=sl["fixtures"])
            row["demo"] = name
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        n_ok += ok
        n += 1
        print(f"動作 {s} 初期状態 {i}: {'成功' if ok else '失敗'} {sl['t']} step(済み {n}・成功 {n_ok}・{(time.time() - t0) / 60:.1f} 分)", flush=True)
        slots[k] = None

    with Timer(a.kind, a.name or f"軽い手(台所 {len(envs)} 個) {os.path.basename(a.out)}", skills=a.skills, inits=a.inits, sigma=a.sigma,
               head=",".join(os.path.basename(h) for h in a.head.split(",")) if a.head else "lite_head.npz",
               backend=",".join(sorted({h["backend"] for h in heads})), n_heads=len(heads), n_exec=a.exec, n_envs=len(envs),
               lazy_camera=a.lazy_camera, home_first=a.home_first, chain=a.chain) as tm:
        while pending or any(sl is not None for sl in slots):
            with mac_lock():
                t_lock = time.time()
                while True:
                    for k in range(len(envs)):
                        if slots[k] is None and pending and time.time() - t_lock < a.slice:
                            slots[k] = start(envs[k], *pending.pop(0))
                    active = [k for k in range(len(envs)) if slots[k] is not None]
                    if not active:
                        break
                    need = [k for k in active if not slots[k]["queue"]]
                    if need:
                        if a.lazy_camera:
                            for k in need:
                                slots[k]["obs"] = envs[k].observe()
                        ts = time.perf_counter()
                        feats = {}
                        for b in {h["backend"] for h in heads}:  # 土台ごとに 1 回だけ特徴を取る
                            feats[b] = extract_batch(b, [slots[k]["obs"] for k in need], [slots[k]["skill"] for k in need], upto)
                        st = np.stack([obs_state(slots[k]["obs"]) for k in need])

                        def fwd(hs, idx):
                            return np.mean([head_forward_batch(h, {nm: feats[h["backend"]][0][idx, j] for j, nm in enumerate(feats[h["backend"]][1])},
                                                               feats[h["backend"]][2][idx], st[idx]) for h in hs], axis=0)
                        if by_skill is None:
                            A = fwd(heads, np.arange(len(need)))
                        else:
                            A = np.zeros((len(need), 8, 7), np.float32)
                            for sk in {slots[k]["skill"] for k in need}:
                                idx = np.array([j for j, k in enumerate(need) if slots[k]["skill"] == sk])
                                A[idx] = fwd(by_skill[sk], idx)
                        t_feat += time.perf_counter() - ts
                        n_feat += 1
                        for j, k in enumerate(need):
                            q = A[j, :a.exec].astype(np.float32)
                            if a.explore > 0:  # 塊ごとに同じ向きの揺らぎ(グリッパは揺らさない)
                                nz = slots[k]["rng"].normal(0.0, a.explore, q.shape[1]).astype(np.float32)
                                nz[6] = 0.0
                                q = q + nz
                            slots[k]["queue"] = list(q)
                    for k in active:
                        sl = slots[k]
                        act = np.clip(sl["queue"].pop(0), -1, 1)
                        if a.record:
                            sl["states"].append(sim_state(envs[k]))
                            sl["actions"].append(act)
                        sl["obs"], _, _, _, info = envs[k].step(act)
                        sl["t"] += 1
                        if info["is_success"] or sl["t"] >= a.max_steps:
                            finish(k, bool(info["is_success"]))
                    if time.time() - t_lock >= a.slice and all(sl is None for sl in slots):
                        break
        tm.info.update(episodes=n, ok=n_ok, sec_per_batch_inference=round(t_feat / max(1, n_feat), 3))
    print(f"合計 {n_ok}/{n}  {time.time() - t0:.0f} 秒(1 試行あたり {(time.time() - t0) / max(1, n):.1f} 秒、まとめた推論 1 回 {t_feat / max(1, n_feat):.3f} 秒)")


if __name__ == "__main__":
    main()
