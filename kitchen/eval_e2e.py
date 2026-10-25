"""閉ループの通し評価: ひとこと → 手順 → 動く前の確認 → 各動作 → ホームへ戻す → 改造 Jev で確認 → やり直し。.venv で実行。
頭は brain_server.py に問い合わせる(先に起動しておく)。

  python kitchen/eval_e2e.py --utterances data/utterances_test.jsonl --inits 40-49 --mode full  --out outputs/e2e/full.jsonl
  python kitchen/eval_e2e.py ... --mode naive --out outputs/e2e/naive.jsonl   # 確認・聞き返し・ホームへ戻す・やり直しを全部外す

1 行 = 発話 1 本: 手順・聞き返したか・各動作の成否(一瞬/落ち着いた)・最後に目的の目標が全部満たされているか・時間。
「正しくふるまった」= 手順が正解の一覧に入っていて最後に全部満たされている、または、できない頼み(正解が空)で動かなかった、
または、聞き返した(間違ったまま動かなかった)。間違った手順で動いた/動作に失敗したら誤り。
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imageio.v2 as imageio  # noqa: E402
import torch  # noqa: E402
from lerobot.envs.libero import get_task_init_states  # noqa: E402

from kitchen.bench import mac_lock, parse_range  # noqa: E402
from kitchen.chain_env import KitchenChainEnv  # noqa: E402
from kitchen.run_chain import go_home  # noqa: E402
from kitchen.runlog import Timer  # noqa: E402
from kitchen.skills import INSTRUCTION  # noqa: E402
from kitchen.vla import VLA  # noqa: E402

BRAIN = os.environ.get("PAI_BRAIN", "http://127.0.0.1:8765")


def ask(path, payload):
    req = urllib.request.Request(BRAIN + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def run_one(env, vla, d, init, mode, snap_dir, tau=0.5, retries=1):
    t0 = time.time()
    p = ask("/plan", {"utterance": d["text"]})
    rec = {"id": d["id"], "kind": d["kind"], "init": init, "plan": p["plan"], "p_ok": p.get("p_ok"), "ok_plans": d["ok"],
           "sec_brain": round(p.get("sec_plan", 0) + (p.get("sec_check") or 0), 2), "steps": []}
    rec["plan_correct"] = p["plan"] in d["ok"]
    rec["asked"] = bool(p.get("ask")) and mode == "full"
    if not p["plan"] or rec["asked"]:
        rec["acted"] = False
        rec["final_ok"] = None
        rec["behaved"] = (not p["plan"] and [] in d["ok"]) or rec["asked"]
        rec["sec"] = round(time.time() - t0, 1)
        return rec
    rec["acted"] = True
    plan = p["plan"]
    env._init_states = get_task_init_states(env._suite, plan[0])
    env.init_state_id = init
    torch.manual_seed(init)
    obs, _ = env.reset()
    home = (obs["robot_state"]["eef"]["pos"].copy(), obs["robot_state"]["eef"]["quat"].copy())
    for k, s in enumerate(plan):
        for attempt in range(retries + 1 if mode == "full" else 1):
            if mode == "full" and (k > 0 or attempt > 0):
                obs, _ = go_home(env, obs, home)
            env.set_goal(s)
            vla.reset()
            for t in range(300):
                obs, _, _, _, info = env.step(vla.act(obs, INSTRUCTION[s]))
                if info["is_success"]:
                    break
            st = {"k": k, "skill": s, "attempt": attempt, "sim_ok": bool(info["is_success"]), "steps": t + 1}
            if mode == "full":
                obs, _ = go_home(env, obs, home)
                img = os.path.join(snap_dir, f"{d['id']}_i{init}_k{k}_a{attempt}.jpg")
                imageio.imwrite(img, env.hires(width=640, height=480), quality=92)
                j = ask("/jev", {"image": os.path.abspath(img)})
                st.update(p_jev=j["probs"][s], settled=env.goal_reached(s))
                rec["steps"].append(st)
                if j["probs"][s] >= tau:
                    break
            else:
                st["settled"] = env.goal_reached(s)
                rec["steps"].append(st)
    if mode == "full":
        obs, _ = go_home(env, obs, home)
    rec["final_goals"] = {str(s): env.goal_reached(s) for s in set(plan)}
    rec["final_ok"] = all(rec["final_goals"].values())
    rec["behaved"] = rec["plan_correct"] and rec["final_ok"]
    rec["sec"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utterances", default="data/utterances_test.jsonl")
    ap.add_argument("--inits", default="40-49")
    ap.add_argument("--mode", choices=["full", "naive"], default="full")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = [json.loads(l) for l in open(a.utterances, encoding="utf-8")]
    inits = parse_range(a.inits)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    snap_dir = os.path.join(os.path.dirname(a.out), "snaps_" + a.mode)
    os.makedirs(snap_dir, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    vla = VLA(n_action_steps=5)
    env = KitchenChainEnv(start_task=1, init_index=0)
    with Timer("eval", f"閉ループの通し評価({a.mode}、{os.path.basename(a.utterances)})", utterances=len(data)) as tm:
        for i, d in enumerate(data):
            if d["id"] in done:
                continue
            with mac_lock():
                r = run_one(env, vla, d, inits[i % len(inits)], a.mode, snap_dir)
            with open(a.out, "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"{r['id']} {'○' if r['behaved'] else '×'} 手順 {r['plan']} 正解 {d['ok'][0]} 聞き返し {r['asked']} "
                  f"最後 {r.get('final_ok')} {r['sec']}s  {d['text']}", flush=True)
        rows = [json.loads(l) for l in open(a.out)]
        tm.info.update(behaved=sum(r["behaved"] for r in rows), n=len(rows),
                       wrong_acted=sum(r["acted"] and not r["behaved"] for r in rows))
    print(f"== {a.mode}: 正しくふるまった {sum(r['behaved'] for r in rows)}/{len(rows)}  "
          f"動いて誤り {sum(r['acted'] and not r['behaved'] for r in rows)}  聞き返し {sum(r['asked'] for r in rows)}")


if __name__ == "__main__":
    main()
