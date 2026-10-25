"""家事ロボットの通しの評価: 同じ台所でコーチ(Claude)が声で 3 回続けて頼む「セッション」を回し、頼みごとの成功と、3 回とも成功したセッションを数える。
.venv で実行(頭・目・耳・口は brain_server.py に問い合わせる)。

  python kitchen/e2e_session.py --hand models/lite/d1.npz --inits 0-9 --tag dev      # 開発中(初期状態 0〜9)
  python kitchen/e2e_session.py --hand ... --inits 40-49 --tag final                    # 最終テスト(1 回だけ)

頼み 1 回の流れ: コーチの声(Irodori-TTS)→ 聞き取り(Qwen3-ASR)→ 場面の画像を撮って改造 Jev で済みの目標を読み、頭が手順から引く
→ 手で各動作(動作の間はホームへ戻す)→ 改造 Jev で確かめる(やり直さず、だめなら正直に伝える)→ ロボットの声で返事。
場面は頼みをまたいで残る(前の頼みで点けた火は次の頼みでも点いている)。聞き返したら、コーチが「うん、お願い」と答えた扱いで続ける。

成功の判定(シミュレータの目標の述語、記録のためだけに使う):
  動く頼み: 頼みが終わった時点で、望んだ目標(goals)がすべて満たされている
  動かない頼み(none: できない・お礼など): 何も動かさない
失敗の段: 聞き取り(文字の誤り率を記録)・手順(要る動作 = 頼みの始めにまだ満たされていない目標、と手順が違う)・手(手順は正しいが目標が満たされない)。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from kitchen.bench import mac_lock, parse_range  # noqa: E402
from kitchen.chain_env import KitchenChainEnv, fixture_seed  # noqa: E402
from kitchen.demo import LiteHand, SmolHand, ask  # noqa: E402
from kitchen.run_chain import go_home  # noqa: E402
from kitchen.runlog import Timer  # noqa: E402
from kitchen.voice_bench import cer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HISTORY = os.environ.get("PAI_HISTORY", "1") == "1"  # 会話の流れを頭に渡す(26 周目)
STT_FIX = os.environ.get("PAI_STT_FIX", "0") == "1"  # 聞き取りの後処理(頭が聞き間違いを直す)。意味まで書き換えるので既定は切る(26 周目)


def run_turn(env, obs, home, hand, turn, snap_prefix, tau=0.5, max_steps=300, history=(), retries=0, retry_explore=0.06, seed=0):
    t0 = time.time()
    r = {"text": turn["text"], "goals": turn["goals"], "none": bool(turn.get("none"))}
    sp = ask("/tts", {"text": turn["text"], "who": "coach"})
    heard = ask("/stt", {"wav": sp["wav"], "fix": STT_FIX})
    r.update(heard=heard["text"], stt_cer=round(cer(turn["text"], heard["text"]), 3), sec_stt=heard["sec"])
    scene = snap_prefix + "_scene.jpg"
    imageio.imwrite(scene, env.hires(width=640, height=480), quality=92)
    before = {g: env.goal_reached(g) for g in range(10)}
    p = ask("/plan", {"utterance": heard["text"], "image": os.path.abspath(scene), "history": list(history)})
    r.update(plan=p["plan"], reply=p["reply"], seen_done=p.get("done", []), skipped=p.get("skipped", []), asked=bool(p.get("ask")),
             p_ok=p.get("p_ok"), seen_true=[g for g in range(10) if before[g]])
    need = [g for g in turn["goals"] if not before[g]]
    r["need"] = need
    r["plan_ok"] = (not p["plan"]) if r["none"] else sorted(p["plan"]) == sorted(need)
    steps = []
    for k, s in enumerate(p["plan"]):  # 聞き返しても、コーチが「うん、お願い」と答えた扱いで動く
        for attempt in range(retries + 1):
            if attempt > 0 and hasattr(hand, "set_explore"):  # Jev が失敗と見た → ホームから揺らぎを変えてやり直す
                hand.set_explore(retry_explore, seed=seed + 100 * k + attempt)
            env.set_goal(s)
            hand.reset()
            for t in range(max_steps):
                obs, _, _, _, info = env.step(hand.act(obs, s))
                if info["is_success"]:
                    break
            obs, _ = go_home(env, obs, home)
            snap = f"{snap_prefix}_k{k}" + (f"_r{attempt}" if attempt else "") + ".jpg"
            imageio.imwrite(snap, env.hires(width=640, height=480), quality=92)
            pj = ask("/jev", {"image": os.path.abspath(snap)})["probs"][s]
            steps.append({"skill": s, "steps": t + 1, "sim_ok": env.goal_reached(s), "p_jev": round(pj, 3), **({"attempt": attempt} if retries else {})})
            if pj >= tau:
                break
        if hasattr(hand, "set_explore"):
            hand.set_explore(0.0)
        if pj < tau:  # 既定はやり直さない(同じ状況からの揺らぎなしのやり直しは 0/6 だった)。正直に伝えて止める
            r["gave_up_at"] = k
            break
    r["steps"] = steps
    after = {g: env.goal_reached(g) for g in range(10)}
    r["ok"] = (not steps) if r["none"] else all(after[g] for g in turn["goals"])
    r["extra"] = [s["skill"] for s in steps if s["skill"] not in turn["goals"]]
    r["stage"] = "成功" if r["ok"] else ("手順" if not r["plan_ok"] else "手")
    if r["stage"] == "手順" and r["stt_cer"] > 0.2:
        r["stage"] = "聞き取り"
    ask("/tts", {"text": r["reply"] or "できました。", "who": "robot"})
    r["sec"] = round(time.time() - t0, 1)
    return obs, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="smolvla", help="smolvla か軽い手の頭(「,」で区切るとアンサンブル)")
    ap.add_argument("--inits", default="0-9")
    ap.add_argument("--sessions", default="data/sessions.jsonl")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-action-steps", type=int, default=5)
    ap.add_argument("--retries", type=int, default=0, help="Jev が失敗と見た動作を、ホームから揺らぎを変えてやり直す回数(軽い手)")
    ap.add_argument("--retry-explore", type=float, default=0.06, help="やり直しの揺らぎ(塊ごと N(0, σ))")
    a = ap.parse_args()
    S = [json.loads(l) for l in open(os.path.join(ROOT, a.sessions))]
    inits = parse_range(a.inits)
    d = os.path.join(ROOT, "outputs", "e2e", a.tag)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, "sessions.jsonl")
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    hand = SmolHand(a.n_action_steps) if a.hand == "smolvla" else LiteHand(a.hand)
    with Timer("eval", f"家事ロボットの通し(セッション){a.tag}", hand=os.path.basename(a.hand), inits=a.inits, sessions=len(S),
               **({"retries": a.retries, "retry_explore": a.retry_explore} if a.retries else {})) as tm:
        for si, s in enumerate(S):
            if s["id"] in done:
                continue
            init = inits[si % len(inits)]
            with mac_lock():
                torch.manual_seed(init)
                env = KitchenChainEnv(start_task=1, init_index=init)
                obs, _ = env.reset()
                env.place_fixtures(fixture_seed(si, init))  # 備え付けをセッションごとの種で置く(どの手で測っても同じ場面)
                home = (obs["robot_state"]["eef"]["pos"].copy(), obs["robot_state"]["eef"]["quat"].copy())
                turns, history = [], []
                for ti, turn in enumerate(s["turns"]):
                    obs, r = run_turn(env, obs, home, hand, turn, os.path.join(d, f"{s['id']}_t{ti}"),
                                      history=history if HISTORY else (), retries=a.retries, retry_explore=a.retry_explore,
                                      seed=1000 * si + 10 * ti)
                    turns.append(r)
                    history.append({"utterance": r["heard"], "reply": r["reply"], "plan": r["plan"]})
                    print(f"{s['id']} {ti + 1}「{turn['text']}」→ 聞き取り「{r['heard']}」手順 {r['plan']}(要る {r['need']})"
                          f" {'○' if r['ok'] else '×'} {r['stage']} {r['sec']} 秒", flush=True)
                env.close() if hasattr(env, "close") else None
            row = {"id": s["id"], "init": init, "turns": turns, "ok": all(t["ok"] for t in turns)}
            with open(out, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        R = [json.loads(l) for l in open(out)]
        T = [t for r in R for t in r["turns"]]
        summ = {"頼み": f"{sum(t['ok'] for t in T)}/{len(T)}", "動く頼み": f"{sum(t['ok'] for t in T if not t['none'])}/{sum(not t['none'] for t in T)}",
                "動かない頼み": f"{sum(t['ok'] for t in T if t['none'])}/{sum(t['none'] for t in T)}",
                "セッション(3 回とも成功)": f"{sum(r['ok'] for r in R)}/{len(R)}",
                "失敗の段": {k: sum(t["stage"] == k for t in T) for k in ("聞き取り", "手順", "手")},
                "聞き返し": sum(t["asked"] for t in T), "やらなくてよい動作": sum(len(t["extra"]) for t in T),
                "聞き取りの誤り率の平均": round(float(np.mean([t["stt_cer"] for t in T])), 3),
                "頼み 1 回の秒(中央)": float(np.median([t["sec"] for t in T]))}
        tm.info.update({k: v for k, v in summ.items() if not isinstance(v, dict)})
    json.dump(summ, open(os.path.join(d, "summary.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
