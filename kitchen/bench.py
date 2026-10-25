"""実行の物差し: 単独の動作と、続けて実行する手順の成功率を初期状態ごとに測る(.venv で実行)。

  python kitchen/bench.py --plans 1 7 1,7 4,7 --inits 0-4 --out outputs/bench/x.jsonl [--n-action-steps 5]

1 行 = (手順, 初期状態) の結果。同じ (手順, 初期状態) は何度回しても同じ行動列になる(決定的)ので、
条件を変えた比較は対応のある比較で読む。途中で止めても、済んだ行は飛ばして続きから回る。
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kitchen.chain_env import KitchenChainEnv  # noqa: E402
from kitchen.run_chain import run_plan  # noqa: E402
from kitchen.vla import VLA  # noqa: E402


# Mac を他の重い評価と取り合わないための鍵(mkdir の原子性で排他)。1 回分(手順 × 初期状態)ごとに取って放すので、
# 相手を待たせるのは長くても 1 回分だけ。PAI_MAC_LOCK が空なら鍵を使わない。
LOCK = os.environ.get("PAI_MAC_LOCK", "")


YIELD_MARK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", ".last_yield")


def _other_job_waiting_possible():
    """鍵を共用するほかの重い処理(環境変数 PAI_YIELD_TO にコマンド名の一部)が動いているか。動いていれば鍵を譲る機会を作る。"""
    import subprocess
    name = os.environ.get("PAI_YIELD_TO", "")
    if not name:
        return False
    r = subprocess.run(["ps", "-ax", "-o", "command"], capture_output=True, text=True)
    return name in r.stdout


def _break_stale_lock():
    """持ち主の pai-final のプロセスが死んでいる鍵を外す(強制終了すると __exit__ が走らず鍵が残る。10/23 に通しの評価が 18 分止まった)。
    ほかの処理の鍵(持ち主が paifinal でない)には触れない。"""
    try:
        owner = open(os.path.join(LOCK, "owner")).read().split()
    except OSError:
        return
    if len(owner) == 2 and owner[0] == "paifinal" and owner[1].isdigit():
        try:
            os.kill(int(owner[1]), 0)
        except ProcessLookupError:
            try:
                os.remove(os.path.join(LOCK, "owner"))
                os.rmdir(LOCK)
                print(f"[鍵] 持ち主 {owner[1]} がいない鍵を外した", flush=True)
            except OSError:
                pass
        except PermissionError:
            pass


def _other_waiters(me):
    """ほかの pai-final が鍵を待っているか(待ち札は 30 秒おきに置き直すので、2 分より古い札は数えない)。"""
    import glob
    now = time.time()
    return any(f != me and now - os.path.getmtime(f) < 120 for f in glob.glob(f"{LOCK}.wait.*"))


class mac_lock:
    _last_release = 0.0

    def __enter__(self):
        if LOCK:
            # ほかの処理が優先。その処理は鍵を 5 分おきにしか見に来ないので、こちらが波ごとに取り直し続けると割り込めない。
            # 1 時間に 1 回、鍵を持たずに 305 秒待って、ほかの処理が待っていれば先に取れるようにする。
            try:
                last = os.path.getmtime(YIELD_MARK)
            except OSError:
                last = 0
                open(YIELD_MARK, "w").close()
            if time.time() - last > 3600 and _other_job_waiting_possible():
                print("[鍵] ほかの処理に譲るため 305 秒待つ", flush=True)
                time.sleep(305)
                os.utime(YIELD_MARK, None)
            # pai-final 同士: 返した直後に取り直すと、30 秒おきに見に来る側が永遠に取れない(10/23 に評価が 20 分以上止まった)。
            # 待っている側は「待ち札」を置き、返したばかりの側は待ち札があれば 35 秒譲る。
            me = f"{LOCK}.wait.{os.getpid()}"
            if time.time() - mac_lock._last_release < 5 and _other_waiters(me):
                time.sleep(35)
            while True:
                try:
                    os.mkdir(LOCK)
                    break
                except FileExistsError:
                    _break_stale_lock()
                    open(me, "w").close()
                    time.sleep(30)
            try:
                os.remove(me)
            except FileNotFoundError:
                pass
            open(os.path.join(LOCK, "owner"), "w").write(f"paifinal {os.getpid()}")
        return self

    def __exit__(self, *exc):
        if LOCK:
            try:
                if open(os.path.join(LOCK, "owner")).read() == f"paifinal {os.getpid()}":
                    os.remove(os.path.join(LOCK, "owner"))
                    os.rmdir(LOCK)
            except FileNotFoundError:
                pass
            mac_lock._last_release = time.time()


def parse_range(s):
    a, _, b = s.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", nargs="+", required=True)
    ap.add_argument("--inits", default="0-4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-action-steps", type=int, default=0)
    ap.add_argument("--home", action="store_true", help="動作の間で手先を最初の姿勢へ戻す")
    ap.add_argument("--snap-dir", default="", help="動作の終わりの画像と目標の正解を保存する(確認係の物差し)")
    ap.add_argument("--num-steps", type=int, default=0, help="改造: ノイズ除去の反復回数(既定 10)")
    ap.add_argument("--dtype", default="", help="改造: bf16 / fp16")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l)
            if r.get("home", False) == a.home:
                done.add((tuple(r["plan"]), r["init"]))
    vla = VLA(n_action_steps=a.n_action_steps or None)
    if a.num_steps:
        vla.policy.config.num_steps = a.num_steps
    if a.dtype:
        import torch
        vla.set_dtype({"bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype])
    for ps in a.plans:
        plan = [int(x) for x in ps.split(",")]
        env = None
        for init in parse_range(a.inits):
            if (tuple(plan), init) in done:
                continue
            if env is None:
                env = KitchenChainEnv(start_task=plan[0], init_index=init)
            env.init_state_id = init  # LiberoEnv は reset のたびに番号を進めるので毎回指定する
            t0 = time.time()
            lk = mac_lock().__enter__()
            snap = os.path.join(a.snap_dir, f"p{'-'.join(map(str, plan))}_i{init}_h{int(a.home)}") if a.snap_dir else ""
            try:
                steps = run_plan(env, vla, plan, home=a.home, snap_prefix=snap, seed=init)
            finally:
                lk.__exit__()
            row = {"plan": plan, "init": init, "n_action_steps": vla.n_action_steps, "home": a.home, "seed": init,
                   "num_steps": vla.policy.config.num_steps, "dtype": a.dtype or "as_loaded", "steps": steps,
                   "all_ok": all(s["ok"] for s in steps), "final_kept": all(steps[-1]["kept"]),
                   "sec": round(time.time() - t0, 1)}
            with open(a.out, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{plan} init {init}: {'○' if row['all_ok'] else '×'} "
                  f"{[(s['skill'], s['ok'], s['steps']) for s in steps]} 最後に全部保たれている {row['final_kept']} {row['sec']}s",
                  flush=True)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
