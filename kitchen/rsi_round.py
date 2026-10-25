"""ミニ RSI の 1 周: 練習 → 改造 Jev で自分で採点 → 採点に通った試行で追加学習 → 検証の初期状態で評価。.venv で実行。

  python kitchen/rsi_round.py --round 1 --skills 1 3 [--sigma 0.05] [--tau 0.5]

  練習   : 学習用の初期状態(既定 10〜39)で、いまの手(前の周の差分)で試す。成否にかかわらず全部記録・試行後の画像も保存。
  採点   : 改造 Jev が画像から「やろうとした目標」の達成確率を出す。tau 以上だけ採用(シミュレータの正解は使わない)。
  学習   : これまでの周で採用した試行を全部使い、元の重みから学習し直す(差分だけ保存)。
  評価   : 検証の初期状態(0〜9)でシミュレータの正解を使って成功率を測る(最終テストの 40〜49 は使わない)。
  記録   : 各段の分・本数は docs/rsi_runs.jsonl(種類 inner)、周のまとめは outputs/rsi/rounds.jsonl。
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kitchen.runlog import Timer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY, PYMLX = os.path.join(ROOT, ".venv/bin/python"), os.path.join(ROOT, ".venv-mlx/bin/python")


def sh(args, log):
    with open(log, "a") as f:
        subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def rows(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--skills", nargs="+", required=True)
    ap.add_argument("--practice-inits", default="10-39")
    ap.add_argument("--val-inits", default="0-9")
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--n-action-steps", default="5")
    ap.add_argument("--train-steps", default="1500")
    ap.add_argument("--k", default="4")
    ap.add_argument("--practice-policy", default="prev", choices=["prev", "base"],
                    help="練習に使う手: prev = 前の周の差分 / base = 元の重み(前の周の更新を採らなかったとき)")
    ap.add_argument("--pool-rounds", nargs="*", type=int, default=None,
                    help="学習に使う練習の周(既定: 1〜今の周)。採点は今の τ でやり直す")
    ap.add_argument("--hard-weight", type=float, default=1.0,
                    help="同じ初期状態で採点が割れた(別の試行は失敗)成功の重み。苦労して取った成功を重くする")
    ap.add_argument("--lr", default="5e-5")
    ap.add_argument("--tag", default="", help="置き場の名前(既定 r<周>)。ミニ RIS の周は ris1 など")
    ap.add_argument("--practice-home-first", action="store_true",
                    help="練習はずれた姿勢から手続きでホームへ戻してから VLA(立て直しの動きごと記録)")
    ap.add_argument("--eval-sigma", type=float, default=None, help="評価の初期姿勢のずれ(既定は --sigma)")
    ap.add_argument("--eval-also-sigma0", action="store_true", help="ずれなしでも評価する(壊していないかの確認)")
    ap.add_argument("--pool-tags", nargs="*", default=None, help="学習に使う練習の置き場の名前(既定は今の周だけ、--pool-rounds があればそちら)")
    ap.add_argument("--seed-offset", type=int, default=None)
    ap.add_argument("--prev-delta", default="", help="練習に使う手の差分を明示する(既定は models/rsi/r<周-1>)")
    a = ap.parse_args()
    n = a.round
    tag = a.tag or f"r{n}"
    d = os.path.join("data/rsi", tag)
    o = os.path.join("outputs/rsi", tag)
    eval_sigma = a.sigma if a.eval_sigma is None else a.eval_sigma
    seed_offset = 1000 * n if a.seed_offset is None else a.seed_offset
    os.makedirs(os.path.join(ROOT, o), exist_ok=True)
    log = os.path.join(ROOT, "logs", f"rsi_{tag}.log")
    prev = a.prev_delta or os.path.join("models/rsi", f"r{n - 1}.safetensors")
    delta = ["--delta", prev] if (a.practice_policy == "prev" and n > 1 and os.path.exists(os.path.join(ROOT, prev))) else []
    common = ["--n-envs", "8", "--n-action-steps", a.n_action_steps]
    summary = {"round": n, "tag": tag, "skills": a.skills, "sigma": a.sigma, "practice_home_first": a.practice_home_first,
               "tau": a.tau, "start": time.strftime("%Y-%m-%d %H:%M")}
    base_val = os.path.join(ROOT, "outputs/rsi/r0/val_same.jsonl")
    if n == 1 and not a.tag and not os.path.exists(base_val):
        # 0) 比べる相手: 元の重みのまま、評価とまったく同じ条件(同じ動作の並び・同じ種)で
        sh([PY, "kitchen/fast_rollout.py", "--skills", *a.skills, "--inits", a.val_inits, *common, "--sigma", str(eval_sigma),
            "--snap", "data/snaps/val_rsi", "--tag", "r0", "--kind", "inner", "--name", "RSI 0 周目: 元の重みの評価(比べる相手)",
            "--out", "outputs/rsi/r0/val_same.jsonl"], log)
    with Timer("inner", f"ミニ RSI {tag}", skills=a.skills, sigma=a.sigma, tau=a.tau) as whole:
        # 1) 練習(全部記録)
        t = time.time()
        sh([PY, "kitchen/fast_rollout.py", "--skills", *a.skills, "--inits", a.practice_inits, *common, *delta,
            "--sigma", str(a.sigma), *(["--home-first"] if a.practice_home_first else []),
            "--record", f"{d}/demos", "--record-all", "--snap", f"{d}/snaps", "--tag", tag, "--seed-offset", str(seed_offset),
            "--kind", "inner", "--name", f"RSI {tag}: 練習", "--out", f"{o}/practice.jsonl"], log)
        summary["practice_min"] = round((time.time() - t) / 60, 1)
        # 2) 自分で採点(改造 Jev)
        t = time.time()
        sh([PYMLX, "kitchen/judge_runs.py", f"{o}/practice.jsonl", f"{d}/snaps"], log)
        judged = rows(os.path.join(ROOT, f"{o}/practice_judged.jsonl"))
        acc = [r for r in judged if r["p_jev"] >= a.tau]
        summary.update(judge_min=round((time.time() - t) / 60, 1), attempts=len(judged), accepted=len(acc),
                       accepted_true=sum(r["ok"] for r in acc), sim_ok=sum(r["ok"] for r in judged))
        # 採用した試行の一覧: 使う周の練習を全部集め、今の τ で採点し直す。同じ (動作, 初期状態) で採点が割れたら成功を重く
        pool = a.pool_tags or ([f"r{m}" for m in a.pool_rounds] if a.pool_rounds else
                                ([tag] if a.tag else [f"r{m}" for m in range(1, n + 1)]))
        allj = []
        for m in pool:
            for r in rows(os.path.join(ROOT, f"outputs/rsi/{m}/practice_judged.jsonl")):
                r["_round"] = m
                allj.append(r)
        by = {}
        for r in allj:
            by.setdefault((r["skill"], r["init"]), []).append(r["p_jev"] >= a.tau)
        lines, n_hard = [], 0
        for r in allj:
            if r["p_jev"] < a.tau:
                continue
            hard = not all(by[(r["skill"], r["init"])])
            n_hard += hard
            w = a.hard_weight if hard else 1.0
            demo_path = os.path.join(ROOT, "data/rsi", r["_round"], "demos", r["demo"])
            lines.append(f"{demo_path}\t{w}")
        man = os.path.join(ROOT, o, "train_list.txt")
        open(man, "w").write("\n".join(lines) + "\n")
        summary.update(train_episodes=len(lines), hard_won=n_hard, pool_rounds=pool)
        # 3) 学習(元の重みから、採用した全部で)
        t = time.time()
        out_delta = f"models/rsi/{tag}.safetensors"
        sh([PY, "kitchen/train_vla.py", "--data", man, "--steps", a.train_steps, "--batch", "8", "--k", a.k,
            "--lr", a.lr, "--out", out_delta], log)
        summary["train_min"] = round((time.time() - t) / 60, 1)
        # 4) 評価(検証の初期状態・シミュレータの正解)
        t = time.time()
        sh([PY, "kitchen/fast_rollout.py", "--skills", *a.skills, "--inits", a.val_inits, *common, "--delta", out_delta,
            "--sigma", str(eval_sigma), "--snap", "data/snaps/val_rsi", "--tag", tag, "--kind", "inner",
            "--name", f"RSI {tag}: 評価(σ={eval_sigma})", "--out", f"{o}/val.jsonl"], log)
        val = rows(os.path.join(ROOT, f"{o}/val.jsonl"))
        summary.update(eval_min=round((time.time() - t) / 60, 1), val_ok=sum(r["ok"] for r in val), val_n=len(val),
                       eval_sigma=eval_sigma)
        if a.eval_also_sigma0 and eval_sigma > 0:
            t = time.time()
            sh([PY, "kitchen/fast_rollout.py", "--skills", *a.skills, "--inits", a.val_inits, *common, "--delta", out_delta,
                "--sigma", "0", "--snap", "data/snaps/val_rsi", "--tag", tag + "_s0", "--kind", "inner",
                "--name", f"RSI {tag}: 評価(ずれなし、壊していないか)", "--out", f"{o}/val_s0.jsonl"], log)
            v0 = rows(os.path.join(ROOT, f"{o}/val_s0.jsonl"))
            summary.update(eval0_min=round((time.time() - t) / 60, 1), val0_ok=sum(r["ok"] for r in v0), val0_n=len(v0))
        whole.info.update({k: v for k, v in summary.items() if k not in ("round", "skills", "sigma", "tau", "start")})
    summary["end"] = time.strftime("%Y-%m-%d %H:%M")
    with open(os.path.join(ROOT, "outputs/rsi/rounds.jsonl"), "a") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
