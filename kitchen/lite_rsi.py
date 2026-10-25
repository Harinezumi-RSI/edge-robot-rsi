"""軽い手のループ 1 周: 練習 → (採点 or 先生) → データ → 頭の学習(数分)→ 評価。.venv で実行。

  python kitchen/lite_rsi.py --tag l1 --mode self   --prev models/lite/l0.npz --pool data/lite/r0.npz
  python kitchen/lite_rsi.py --tag d1 --mode dagger --prev models/lite/l0.npz --pool data/lite/r0.npz

mode:
  self   : 改造 Jev が練習の画像から成功を判定し、τ 以上の試行だけ自分の行動を学ぶ(先生なし・シミュレータの正解は使わない)
  dagger : 練習で行った場面すべてに先生(SmolVLA)が正しい 8 step をつける(先生あり)
  both   : 両方
  explore: 自己進化(LLM の RL の GRPO・ReST と同じ形、先生なし)。同じ場面を揺らぎつきで --samples 回ずつ試し、改造 Jev の採点で
           成功と失敗が混ざった組の成功だけを、組の成功率が低いほど重く(重み 1 + round(2(1 - 組の成功率)) 回)学び直す。
           揺らぎが新しい動きを持ち込み、自分の採点が選ぶ(自分の成功をなぞるだけの self は新しい情報が入らず 2 回とも上がらなかった)
評価は練習と重ならない初期状態(既定 0〜9)。練習と評価は台所 8 個を同時に動かす(lite_batch.py)。結果は outputs/lite/rounds.jsonl に 1 行。
"""
import argparse
import json
import os
import subprocess
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kitchen.runlog import Timer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY, PYMLX = os.path.join(ROOT, ".venv/bin/python"), os.path.join(ROOT, ".venv-mlx/bin/python")


def sh(cmd, log):
    with open(log, "a") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        f.flush()
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def rows(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", default="self", choices=["self", "dagger", "both", "explore"])
    ap.add_argument("--explore", type=float, default=0.2, help="explore: 塊ごとの揺らぎ N(0, σ)")
    ap.add_argument("--samples", type=int, default=8, help="explore: 同じ場面を何回ずつ試すか(組の大きさ)")
    ap.add_argument("--prev", required=True, help="練習に使う頭")
    ap.add_argument("--pool", nargs="*", default=[], help="前の周までの学習データ")
    ap.add_argument("--skills", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--practice-inits", default="10-39")
    ap.add_argument("--val-inits", default="0-9")
    ap.add_argument("--sigma", type=float, default=0.0)
    ap.add_argument("--eval-sigma", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--teacher-delta", default="")
    ap.add_argument("--arch", default="auto")
    ap.add_argument("--exec", type=int, default=4)
    ap.add_argument("--dagger-sparse-ok", action="store_true", help="DAgger: 改造 Jev が成功と見た試行は 32 step おきにだけ答えをつける")
    ap.add_argument("--teacher", default="smolvla", choices=("smolvla", "oft"), help="DAgger の答えをつける先生(oft は鍵を持ってサーバを立てる)")
    ap.add_argument("--lazy-camera", action="store_true", help="練習と評価でカメラは手が見る step だけ撮る(20 周目)")
    ap.add_argument("--practice-home-first", action="store_true", help="練習はずらした姿勢から手続きでホームへ戻してから(立て直しを学ぶ)")
    ap.add_argument("--eval-also-sigma0", action="store_true", help="ずれなしでも評価する(壊していないか)")
    ap.add_argument("--practice-chain", action="store_true", help="練習は別の動作をやり終えた後の場面から(続けて動く、変わった場面への強さを学ぶ)")
    ap.add_argument("--eval-also-chain", action="store_true", help="続けて動く評価もする")
    ap.add_argument("--reuse-practice", action="store_true", help="同じ tag の練習と採点の記録があれば使い回す(答えつけからやり直す)")
    ap.add_argument("--dagger-every", type=int, default=8, help="DAgger: Jev が成功と見なかった試行に答えをつける間隔(step)")
    a = ap.parse_args()
    tag = a.tag
    import numpy as np
    backend = str(np.load(os.path.join(ROOT, a.prev), allow_pickle=True).get("backend", "qwen08"))
    d, o = f"data/lite_rsi/{tag}", f"outputs/lite/{tag}"
    os.makedirs(os.path.join(ROOT, o), exist_ok=True)
    log = os.path.join(ROOT, f"logs/lite_rsi_{tag}.log")
    skills = [str(s) for s in a.skills]
    s = {"tag": tag, "mode": a.mode, "backend": backend, "lazy_camera": a.lazy_camera, "prev": a.prev, "pool": a.pool, "sigma": a.sigma, "tau": a.tau, "start": time.strftime("%Y-%m-%d %H:%M")}
    with Timer("inner", f"軽い手のループ {tag}({a.mode})", mode=a.mode, sigma=a.sigma, tau=a.tau) as whole:
        t = time.time()
        reuse = a.reuse_practice and bool(rows(os.path.join(ROOT, o, "practice.jsonl")))
        if reuse:
            s["reuse_practice"] = True
        else:
            sh([PY, "kitchen/lite_batch.py", "--skills", *skills, "--inits", a.practice_inits, "--sigma", str(a.sigma),
                "--seed-offset", str(zlib.crc32(tag.encode()) % 100_000), "--record", f"{d}/demos", "--snap", f"{d}/snaps", "--tag", tag,
                "--head", a.prev, "--exec", str(a.exec), "--kind", "practice", "--name", f"軽い手の練習 {tag}", "--out", f"{o}/practice.jsonl"]
               + (["--lazy-camera"] if a.lazy_camera else []) + (["--home-first"] if a.practice_home_first else [])
               + (["--chain"] if a.practice_chain else [])
               + (["--explore", str(a.explore), "--samples", str(a.samples)] if a.mode == "explore" else []), log)
        P = rows(os.path.join(ROOT, o, "practice.jsonl"))
        s.update(practice_min=round((time.time() - t) / 60, 1), attempts=len(P), sim_ok=sum(r["ok"] for r in P))
        new = []
        if a.mode in ("self", "both", "explore") or a.dagger_sparse_ok:
            t = time.time()
            if not (reuse and len(rows(os.path.join(ROOT, o, "practice_judged.jsonl"))) >= len(P)):
                sh([PYMLX, "kitchen/judge_runs.py", f"{o}/practice.jsonl", f"{d}/snaps"], log)
            J = rows(os.path.join(ROOT, o, "practice_judged.jsonl"))
            acc = [r for r in J if r["p_jev"] >= a.tau]
            reps = {id(r): 1 for r in acc}
            if a.mode == "explore":  # 組(動作・初期状態)ごとの成功率で、混ざった組の成功だけを重く
                grp = {}
                for r in J:
                    grp.setdefault((r["skill"], r["init"]), []).append(r["p_jev"] >= a.tau)
                rate = {k: sum(v) / len(v) for k, v in grp.items()}
                acc = [r for r in acc if rate[(r["skill"], r["init"])] < 1.0]
                reps = {id(r): 1 + round(2 * (1 - rate[(r["skill"], r["init"])])) for r in acc}
                s.update(groups=len(grp), groups_mixed=sum(0 < x < 1 for x in rate.values()),
                         groups_all_ok=sum(x == 1 for x in rate.values()), groups_all_ng=sum(x == 0 for x in rate.values()),
                         weighted=sum(reps.values()), explore=a.explore, samples=a.samples)
            with open(os.path.join(ROOT, o, "accept.txt"), "w") as f:
                f.writelines(f"{d}/demos/{r['demo']}\n" * reps[id(r)] for r in acc)
            true = [bool(r["labels"][str(r["skill"])]) for r in J]
            s.update(judge_min=round((time.time() - t) / 60, 1), accepted=len(acc), true_ok=sum(true),
                     accepted_true=sum(bool(r["labels"][str(r["skill"])]) for r in acc))
            if acc and a.mode in ("self", "both", "explore"):
                t = time.time()
                sh([PY, "kitchen/lite_data.py", "--demos", f"{d}/demos", "--accept-list", f"{o}/accept.txt", "--every", "4",
                    "--backend", backend, "--out", f"data/lite/{tag}_self.npz"], log)
                new.append(f"data/lite/{tag}_self.npz")
                s["data_min"] = round((time.time() - t) / 60, 1)
        if a.mode in ("dagger", "both"):
            t = time.time()
            cmd = [PY, "kitchen/lite_dagger.py", "--demos", f"{d}/demos", "--every", str(a.dagger_every), "--batch", "16", "--backend", backend, "--out", f"data/lite/{tag}_dagger.npz"]
            if a.teacher_delta:
                cmd += ["--teacher-delta", a.teacher_delta]
            if a.dagger_sparse_ok:
                cmd += ["--judged", f"{o}/practice_judged.jsonl", "--every-ok", "32", "--tau", str(a.tau)]
            if a.teacher == "oft":
                cmd = [PY, "kitchen/oft_session.py", "--"] + cmd + ["--teacher", "oft"]
            s["teacher"] = a.teacher
            s["dagger_every"] = a.dagger_every
            sh(cmd, log)
            new.append(f"data/lite/{tag}_dagger.npz")
            s["label_min"] = round((time.time() - t) / 60, 1)
        t = time.time()
        head = f"models/lite/{tag}.npz"
        sh([PY, "kitchen/lite_train.py", "--data", *a.pool, *new, "--arch", a.arch, "--out", head], log)
        s.update(train_min=round((time.time() - t) / 60, 1), head=head, data=a.pool + new)
        t = time.time()
        sh([PY, "kitchen/lite_batch.py", "--skills", *[str(x) for x in range(10)], "--inits", a.val_inits, "--sigma", str(a.eval_sigma),
            "--head", head, "--exec", str(a.exec), "--name", f"軽い手の評価 {tag}", "--out", f"{o}/val.jsonl"]
           + (["--lazy-camera"] if a.lazy_camera else []), log)
        V = rows(os.path.join(ROOT, o, "val.jsonl"))
        s.update(eval_min=round((time.time() - t) / 60, 1), val_ok=sum(r["ok"] for r in V), val_n=len(V),
                 val_by_skill={str(k): sum(r["ok"] for r in V if r["skill"] == k) for k in range(10)})
        if a.eval_also_chain:
            t = time.time()
            sh([PY, "kitchen/lite_batch.py", "--skills", *[str(x) for x in range(10)], "--inits", a.val_inits, "--chain",
                "--head", head, "--exec", str(a.exec), "--name", f"軽い手の評価 {tag}(続けて動く)", "--out", f"{o}/val_chain.jsonl"], log)
            VC = rows(os.path.join(ROOT, o, "val_chain.jsonl"))
            s.update(evalc_min=round((time.time() - t) / 60, 1), valc_ok=sum(r["ok"] for r in VC), valc_n=len(VC))
        if a.eval_also_sigma0 and a.eval_sigma > 0:
            t = time.time()
            sh([PY, "kitchen/lite_batch.py", "--skills", *[str(x) for x in range(10)], "--inits", a.val_inits, "--sigma", "0",
                "--head", head, "--exec", str(a.exec), "--name", f"軽い手の評価 {tag}(ずれなし)", "--out", f"{o}/val_s0.jsonl"]
               + (["--lazy-camera"] if a.lazy_camera else []), log)
            V0 = rows(os.path.join(ROOT, o, "val_s0.jsonl"))
            s.update(eval0_min=round((time.time() - t) / 60, 1), val0_ok=sum(r["ok"] for r in V0), val0_n=len(V0))
        whole.info.update({k: v for k, v in s.items() if k not in ("tag", "start", "pool", "data", "val_by_skill")})
    s["total_min"] = round(sum(s.get(k, 0) for k in ("practice_min", "judge_min", "data_min", "label_min", "train_min", "eval_min")), 1)
    with open(os.path.join(ROOT, "outputs/lite/rounds.jsonl"), "a") as f:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
