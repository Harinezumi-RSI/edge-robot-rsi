"""実行器の結果(--snap の画像つき)に、改造 Jev(とゼロショットの判定)の確率を付ける。.venv-mlx で実行。

  python kitchen/judge_runs.py outputs/rsi/r1_practice.jsonl data/rsi/r1/snaps [--zero]

各試行の「やろうとした動作の目標」について:
  p_jev  = 改造 Jev(画像 1 回の読み込みで 10 目標)の確率
  p_zero = ゼロショット(「はい/いいえ」の記号の確率、--zero のときだけ。10 倍遅い)
シミュレータの正解(labels)は採点に使わない。判定がどれだけ当たったかを記録するためだけに並べて出す。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kitchen import brain, jev  # noqa: E402
from kitchen.skills import CHECK_Q  # noqa: E402


def main():
    runs, snap_dir = sys.argv[1], sys.argv[2]
    zero = "--zero" in sys.argv
    out = runs.replace(".jsonl", "_judged.jsonl")
    rows = [json.loads(l) for l in open(runs)]
    done = {json.loads(l)["img"] for l in open(out)} if os.path.exists(out) else set()
    t0 = time.time()
    for r in rows:
        if "img" not in r or r["img"] in done:
            continue
        img = os.path.join(snap_dir, r["img"])
        ts = time.time()
        p = jev.predict(img)
        r["p_jev_all"] = [round(float(x), 4) for x in p]
        r["p_jev"] = r["p_jev_all"][r["skill"]]
        r["sec_jev"] = round(time.time() - ts, 3)
        if zero:
            ts = time.time()
            r["p_zero"] = round(float(brain.p_yes(CHECK_Q[r["skill"]], img)[0]), 4)
            r["sec_zero"] = round(time.time() - ts, 3)
        with open(out, "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    judged = [json.loads(l) for l in open(out)]
    print(f"判定 {len(judged)} 本({time.time() - t0:.0f} 秒)→ {out}")


if __name__ == "__main__":
    main()
