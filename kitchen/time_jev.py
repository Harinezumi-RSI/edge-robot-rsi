"""改造 Jev の速さ: 全層(+ 語彙への出力)/ 8 層目で打ち切り / ゼロショット(問い 10 回)を、同じ画像で比べる。.venv-mlx で実行。
打ち切った特徴が全層で計算した 8 層目の特徴と一致することも確かめる。

  python kitchen/time_jev.py data/judge_val 20
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from kitchen import brain, jev  # noqa: E402
from kitchen.runlog import Timer  # noqa: E402
from kitchen.skills import CHECK_Q  # noqa: E402


def med(ts):
    return float(np.median(ts))


def main():
    d, n = sys.argv[1], int(sys.argv[2])
    imgs = [json.loads(l)["img"] for l in open(os.path.join(d, "index.jsonl"))][:n]
    jev.features(imgs[0]); jev.features(imgs[0], upto=8); brain.p_yes(CHECK_Q[1], imgs[0])  # ウォームアップ
    t_full, t_cut, t_zero, diff = [], [], [], []
    with Timer("judge", "改造 Jev の速さ(全層 / 8 層で打ち切り / ゼロショット 10 問)", images=n) as tm:
        for im in imgs:
            t = time.perf_counter(); f_full = jev.features(im); t_full.append(time.perf_counter() - t)
            t = time.perf_counter(); f_cut = jev.features(im, upto=8); t_cut.append(time.perf_counter() - t)
            diff.append(float(np.abs(f_full[:2] - f_cut).max() / (np.abs(f_full[:2]).max() + 1e-9)))
            t = time.perf_counter()
            for g in range(10):
                brain.p_yes(CHECK_Q[g], im)
            t_zero.append(time.perf_counter() - t)
        tm.info.update(full_sec=med(t_full), cut8_sec=med(t_cut), zero10_sec=med(t_zero), max_rel_diff=max(diff))
    print(f"画像 {n} 枚の中央値: 全層 {med(t_full):.3f} 秒 / 8 層で打ち切り {med(t_cut):.3f} 秒 / ゼロショット 10 問 {med(t_zero):.2f} 秒")
    print(f"速さの比: 打ち切りは全層の {med(t_full) / med(t_cut):.1f} 倍、ゼロショット 10 問の {med(t_zero) / med(t_cut):.0f} 倍")
    print(f"打ち切った特徴と全層の 8 層目の相対差の最大 {max(diff):.2e}")


if __name__ == "__main__":
    main()
