"""AI(Claude)が回した研究のループの年表(docs/fig/timeline.png)。.venv で実行。

  python kitchen/make_timeline.py

上の段: 外側のループ(仮説 → 実験 → 採否)の各周。色は 採用 / 不採用 / 測定(確かめただけ)/ 進行中。
下の段: その間に Mac で走った計算(ロボットの練習・評価・学習・採点)。
数は docs/rsi_runs.jsonl(各処理が自分で書いた開始・終了の時刻)から数える。周の採否は docs/rsi_log.md を読んで下の表に写した。
"""
import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()

# 周の採否(docs/rsi_log.md から)。番号は記録の「N 周目」
OUTCOME = {1: "採用", 2: "不採用", 3: "採用", 4: "不採用", 5: "不採用", 6: "測定", 7: "採用", 8: "採用", 9: "測定", 10: "測定",
           11: "採用", 12: "不採用", 13: "採用", 14: "進行中", 15: "進行中", 16: "採用", 17: "採用", 18: "採用", 19: "採用", 20: "採用", 21: "測定", 22: "測定", 23: "測定", 24: "不採用", 25: "採用", 26: "測定", 27: "測定",
           28: "測定", 29: "採用", 30: "不採用", 31: "採用", 32: "採用", 33: "採用", 34: "測定", 35: "採用",
           36: "採用", 37: "不採用", 38: "不採用", 39: "採用"}
COLOR = {"採用": "#2e8b57", "不採用": "#9a9a9a", "測定": "#3b6fb6", "進行中": "#e0a020"}
LANES = [("inner", "ミニ RSI の 1 周"), ("practice", "ロールアウト"), ("eval", "評価"), ("train", "学習"), ("label", "教師の行動付与"), ("judge", "採点")]
LANE_COLOR = {"inner": "#c9d8ef", "practice": "#d8e8d0", "eval": "#f1d3b3", "train": "#e7c6e0", "label": "#f3e3a6", "judge": "#cfe7e7"}


def t(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")


def main():
    R = [json.loads(l) for l in open(os.path.join(ROOT, "docs/rsi_runs.jsonl"))]
    last = max(r["end"] for r in R if r["kind"] == "outer")
    R = [r for r in R if r["start"] <= last]  # 研究ループの最後の周まで(その後の他スイートの実行は含めない)
    t0 = min(t(r["start"]) for r in R)
    h = lambda x: (t(x) - t0).total_seconds() / 3600  # noqa: E731
    outer = [r for r in R if r["kind"] == "outer"]
    fig, ax = plt.subplots(figsize=(13, 5.2), dpi=150)
    y = len(LANES) + 1.2
    seen = {}
    for r in sorted(outer, key=lambda r: r["start"]):
        m = re.match(r"(\d+) 周目", r["name"])
        n = int(m.group(1)) if m else 0
        out = OUTCOME.get(n, "測定")
        x0, x1 = h(r["start"]), max(h(r["end"]), h(r["start"]) + 0.15)
        lane = y + (seen.get(n, len(seen)) % 3) * 0.55
        seen.setdefault(n, len(seen))
        ax.barh(lane, x1 - x0, left=x0, height=0.45, color=COLOR[out], edgecolor="white")
        ax.text(x0 + 0.03, lane, str(n), va="center", ha="left", fontsize=7, color="white")
    for i, (k, label) in enumerate(LANES):
        for r in R:
            if r["kind"] == k:
                ax.barh(len(LANES) - i - 1, max(h(r["end"]) - h(r["start"]), 0.02), left=h(r["start"]), height=0.7,
                        color=LANE_COLOR[k], edgecolor="#777777", linewidth=0.3)
    ax.set_yticks(list(range(len(LANES))) + [y + 0.55])
    ax.set_yticklabels([label for _, label in LANES][::-1] + ["研究ループ\n(Claude)"], fontsize=9)
    ax.set_xlabel(f"経過時間(時間、{t0:%m/%d %H:%M} から)")
    counts = {o: sum(OUTCOME.get(n) == o for n in set(seen)) for o in COLOR}
    mins = {k: sum(r["minutes"] for r in R if r["kind"] == k) for k, _ in LANES}
    span = max(h(r["end"]) for r in R)
    ax.set_title(f"Claude が実行した研究ループ: {len(seen)} 周・{span:.0f} 時間(立案・実行・採否は Claude、方針は人)"
                 "  " + "・".join(f"{k} {v}" for k, v in counts.items() if v), fontsize=11)  # 0 の区分は出さない
    for o, c in COLOR.items():
        if counts[o]:
            ax.barh(-5, 0, color=c, label=o)
    ax.legend(loc="lower right", fontsize=8, ncol=4, frameon=False, bbox_to_anchor=(1.0, -0.32))
    ax.set_ylim(-0.6, y + 1.8)
    ax.set_xlim(0, span + 0.3)
    ax.grid(axis="x", color="#eeeeee")
    ax.text(0.0, -0.3, "計算時間の合計: " + "・".join(f"{label} {mins[k] / 60:.1f} 時間" for k, label in LANES if mins[k] > 0),
            transform=ax.transAxes, fontsize=8, color="#555555")
    fig.tight_layout()
    out = os.path.join(ROOT, "docs/fig/timeline.png")
    fig.savefig(out)
    print(out, {o: counts[o] for o in counts}, f"{span:.1f} 時間")


if __name__ == "__main__":
    main()
