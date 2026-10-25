"""ベンチ: 1 step あたりの計算時間(横)と LIBERO-Goal の成功(縦)の図 docs/fig/bench.png と表 docs/bench.md。.venv で実行。

  python kitchen/make_bench.py

横軸は「1 回の推論の時間 ÷ 1 回の推論で実行する step 数」。同じ SmolVLA でも毎 step 推論すると 1 step の計算が 5 倍になるので、
1 回の推論の時間で並べると「速さはそのまま成功だけ上がる」ように見えてしまう。LIBERO の制御は 20Hz なので、50ms より左なら止まらずに動ける。
同じ Mac mini・同じ 100 試行(初期状態 0〜9 × 10 動作、ずれなし、備え付けを固定)で測ったもの(●)と、論文の値(□、A100・公式の 1 動作 50 回)を印で分ける。
軽い手の速さは別の実験と並走中に測った控えめな値。
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()


def ok(path):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return None
    R = [json.loads(l) for l in open(p)]
    return (100.0 * sum(r["ok"] for r in R) / len(R), len(R)) if len(R) >= 100 else None


# (名前, 何で測ったか, 1 回の推論 ms, 1 回で実行する step 数, 成功 %, 試行数, 大きさ, メモ)
def rows():
    out = []
    r = ok("outputs/lite/c1o/val.jsonl")  # 最終の手 c1o(先生 OFT + DAgger)。最終テスト(初期状態 40〜49)も同じ 91(outputs/final2/lite.jsonl)
    if r:
        out.append(("軽量方策(本研究)", "Mac mini GPU", 15.5, 4, r[0], r[1], "DINOv2-small 22M + ヘッド 6MB", "学習 1 分以内・人手デモなし・教師 OFT"))
        out.append(("軽量方策(GPU なし)", "Mac mini CPU だけ", 25.9, 4, r[0], r[1], "同上", "同一ヘッド"))
    for path, label, k, note in (("outputs/fast/val_all_fix.jsonl", "SmolVLA(5 step ごとに推論)", 5, "速度優先の設定"),
                                 ("outputs/fast/val_all_n1_fix.jsonl", "SmolVLA(毎 step 推論、原論文の設定)", 1, "原論文と同じ設定")):
        r = ok(path)
        if r:
            out.append((label, "Mac mini GPU", 537.0, k, r[0], r[1], "0.45B", note))
    oft = os.path.join(ROOT, "outputs", "bench_oft_mac.json")
    if os.path.exists(oft):
        o = json.load(open(oft))
        r = ok("outputs/fast/val_all_oft.jsonl")
        if r:  # 公開の重み(LIBERO-Goal)で同じ 100 試行を実測。速さは同じ作りで測った値(重みに依らない)
            out.append(("OpenVLA-OFT(同じ Mac mini)", "Mac mini GPU", o["latency_ms_median"], 8, r[0], r[1], "7.7B", "公開の重み・16GB"))
        else:
            out.append(("OpenVLA-OFT(同じ Mac mini)", "Mac mini GPU で速さを実測・成功は論文", o["latency_ms_median"], 8, 97.9, 500, "7.7B",
                        "重みは乱数で同じ作り(速さは重みに依らない)"))
    out.append(("OpenVLA-OFT", "論文(A100)", 112.0, 8, 97.9, 500, "7B", "表 I・II(手首カメラ・手先の状態あり)"))
    out.append(("OpenVLA", "論文(A100)", 239.6, 1, 79.2, 500, "7B", "表 I・II"))
    return out


def main():
    R = rows()
    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=160)
    ax.axvline(50, color="#c44e52", lw=1.2, ls="--", zorder=1)
    ax.text(50 * 0.93, 100.5, "← 20Hz 制御(LIBERO)の時間制約 50ms", fontsize=8, color="#c44e52", ha="right", va="center")
    for name, where, ms_inf, k, s, n, size, note in R:
        ms = ms_inf / k
        mixed = "実測・成功は論文" in where
        mine = "論文" not in where or mixed
        star = name.startswith("軽量方策(本研究")
        ax.scatter(ms, s, s=320 if star else 130, marker="*" if star else ("D" if mixed else ("o" if mine else "s")),
                   facecolors=("#d9822b" if star else ("none" if mixed else ("#3b6fb6" if mine else "none"))),
                   edgecolors=("#d9822b" if star else "#3b6fb6" if mine else "#666666"), linewidths=1.8, zorder=3)
        if star:
            xy, ha, va = (ms * 0.62, s + 2.0), "left", "bottom"  # 右隣の論文の OFT の説明と重ならないよう左へ
        elif name.startswith("軽量方策"):
            xy, ha, va = (ms * 1.12, s - 2.2), "left", "top"
        else:
            xy, ha, va = ((ms * 1.15, s - 1.2), "left", "top") if ms < 300 else ((ms * 0.87, s - 1.2), "right", "top")
        ax.annotate(f"{name}\n{s:.0f}%・{ms:.1f}ms/step", (ms, s), xytext=xy, fontsize=8.5, ha=ha, va=va, color="#222222",
                    fontweight="bold" if star else "normal")
    ax.set_xscale("log")
    ax.set_xlim(2, 1500)
    ax.set_ylim(55, 102)
    ax.set_xlabel("1 step あたりの計算時間(ms、対数)= 1 回の推論 ÷ 実行する step 数  ← 速い")
    ax.set_ylabel("LIBERO-Goal の成功率(%)")
    ax.grid(color="#eeeeee")
    ax.scatter([], [], marker="o", color="#3b6fb6", label="家庭用 Mac mini で測定(同じ 100 試行)")
    ax.scatter([], [], marker="D", facecolors="none", edgecolors="#3b6fb6", label="速さは Mac mini で実測・成功は論文")
    ax.scatter([], [], marker="s", facecolors="none", edgecolors="#666666", label="論文の値(A100・公式 500 試行)")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.set_title("軽量方策と公開 VLA の比較(左上ほど良い)", fontsize=11)
    fig.text(0.01, 0.005, "注: Mac mini の成功率は初期状態 0〜9 × 10 動作 = 100 試行(備品配置を固定)、論文は公式の 50 × 10 = 500 試行。"
             "1 回の推論で実行する step 数: 軽量方策 4・OFT 8・SmolVLA 5 か 1・OpenVLA 1。", fontsize=6.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(ROOT, "docs/fig/bench.png"))
    lines = ["| 方策 | 測定環境 | 1 回の推論 | 実行する step | 1 step あたり | 1 秒に出せる行動 | LIBERO-Goal | 試行 | 大きさ | メモ |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {n} | {w} | {ms:.1f}ms | {k} | {ms / k:.1f}ms | {1000 * k / ms:.0f} | {s:.0f}% | {c} | {sz} | {note} |" for n, w, ms, k, s, c, sz, note in R]
    open(os.path.join(ROOT, "docs/bench.md"), "w").write("# ベンチ(make_bench.py が作る)\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
