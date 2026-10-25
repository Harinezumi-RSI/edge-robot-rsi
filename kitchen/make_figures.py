"""レポートの図を作る(数字は outputs/ と docs/ の記録から読む)。.venv で実行。

  python kitchen/make_figures.py      # docs/fig/*.png
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "fig")
FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_B = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()


def ok(path):
    return sum(json.loads(l)["ok"] for l in open(os.path.join(ROOT, path)))


def fig_rsi():
    labels = ["元の手", "RSI 1 周目", "RSI 2 周目"]
    val = [ok("outputs/ris/val_s10_nohome.jsonl"), ok("outputs/rsi/ris1/val.jsonl"), ok("outputs/rsi/ris2/val.jsonl")]
    fin = [ok("outputs/final/s10_base.jsonl"), None, ok("outputs/final/s10_ris2.jsonl")]
    home = ok("outputs/ris/val_s10_home.jsonl")
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    x = range(3)
    ax.bar([i - 0.18 for i in x], val, width=0.34, color="#4a7fc1", label="検証(初期状態 0〜9)")
    ax.bar([i + 0.18 for i in (0, 2)], [fin[0], fin[2]], width=0.34, color="#e08a2e", label="テスト(40〜49、1 回のみ)")
    ax.axhline(home, color="#888", ls="--", lw=1)
    ax.text(2.45, home + 0.6, f"手続きでホームへ戻す: {home}", ha="right", fontsize=8, color="#555")
    for i, v in enumerate(val):
        ax.text(i - 0.18, v + 0.5, str(v), ha="center", fontsize=9)
    for i in (0, 2):
        ax.text(i + 0.18, fin[i] + 0.5, str(fin[i]), ha="center", fontsize=9)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 40)
    ax.set_ylabel("成功(40 本中)")
    ax.set_title("腕の初期姿勢を約 10cm ずらしたとき(戻す手続きなし)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rsi.png"))


def fig_jev():
    d = json.load(open(os.path.join(ROOT, "data/judge_val/evaltest_Qwen3.5-4B-MLX-4bit.json")))
    j, z = d["jev"]["L8_末尾"], d["zero"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), dpi=200)
    axes[0].bar(["改造 Jev", "問いかけ(10 問)"], [j["auc_mean"], z["auc_mean"]], color=["#2e8b57", "#bbb"])
    axes[0].set_ylim(0.8, 1.04)
    axes[0].set_title("当たり具合(AUC、学習外の画像 100 枚)", fontsize=9)
    for i, v in enumerate([j["auc_mean"], z["auc_mean"]]):
        axes[0].text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].bar(["改造 Jev\n(8 層で打ち切り)", "改造 Jev\n(全層)", "問いかけ\n(10 問)"], [0.35, 0.70, 7.80], color=["#2e8b57", "#7fbf9b", "#bbb"])
    axes[1].set_title("1 枚あたりの時間(秒、Mac)", fontsize=9)
    for i, v in enumerate([0.35, 0.70, 7.80]):
        axes[1].text(i, v + 0.15, f"{v:.2f}", ha="center", fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "jev.png"))


def fig_system():
    W, H = 1600, 640
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    f, fb, fs = ImageFont.truetype(FONT, 26), ImageFont.truetype(FONT_B, 30), ImageFont.truetype(FONT, 22)

    def box(x, y, w, h, title, body, color):
        d.rounded_rectangle([x, y, x + w, y + h], 18, outline=color, width=4, fill=(250, 250, 250))
        d.text((x + 20, y + 14), title, font=fb, fill=color)
        for i, line in enumerate(body):
            d.text((x + 20, y + 62 + i * 34), line, font=fs, fill=(40, 40, 40))

    def arrow(x1, y1, x2, y2, text=""):
        d.line([x1, y1, x2, y2], fill=(90, 90, 90), width=4)
        d.polygon([(x2, y2), (x2 - 16, y2 - 9), (x2 - 16, y2 + 9)] if x2 > x1 else [(x2, y2), (x2 - 9, y2 - 16), (x2 + 9, y2 - 16)],
                  fill=(90, 90, 90))
        if text:
            d.text(((x1 + x2) // 2 - 60, min(y1, y2) - 34), text, font=fs, fill=(90, 90, 90))

    d.text((40, 30), "「寒くなってきたね」", font=fb, fill=(20, 90, 170))
    box(40, 100, 470, 230, "頭(Qwen3.5-4B, 4bit)", ["ひとこと → 手順(生成)", "動く前の確認(Jev 風の確率)", "低ければ聞き返す"], (20, 90, 170))
    box(600, 100, 420, 230, "手(SmolVLA, 公開重み)", ["画像 2 枚 + 指示文 → 腕", "5 step まとめて実行", "動作の間はホームへ戻す"], (200, 120, 0))
    box(1110, 100, 450, 230, "台所(LIBERO-Goal)", ["10 の動作・同じ場面", "目標だけ差し替えて続ける", "Mac の MuJoCo"], (90, 90, 90))
    box(600, 390, 420, 210, "改造 Jev(判定器)", ["Qwen の 8 層目 + 線形の頭", "画像 1 回 → 10 目標の確率", "0.35 秒"], (46, 139, 87))
    box(1110, 390, 450, 210, "ミニ RSI(自分で上達)", ["練習 → Jev で採点 → 学習", "状態だけ記録(数 MB)", "外側のループは Claude"], (150, 60, 160))
    arrow(510, 215, 600, 215, "指示文")
    arrow(1020, 215, 1110, 215, "行動")
    arrow(1335, 330, 1335, 390)
    arrow(1110, 495, 1020, 495)
    d.text((1036, 540), "画像", font=fs, fill=(90, 90, 90))
    arrow(600, 495, 300, 330)
    d.text((60, 560), "できた / だめ → やり直す・正直に伝える", font=fs, fill=(46, 139, 87))
    img.save(os.path.join(OUT, "system.png"))


def fig_system2():
    """第 2 版のしくみ: パソコンの Claude(コーチ)と、家庭用 Mac mini 1 台の家事ロボット(耳・頭・目・手・口、すべて公開モデル)。"""
    import math
    W, H = 1600, 860
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    fb, fs, ft = ImageFont.truetype(FONT_B, 27), ImageFont.truetype(FONT, 21), ImageFont.truetype(FONT_B, 24)

    def box(x, y, w, h, title, body, color, fill=(250, 250, 250)):
        d.rounded_rectangle([x, y, x + w, y + h], 16, outline=color, width=4, fill=fill)
        d.text((x + 16, y + 12), title, font=fb, fill=color)
        for i, line in enumerate(body):
            d.text((x + 16, y + 56 + i * 30), line, font=fs, fill=(40, 40, 40))

    def arrow(pts, text="", at=None, color=(90, 90, 90)):
        d.line(pts, fill=color, width=4, joint="curve")
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        a = math.atan2(y2 - y1, x2 - x1)
        tip = [(x2, y2), (x2 - 18 * math.cos(a) + 10 * math.sin(a), y2 - 18 * math.sin(a) - 10 * math.cos(a)),
               (x2 - 18 * math.cos(a) - 10 * math.sin(a), y2 - 18 * math.sin(a) + 10 * math.cos(a))]
        d.polygon(tip, fill=color)
        if text:
            d.text(at or ((x1 + x2) // 2 + 6, (y1 + y2) // 2 - 28), text, font=fs, fill=color)

    COACH, EAR, BRAIN, EYE, HAND, MOUTH, LOOP = (200, 90, 40), (60, 120, 170), (20, 90, 170), (46, 139, 87), (200, 120, 0), (150, 60, 160), (120, 70, 40)
    box(30, 60, 330, 300, "Claude(外側の RSI)", ["研究ループの実行", "・音声で指示", "・学習の計画", "・仮説 → 実験 → 採否", "  (研究ループ)"], COACH, (255, 247, 240))
    d.rounded_rectangle([400, 20, 1200, 640], 22, outline=(130, 130, 130), width=3)
    d.text((424, 30), "台所ロボット: 家庭用 Mac mini 1 台(公開モデルのみ)", font=ft, fill=(90, 90, 90))
    box(424, 80, 250, 160, "音声認識", ["Qwen3-ASR 0.6B", "混入時に再認識"], EAR)
    box(714, 80, 460, 160, "計画器  Qwen3.5-4B(4bit)", ["意図を推定して計画を生成", "達成済みの目標を除外"], BRAIN)
    box(424, 290, 250, 160, "音声合成", ["Irodori-TTS", "応答を音声で出力"], MOUTH)
    box(714, 290, 300, 160, "状態判定器  改造 Jev", ["Qwen 第 8 層 + 線形", "10 目標を 0.35 秒"], EYE)
    box(714, 470, 460, 150, "行動方策  軽量方策(6MB)", ["DINOv2 + 行動ヘッド、学習 1 分以内", "教師 OpenVLA-OFT から DAgger"], HAND)
    box(1240, 230, 330, 380, "環境", ["LIBERO-Goal", "10 動作・同一場面", "MuJoCo(Mac mini)"], (90, 90, 90))
    arrow([(360, 150), (424, 150)], "音声", at=(375, 112), color=COACH)
    arrow([(674, 160), (714, 160)])
    arrow([(760, 240), (600, 290)], "応答", at=(612, 247), color=BRAIN)
    arrow([(424, 380), (360, 300)], "音声", at=(370, 345), color=MOUTH)
    arrow([(880, 290), (880, 240)], "達成済み", at=(890, 248), color=EYE)
    arrow([(1130, 240), (1130, 470)], "動作番号", at=(1022, 300), color=BRAIN)
    arrow([(1174, 560), (1240, 560)], "行動", at=(1182, 524), color=HAND)
    arrow([(1240, 360), (1014, 400)], "画像", at=(1150, 400), color=EYE)
    box(30, 690, 1540, 150, "自己改善(ミニ RIS・ミニ RSI・DAgger)", [
        "ロールアウト(8 環境を並列)→ 改造 Jev が採点(シミュレータの正解は用いない)→ 教師 OpenVLA-OFT が行動を付与 → ヘッドのみ再学習 → 評価",
        "教師(7B)は学習時のみ同じ Mac mini で実行。改造 Jev が成功と判定した試行は教師への問い合わせを間引く(付与は 4 割)",
        "外側の「仮説 → 実験 → 採否」のループは Claude が立案・実行・採否を担い、人は方針を示す(研究ループの経過の図)"], LOOP, (252, 248, 244))
    arrow([(195, 360), (195, 690)], "学習の計画", at=(205, 600), color=COACH)
    arrow([(940, 690), (940, 620)], "再学習したヘッド", at=(950, 640), color=LOOP)
    img.save(os.path.join(OUT, "system2.png"))


def _count(path):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return None
    R = [json.loads(l) for l in open(p)]
    return sum(r["ok"] for r in R) if len(R) >= 100 else None


def _bars(ax, labels, val, fin):
    x = range(len(labels))
    ax.bar([i - 0.18 for i in x], val, width=0.34, color="#4a7fc1", label="検証(初期状態 0〜9)")
    if any(v is not None for v in fin):
        ax.bar([i + 0.18 for i in x], [v or 0 for v in fin], width=0.34, color="#e08a2e", label="テスト(40〜49、1 回のみ)")
    for i, (v, f) in enumerate(zip(val, fin)):
        ax.text(i - 0.18, v + 1, str(v), ha="center", fontsize=9)
        if f is not None:
            ax.text(i + 0.18, f + 1, str(f), ha="center", fontsize=9)
    ax.set_xticks(list(x), labels, fontsize=8.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("成功数(100 試行中)")


def fig_teacher():
    """手の作り(DINOv2-small + 頭 6MB)はそのままで先生を替える(10 動作 × 初期状態 10 = 100 試行、ずれなし)。"""
    heads = [("l0f", "SmolVLA の行動クローニング"), ("l0o", "OpenVLA-OFT の行動クローニング")]
    val = [_count(f"outputs/lite/{h}/val.jsonl") for h, _ in heads]
    fin = [_count(f"outputs/final2/cmp_{h}.jsonl") for h, _ in heads]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    _bars(ax, [lab for _, lab in heads], val, fin)
    for y, txt in ((97, "教師 OpenVLA-OFT 97(7B・226ms/step)"), (74, "教師 SmolVLA 74(5 step ごと・107ms/step)")):
        ax.axhline(y, color="#888", ls="--", lw=1)
        ax.text(len(heads) - 0.55, y + 1, txt, ha="right", fontsize=7.5, color="#555")
    ax.set_title("軽量方策(3.9ms/step)の構造を固定し、教師のみを交換", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "teacher.png"))


def fig_dagger():
    """前の家事で変わった場面(続けて動く: 狙いの動作の前に別の動作をやってホームへ戻す)。先生を写すだけ vs 自分の練習に先生の答え(DAgger)。"""
    heads = [("l0o", "OFT の行動クローニング"), ("d1o", "+ DAgger\n(標準条件から)"), ("c1o", "+ DAgger\n(連続実行条件から)")]
    val = [_count("outputs/lite/c1o/val_chain.jsonl" if h == "c1o" else f"outputs/lite/{h}_chain/val.jsonl") for h, _ in heads]
    fin = [_count("outputs/final2/lite_chain.jsonl" if h == "c1o" else f"outputs/final2/cmp_{h}_chain.jsonl") for h, _ in heads]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    _bars(ax, [lab for _, lab in heads], val, fin)
    ax.set_title("連続実行条件(前の動作の後に開始)での成功数", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dagger.png"))


def _rate(path):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return None
    R = [json.loads(l) for l in open(p)]
    return 100.0 * sum(r["ok"] for r in R) / len(R) if R else None


def fig_grpo():
    """先生なしの自己進化(GRPO、報酬は改造 Jev だけ)の回数ごとの成功率。どれも学習・選択に使っていない新しい初期状態 400 試行。
    線 = 選ぶ用(140〜179)、点 = 最後の確かめ用(180〜219、220〜299、どれも 1 回だけ)。"""
    pts = [(0, "outputs/lite/dev2_c1o/eval_only_{}.jsonl"), (20, "outputs/lite/dev2_g6_it20/eval_only_{}.jsonl"),
           (40, "outputs/lite/g6b/eval_it20_{}.jsonl"), (60, "outputs/lite/g6b/eval_it40_{}.jsonl"),
           (80, "outputs/lite/g6c/eval_it20_{}.jsonl"), (100, "outputs/lite/g6c/eval_it40_{}.jsonl")]
    tests = [("180〜219", 60, "outputs/lite/test2_c1o/eval_only_{}.jsonl", "outputs/lite/test2_g6b_it40/eval_only_{}.jsonl")]
    sel3 = os.path.join(ROOT, "outputs", "lite")
    for d in sorted(os.listdir(sel3)):
        if d.startswith("test3_") and d != "test3_c1o":
            it = {"g6_it20": 20, "g6b_it20": 40, "g6b_it40": 60, "g6c_it20": 80, "g6c_it40": 100}.get(d[len("test3_"):])
            if it is not None:
                tests.append(("220〜299・800 試行", it, "outputs/lite/test3_c1o/eval_only_{}.jsonl", f"outputs/lite/{d}/eval_only_{{}}.jsonl"))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=200)
    for ax, mode, title in ((axes[0], "chain", "連続実行条件"), (axes[1], "normal", "標準条件")):
        xs, ys = zip(*[(it, _rate(f.format(mode))) for it, f in pts if _rate(f.format(mode)) is not None])
        ax.plot(xs, ys, "-o", color="#4a7fc1", ms=4, label="選択用セット(140〜179)")
        ax.axhline(ys[0], color="#888", ls="--", lw=1)
        ax.text(xs[0] + 2, ys[0] - 1.2, f"出発点 c1o {ys[0]:.1f}%", ha="left", va="top", fontsize=7.5, color="#555")
        for k, (name, it, fc, fg) in enumerate(tests):
            c, g = _rate(fc.format(mode)), _rate(fg.format(mode))
            if c is None or g is None:
                continue
            col = ["#e08a2e", "#2e9e5b"][k % 2]
            ax.plot([it, it], [c, g], color=col, lw=1)
            ax.plot(it, c, "x", color=col, ms=6)
            ax.plot(it, g, "s", color=col, ms=5, label=f"確認用 {name}(□ GRPO・× c1o)")
        for x_, y_ in zip(xs, ys):
            ax.text(x_, y_ + 1.0, f"{y_:.1f}", ha="center", fontsize=7)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("GRPO の反復数(1 反復 = 96 試行)", fontsize=8.5)
        ax.set_ylabel("成功率(%、400 試行)", fontsize=8.5)
        ax.set_ylim(*((58, 81) if mode == "chain" else (82, 100)))
        ax.legend(fontsize=6.5, loc="upper left" if mode == "chain" else "lower right")
    fig.suptitle("教師を用いない自己改善(報酬は改造 Jev の判定のみ)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "grpo.png"))


def fig_session(tag, sid, out_dir=None):
    """家事の通しの 1 セッション(頼み 3 回)を 1 枚に: 頼み(コーチの声)・聞き取り・返事・動いた後の場面・成否。
    画像は e2e_session.py が撮った outputs/e2e/<tag>/<sid>_t<i>_k<k>.jpg(動作の後)/ _scene.jpg(頼まれた時)。"""
    d0 = os.path.join(ROOT, "outputs", "e2e", tag)
    S = {s["id"]: s for s in map(json.loads, open(os.path.join(d0, "sessions.jsonl")))}[sid]
    W, cw = 1600, 520
    fb, fs, fm = ImageFont.truetype(FONT_B, 26), ImageFont.truetype(FONT, 21), ImageFont.truetype(FONT_B, 34)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))

    def wrap(text, font, width):
        out, line = [], ""
        for ch in text:
            if probe.textlength(line + ch, font=font) > width:
                out.append(line)
                line = ch
            else:
                line += ch
        return out + ([line] if line else [])

    texts = [[(ln, col) for txt, col in ((f"コーチ「{t['text']}」", (30, 30, 30)), (f"認識結果「{t['heard']}」", (110, 110, 110)),
                                          (f"ロボット「{t['reply'] or 'できました。'}」", (150, 60, 160))) for ln in wrap(txt, fs, cw)]
             for t in S["turns"]]
    top = 56 + 28 * max(len(x) for x in texts) + 14
    ph = int(cw * 0.75)
    H = top + ph + 64
    img = Image.new("RGB", (W, H), (248, 247, 243))
    d = ImageDraw.Draw(img)
    for i, t in enumerate(S["turns"]):
        x = 20 + i * (cw + 20)
        d.text((x, 16), f"指示 {i + 1}", font=fb, fill=(20, 90, 170))
        for j, (ln, col) in enumerate(texts[i]):
            d.text((x, 56 + 28 * j), ln, font=fs, fill=col)
        ks = sorted(glob.glob(os.path.join(d0, f"{sid}_t{i}_k*.jpg")))
        pic = ks[-1] if ks else os.path.join(d0, f"{sid}_t{i}_scene.jpg")
        if os.path.exists(pic):
            img.paste(Image.open(pic).convert("RGB").resize((cw, ph)), (x, top))
        mark, col = ("成功", (46, 139, 87)) if t["ok"] else ("失敗", (200, 60, 60))
        d.text((x, top + ph + 12), mark, font=fm, fill=col)
    path = os.path.join(out_dir or OUT, f"session_{tag}_{sid}.png")
    img.save(path)
    return path


def fig_demo():
    src = os.path.join(ROOT, "outputs/demo/d09_i3.mp4")
    if os.path.exists(src):
        os.system(f"ffmpeg -v error -y -ss 7.4 -i '{src}' -frames:v 1 '{os.path.join(OUT, 'demo.png')}'")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_rsi()
    fig_jev()
    fig_system()
    fig_system2()
    fig_teacher()
    fig_dagger()
    fig_grpo()
    fig_demo()
    if os.path.exists(os.path.join(ROOT, "outputs/e2e/final2/sessions.jsonl")):  # 動画の通しの絵(make_video.py と同じ選び方)
        from kitchen.make_video import session_pick_id
        fig_session("final2", session_pick_id("final2")[0])
    print("図:", sorted(os.listdir(OUT)))
