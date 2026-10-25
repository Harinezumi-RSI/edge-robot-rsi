"""提出用の動画(1〜3 分、第 3 版)をつなぐ: 説明の板・図・デモ映像を台本(docs/video_script.md)の順に並べる。.venv-mlx で実行。

  .venv-mlx/bin/python kitchen/make_video.py --dry                          # 並び・ナレーション・素材の有無と、足りない数字(【】)を表示
  .venv-mlx/bin/python kitchen/make_video.py --out outputs/video/pai_final_v3.mp4 [--account-id <Omnicampus の ID>]

ナレーションは GCP の Text-to-Speech(Chirp3-HD の落ち着いた男性の声。ロボットとコーチの声と聞き分けやすいように、10/24 人の案)。
アカウントと課金先は env.local.sh の PAI_GCP_ACCOUNT・PAI_GCP_PROJECT(git に入れない)。無ければ Irodori-TTS のコーチの声に戻す。
デモ映像は demo.py の録画(コーチとロボットの声は Irodori-TTS、Wi-Fi なしで動く。2 倍速)をそのまま使い、下に説明の帯を付ける(声は消さない)。
数字は記録から読む(ループの周・時間は docs/rsi_runs.jsonl、成功は outputs/ の jsonl)。最終テストの数字が無いうちは【】のまま(--dry で確かめる)。
重い処理(評価・学習・先生の OFT)と同時に回さない(声を作るのに GPU を使う)。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_B = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
W, H, FPS = 1280, 720, 30
BG, INK, ACC = (248, 247, 243), (30, 30, 30), (20, 90, 170)
LIMIT = 180.0  # 課題の上限 3 分
FIG = os.path.join(ROOT, "docs", "fig")
DEMO = os.path.join(ROOT, "outputs", "demo")


# ---- 数字(記録から) ----
def count(path, n_min=100):
    p = os.path.join(ROOT, path)
    if not os.path.exists(p):
        return None
    R = [json.loads(l) for l in open(p)]
    return (sum(r["ok"] for r in R), len(R)) if len(R) >= n_min else None


def e2e(tag):
    p = os.path.join(ROOT, "outputs", "e2e", tag, "summary.json")
    return json.load(open(p))["頼み"] if os.path.exists(p) else None


OUTCOME_RE = re.compile(r"^(\d+) 周目")


def loop_stats():
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "docs/rsi_runs.jsonl"))]
    outer = [r for r in rows if r["kind"] == "outer"]
    rounds = {int(m.group(1)) for r in outer if (m := OUTCOME_RE.match(r["name"]))}
    last = max(r["end"] for r in outer)
    rows = [r for r in rows if r["start"] <= last]  # 研究ループの最後の周まで(図 timeline.png・報告書と同じ数え方)
    t = sorted(r["start"] for r in rows) + sorted(r["end"] for r in rows)
    from datetime import datetime
    span = (datetime.strptime(max(t), "%Y-%m-%d %H:%M:%S") - datetime.strptime(min(t), "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600
    return {"rounds": len(rounds), "hours": span}


def kanji(n):
    """0〜9999 を漢数字の読みに(声にするときだけ。「73」を「ななじゅうさん」と読ませる)。"""
    d = "〇一二三四五六七八九"
    if n == 0:
        return "ゼロ"
    out = ""
    for v, u in ((1000, "千"), (100, "百"), (10, "十")):
        q, n = divmod(n, v)
        if q:
            out += ("" if q == 1 else d[q]) + u
    return out + (d[n] if n else "")


def say_num(x):
    return kanji(x) if isinstance(x, int) else "【】"


def session_pick_id(tag="final2"):
    """通しの絵にするセッション: 最終テストで 3 回ともできたもののうち番号が最初のもの(無ければ E01)。→ (id, 3 回ともの数, セッション数)"""
    p = os.path.join(ROOT, "outputs", "e2e", tag, "sessions.jsonl")
    if not os.path.exists(p):
        return "E01", 0, 0
    S = [json.loads(l) for l in open(p)]
    full = [s["id"] for s in S if all(t["ok"] for t in s["turns"])]
    return (full[0] if full else S[0]["id"]), len(full), len(S)


def session_pick(tag="final2"):
    """絵(docs/fig/session_<tag>_<id>.png、.venv で make_figures.py が作る)のパスと帯の補足。"""
    sid, nfull, n = session_pick_id(tag)
    note = f"・3 回ともできた {nfull}/{n} セッションの 1 つ" if nfull else ""
    return os.path.join(FIG, f"session_{tag}_{sid}.png"), note


def numbers():
    lite = count("outputs/final2/lite.jsonl")
    fin = e2e("final2")
    st = loop_stats()
    sess, note = session_pick()
    return {
        "session_png": sess, "session_note": note,
        "final_hand": f"{lite[0]}/{lite[1]}" if lite else "【】",
        "final_hand_n": lite[0] if lite else None,
        "final_e2e": fin or "【】",
        "final_e2e_n": int(fin.split("/")[0]) if fin else None,
        "rounds": st["rounds"], "hours": round(st["hours"]),
    }


# ---- 素材 ----
def fit(d, text, font, size, width):
    """幅に収まるまで文字を小さくする(題が切れないように)。"""
    while size > 18 and d.textlength(text, font=ImageFont.truetype(font, size)) > width:
        size -= 2
    return ImageFont.truetype(font, size)


def slide(path, title, lines, sub="", hero=None):
    img = Image.new("RGB", (W, H), BG)
    if hero and os.path.exists(hero):
        h = Image.open(hero).convert("RGB")
        h = h.resize((560, int(560 * h.height / h.width)))
        img.paste(h, (W - 560 - 50, (H - h.height) // 2 + 30))
    d = ImageDraw.Draw(img)
    d.text((70, 60), title, font=fit(d, title, FONT_B, 44, W - 120), fill=INK)
    if sub:
        d.text((72, 126), sub, font=ImageFont.truetype(FONT, 24), fill=(110, 110, 110))
    y = 200
    for ln in lines:
        big = ln.startswith("!")
        text = ln.lstrip("!")
        f = ImageFont.truetype(FONT_B if big else FONT, 32 if big else 27)
        d.text((90, y), text, font=f, fill=ACC if big else INK)
        y += 54 if big else 46
    foot, ff = "家庭用 Mac mini 1 台(M4 Pro・24GB)・シミュレーション", ImageFont.truetype(FONT, 20)
    d.text((W - 30 - d.textlength(foot, font=ff), H - 44), foot, font=ff, fill=(140, 140, 140))
    img.save(path)


def figure(path, title, png, note=""):
    """図を 1 枚、題の下に収める。"""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((50, 28), title, font=fit(d, title, FONT_B, 38, W - 100), fill=INK)
    top, bottom = 96, H - (70 if note else 30)
    f = Image.open(png).convert("RGB")
    s = min((W - 80) / f.width, (bottom - top) / f.height)
    f = f.resize((int(f.width * s), int(f.height * s)), Image.LANCZOS)
    img.paste(f, ((W - f.width) // 2, top + (bottom - top - f.height) // 2))
    if note:
        d.text((50, H - 56), note, font=ImageFont.truetype(FONT, 24), fill=ACC)
    img.save(path)


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return float(out or 0)


def has_audio(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path],
                         capture_output=True, text=True).stdout.strip()
    return bool(out)


GCP_VOICE = os.environ.get("PAI_GCP_VOICE", "ja-JP-Chirp3-HD-Charon")


def gcp_tts(text):
    """GCP の Text-to-Speech → wav のパス(同じ声と文は作り直さない。トークンは表示しない)。"""
    import base64
    import hashlib
    import urllib.request
    cache = os.path.join(ROOT, "outputs", "voice", "cache_gcp")
    os.makedirs(cache, exist_ok=True)
    out = os.path.join(cache, hashlib.sha1(f"{GCP_VOICE}|{text}".encode()).hexdigest()[:16] + ".wav")
    if os.path.exists(out):
        return out
    token = subprocess.run(["gcloud", "auth", "print-access-token", f"--account={os.environ['PAI_GCP_ACCOUNT']}"],
                           capture_output=True, text=True, check=True).stdout.strip()
    body = {"input": {"text": text}, "voice": {"languageCode": "ja-JP", "name": GCP_VOICE},
            "audioConfig": {"audioEncoding": "LINEAR16", "sampleRateHertz": 24000}}
    req = urllib.request.Request("https://texttospeech.googleapis.com/v1/text:synthesize", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {token}", "x-goog-user-project": os.environ["PAI_GCP_PROJECT"],
                                          "Content-Type": "application/json"})
    audio = json.loads(urllib.request.urlopen(req, timeout=120).read())["audioContent"]
    open(out, "wb").write(base64.b64decode(audio))
    return out


def narrator():
    return f"GCP {GCP_VOICE}" if os.environ.get("PAI_GCP_PROJECT") and os.environ.get("PAI_GCP_ACCOUNT") else "Irodori-TTS(コーチ)"


def narrate(text, tmp, i):
    """ナレーションの声 → 44.1kHz ステレオの wav と長さ。"""
    if narrator().startswith("GCP"):
        src = gcp_tts(text)
    else:
        from kitchen.voice import tts
        src, _, _ = tts(text, who="coach")
    out = os.path.join(tmp, f"narr{i:02d}.wav")
    run(["ffmpeg", "-y", "-i", src, "-ar", "44100", "-ac", "2", out])
    return out, duration(out)


def seg_still(png, sec, wav, out):
    """静止画 + ナレーション(無ければ無音)。"""
    a = ["-i", wav] if wav else ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
    run(["ffmpeg", "-y", "-loop", "1", "-t", f"{sec:.2f}", "-i", png, *a, "-filter_complex", "[1:a]apad[a]", "-map", "0:v", "-map", "[a]",
         "-t", f"{sec:.2f}", "-r", str(FPS), "-vf", f"scale={W}:{H}", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", "-ar", "44100",
         "-ac", "2", out])


def seg_clip(clip, caption, out, tmp, speed=1.0):
    """デモ映像 + 下に説明の帯。映像の声(コーチとロボット)は残す。speed で更に速める(声は音程を変えずに速める)。"""
    band = os.path.join(tmp, os.path.basename(out) + "_band.png")
    img = Image.new("RGB", (W, 60), (40, 40, 40))
    ImageDraw.Draw(img).text((20, 14), caption, font=ImageFont.truetype(FONT, 26), fill=(255, 255, 255))
    img.save(band)
    v = (f"[0:v]setpts=PTS/{speed},scale={W}:{H - 60}:force_original_aspect_ratio=decrease,"
         f"pad={W}:{H - 60}:(ow-iw)/2:(oh-ih)/2:color=0xF8F7F3[v];[v][1:v]vstack=inputs=2,fps={FPS}[vo]")
    if has_audio(clip):
        fc, amap, extra = v + f";[0:a]atempo={speed},aresample=44100,apad[ao]", "[ao]", []  # 声が先に終わっても映像の最後まで
    else:
        fc, amap, extra = v, "2:a", ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]
    run(["ffmpeg", "-y", "-i", clip, "-i", band, *extra, "-filter_complex", fc, "-map", "[vo]", "-map", amap, "-shortest",
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-c:a", "aac", "-ar", "44100", "-ac", "2", out])


# ---- 台本(docs/video_script.md と同じ順)----
def plan(N, account_id):
    """(種類, 素材, 帯や題, ナレーション, 最短の秒, 速さ)。ナレーションの文は声にするとき用(英字・数字は読みで書く)。"""
    t = f"PAI最終課題{('_' + account_id) if account_id else ''}"
    return [
        ("slide", ("エッジで動く AI ロボットを、家庭用パソコンで RSI",
                   ["Wi-Fi なしで動き、", "Mac mini 1 台で自分を鍛える家事ロボット"], t, os.path.join(DEMO, "v3_oneshot_u1_k0_a0.jpg")),
         "", "エッジ、つまり手元で動くエーアイロボットを、家庭用パソコンで鍛えます。ロボットを鍛えるのは、パソコンの中のクロードです。", 6, 1.0),
        ("figure", os.path.join(FIG, "system2.png"), "Wi-Fi なしで動く(すべて公開モデル、動くときの重み 約 5.3GB)",
         "ロボットは、ワイファイなしで動きます。聞くのも、考えるのも、見るのも、話すのも、このマックミニの中です。", 6, 1.0),
        ("clip", os.path.join(DEMO, "v3_oneshot.mp4"), "声で頼む → 場面を見て察する → 動く(Wi-Fi なし・2 倍速)", "", 0, 1.0),
        ("clip", os.path.join(DEMO, "v3_askback.mp4"), "手順が頼みと合わない(引き出し → 棚の上)と、動く前に聞き返す → 言い直しで動く(2 倍速)", "", 0, 1.0),
        ("figure", os.path.join(FIG, "jev.png"), "確かめる: 改造 Jev(Qwen の 8 層目 + 目標ごとの小さな頭)",
         "動いたあとは、カメラの画像で、うまくできたかを確かめます。改造ジェブは、問いかける方式より正確で、二十二倍速く判定します。", 6, 1.0),
        ("figure", os.path.join(FIG, "bench.png"), "ベースから作った手: 1 step 3.9ms",
         "ロボットの手は、このマックミニで一から作りました。公開の画像の土台に、六メガバイトの頭を足しただけです。一ステップ、四ミリ秒。"
         "同じマックで、七十億パラメータのオーエフティーは、その五十倍以上かかります。", 6, 1.0),
        ("figure", os.path.join(FIG, "teacher.png"), "手の天井は先生だった",
         "手を良くしたのは、手の作りではなく、先生でした。先生を替えるだけで、七十三から九十に上がりました。", 6, 1.0),
        ("figure", os.path.join(FIG, "dagger.png"),
         "自分で鍛える: 練習 → 改造 Jev で採点 → 先生の答え → 学び直す(1 分以内)|Jev が自信のある試行は先生に聞く場面を間引く(答えつけ 4 割)",
         "練習して、自分で採点して、学び直す。自分の練習に先生の答えをもらうと、前の作業で変わった場面が、五十二から六十二に上がりました。", 6, 1.0),
        ("figure", os.path.join(FIG, "grpo.png"),
         "先生なしの自己進化: 報酬は改造 Jev だけ・LLM の RL の部品(GSPO・DAPO・PPO-ptx ほか)で 100 回・9,600 試行"
         "|使っていない初期状態 800 試行で: 変わった場面 64.8 → 71.3%(p<0.0001)・普段 92.6 → 94.0%",
         "先生がいなくても鍛えられます。報酬は、自分の目だけ。エルエルエムの強化学習の工夫を借りて、九千六百回練習すると、"
         "前の作業で変わった場面が、六十五から七十一パーセントに上がりました。", 6, 1.0),
        ("figure", N["session_png"], f"声で 3 回続けて頼む(最終テストの通し: 頼み {N['final_e2e']}){N['session_note']}",
         f"声で三回続けて頼むと、最終テストでは、三十回のうち{say_num(N['final_e2e_n'])}回できました。", 10, 1.0),
        ("figure", os.path.join(FIG, "timeline.png"), f"コーチ Claude の研究のループ: {N['rounds']} 周・{N['hours']} 時間",
         f"考えを立てて、試して、採用か不採用かを決める。このループを、{kanji(N['rounds'])}周、Claude が回しました。方針は人が決めます。", 6, 1.0),
        ("slide", ("聞く・見る・察する・動く・確かめる・鍛える",
                   ["耳 Qwen3-ASR 0.6B / 頭・目 Qwen3.5-4B(改造 Jev)", "手 DINOv2-small + 頭 6MB(先生 OpenVLA-OFT)/ 口 Irodori-TTS",
                    f"!最終テスト: 手だけ {N['final_hand']}・家事の通し {N['final_e2e']}", "コードと記録は GitHub に公開"], "", None),
         "", "聞く、見る、察する、動く、確かめる、鍛える。全部、このマックミニの中です。", 6, 1.0),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/video/pai_final_v3.mp4")
    ap.add_argument("--account-id", default="", help="題名に入れる Omnicampus のアカウント ID(人が決める)")
    ap.add_argument("--dry", action="store_true", help="並び・ナレーション・素材の有無と【】を表示するだけ")
    a = ap.parse_args()
    N = numbers()
    P = plan(N, a.account_id)
    if a.dry:
        for i, (kind, src, cap, narr, sec, sp) in enumerate(P):
            path = src[3] if kind == "slide" else src
            have = "" if kind == "slide" or os.path.exists(path) else "  ★素材が無い"
            print(f"{i:2d} {kind:6s} {os.path.basename(path) if path else '':22s} {cap or (src[0] if kind == 'slide' else '')}{have}")
            if narr:
                print(f"     声: {narr}")
        print("ナレーションの声:", narrator())
        miss = [k for k, v in N.items() if v in ("【】", None)]
        print("数字:", {k: v for k, v in N.items() if not k.endswith("_n")}, "/ 足りない:", miss or "なし")
        return
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        segs, total = [], 0.0
        for i, (kind, src, cap, narr, sec, sp) in enumerate(P):
            out = os.path.join(tmp, f"seg{i:02d}.mp4")
            if kind == "clip":
                if not os.path.exists(src):
                    print(f"[とばす] 映像が無い: {src}")
                    continue
                seg_clip(src, cap, out, tmp, sp)
            else:
                wav, nlen = narrate(narr, tmp, i) if narr else ("", 0.0)
                png = os.path.join(tmp, f"still{i:02d}.png")
                if kind == "slide":
                    title, lines, sub, hero = src
                    slide(png, title, lines, sub, hero)
                else:
                    if not os.path.exists(src):
                        print(f"[とばす] 図が無い: {src}")
                        continue
                    title, _, note = cap.partition("|")  # 「題|下の注」
                    figure(png, title, src, note)
                seg_still(png, max(sec, nlen + 0.6), wav, out)
            d = duration(out)
            total += d
            print(f"{i:2d} {kind:6s} {d:5.1f} 秒(累計 {total:5.1f})")
            segs.append(out)
        lst = os.path.join(tmp, "list.txt")
        open(lst, "w").write("".join(f"file '{s}'\n" for s in segs))
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
             "-ar", "44100", "-ac", "2", os.path.join(ROOT, a.out)])
    dur = duration(os.path.join(ROOT, a.out))
    print(f"動画: {a.out}({dur:.1f} 秒、{len(segs)} 区間){'  ★3 分を超えた' if dur > LIMIT else ''}")


if __name__ == "__main__":
    main()
