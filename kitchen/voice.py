"""声の出入り(公開モデル、Mac の中だけで動く)。.venv-mlx で使う(brain_server の /tts・/stt から呼ぶ)。

  話す: Irodori-TTS v4.1 Small(MIT、日本語専用、4 step の MeanFlow 版 8bit)。声は文章で設計する(VoiceDesign)。
  聞く: Qwen3-ASR 0.6B(Apache-2.0、4bit)。台所の言葉をヒントに渡し、ヒントの並びが書き起こしに漏れたらヒントなしで聞き直す
        (往復テスト 114 本: 誤り率の平均 ヒントあり 0.117・なし 0.084・漏れたら聞き直す 0.076、完全一致 59・54・60)。

役は 2 つ: coach = パソコンの中の Claude(ロボットを鍛えるコーチ)/ robot = 家事ロボット。
"""
import hashlib
import os
import re
import time
import wave

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TTS_MODEL = os.environ.get("PAI_TTS", "mlx-community/Irodori-TTS-v4.1-Small-MF-8bit")
STT_MODEL = os.environ.get("PAI_STT", "mlx-community/Qwen3-ASR-0.6B-4bit")
CACHE = os.path.join(ROOT, "outputs", "voice", "cache")
VOICES = {
    "coach": "落ち着いた温かい声の大人の女性が、ロボットを励ますコーチのように、はっきりと聞き取りやすく話す。",
    "robot": "明るく素直な若い声で、少しだけ機械っぽく、元気よく返事をする。",
}
MAX_TOKENS = 96  # 話しかける文は短い。壊れた音で書き起こしが止まらなくなっても待たされないように(上限は既定 8192)
HOTWORDS = ["器", "コンロ", "火", "引き出し", "棚", "皿", "クリームチーズ", "ワインボトル", "ワインラック"]
# 声にするときだけの読みのヒント(表示の文はそのまま)。「器」は「うつわ」と読ませないと「き」「うつわもの」に揺れる
READING = [("黒い器", "黒いうつわ"), ("器", "うつわ"), ("「", ""), ("」", "")]
_m = {}


def speakable(text):
    for a, b in READING:
        text = text.replace(a, b)
    return text


def wav_seconds(path):
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def tts(text, who="robot", seed=0):
    """→ (wav のパス, 長さ[秒], 作るのにかかった秒)。同じ文と役は作り直さない。"""
    import mlx.core as mx
    from mlx_audio.tts.generate import generate_audio
    os.makedirs(CACHE, exist_ok=True)
    key = hashlib.sha1(f"{TTS_MODEL}|{who}|{seed}|{text}".encode()).hexdigest()[:16]
    out = os.path.join(CACHE, f"{who}_{key}")
    if os.path.exists(out + "_000.wav"):
        return out + "_000.wav", wav_seconds(out + "_000.wav"), 0.0
    t = time.time()
    mx.random.seed(seed)
    generate_audio(model=TTS_MODEL, text=speakable(text), instruct=VOICES[who], file_prefix=out, audio_format="wav", verbose=False)
    return out + "_000.wav", wav_seconds(out + "_000.wav"), time.time() - t


def stt(path):
    """→ (聞き取った文, かかった秒)"""
    if "stt" not in _m:
        from mlx_audio.stt.utils import load_model
        _m["stt"] = load_model(STT_MODEL)
    t = time.time()
    r = _m["stt"].generate(path, language="Japanese", hotwords=HOTWORDS, max_tokens=MAX_TOKENS)
    text = re.sub(r"\s+", "", getattr(r, "text", str(r)))
    if leaked(text):  # ヒントの並びを書き起こしに漏らした → ヒントなしで聞き直す(往復テスト 114 本で 2 本)
        r = _m["stt"].generate(path, language="Japanese", max_tokens=MAX_TOKENS)
        text = re.sub(r"\s+", "", getattr(r, "text", str(r)))
    return text, time.time() - t


def leaked(text):
    """ヒントの単語の並び(「器,コンロ,火,…」)が書き起こしに漏れたか。日本語の読点は「、」で、漏れた並びは半角のカンマになる。"""
    return "," in text
