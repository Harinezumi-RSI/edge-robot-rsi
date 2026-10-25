"""頭(Qwen3.5-4B、MLX)を小さな HTTP サーバにする。手(SmolVLA)は別の venv で動くので、ここに問い合わせる。.venv-mlx で実行。

  python kitchen/brain_server.py --port 8765          # 127.0.0.1 だけで待ち受け

  POST /plan     {"utterance": "..."}                  → 手順・返事・確認の確率(Jev 風)・聞き返すか
  POST /jev      {"image": "/abs/path.jpg"}            → 10 目標の達成確率(改造 Jev、画像 1 回の読み込み)
  POST /p_yes    {"question": "...", "image": "..."}   → ゼロショットの「はい」の確率(比較用)
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from kitchen import brain, jev, voice  # noqa: E402

app = FastAPI()
# 確認の確率がこれ未満なら動く前に聞き返す。発話 30 本(data/utterances.jsonl)の結果を見て置いた値なので、
# あの 30 本はもう「調整用」。最終の評価は、調整に一度も使っていない別の発話(data/utterances_test.jsonl、後で作る)で測る。
ASK_BELOW = 0.15


class PlanReq(BaseModel):
    utterance: str
    image: str = ""  # いまの場面の画像(ホームの姿勢で撮ったもの)。あれば「見てから察する」: 済みの目標を引き算する
    history: list = []  # この会話のこれまでの頼み [{"utterance", "intent", "reply", "plan"}, ...]


class ImgReq(BaseModel):
    image: str


class YesReq(BaseModel):
    question: str
    image: str


@app.post("/plan")
def plan(r: PlanReq):
    t = time.time()
    hist = [(h["utterance"], {"intent": h.get("intent", ""), "reply": h.get("reply", ""), "plan": h.get("plan", [])}) for h in r.history]
    seen = {}
    if r.image:
        pj = jev.predict(r.image)
        seen = {"done": [g for g in range(10) if pj[g] >= 0.5], "p_jev": [round(float(x), 3) for x in pj], "sec_seen": round(time.time() - t, 2)}
        t = time.time()
        p = brain.plan_subtract(r.utterance, seen["done"], hist)
        seen["skipped"] = p["skipped"]
    else:
        p = brain.plan(r.utterance, hist)
    out = {"plan": p["plan"], "reply": p["reply"], "intent": p["intent"], "sec_plan": round(time.time() - t, 2), **seen}
    if p["plan"]:
        t = time.time()
        out["p_ok"] = round(brain.plan_ok(r.utterance, p["plan"]), 3)
        out["ask"] = out["p_ok"] < ASK_BELOW
        out["sec_check"] = round(time.time() - t, 2)
    else:
        out.update(p_ok=None, ask=False)
    return out


@app.post("/jev")
def judge(r: ImgReq):
    t = time.time()
    p = jev.predict(r.image)
    return {"probs": [round(float(x), 4) for x in p], "sec": round(time.time() - t, 3)}


@app.post("/p_yes")
def p_yes(r: YesReq):
    t = time.time()
    p, mass = brain.p_yes(r.question, r.image)
    return {"p": round(float(p), 4), "mass": round(float(mass), 4), "sec": round(time.time() - t, 3)}


class TtsReq(BaseModel):
    text: str
    who: str = "robot"
    seed: int = 0


class SttReq(BaseModel):
    wav: str
    fix: bool = False  # 聞き取りの後処理(頭が聞き間違いを直す)


@app.post("/tts")
def tts(r: TtsReq):
    """声にする(Irodori-TTS、役ごとに設計した声)。"""
    path, sec, gen = voice.tts(r.text, r.who, r.seed)
    return {"wav": path, "sec": round(sec, 2), "sec_gen": round(gen, 2)}


@app.post("/stt")
def stt(r: SttReq):
    """聞き取る(Qwen3-ASR)。"""
    text, sec = voice.stt(r.wav)
    out = {"text": text, "sec": round(sec, 2)}
    if r.fix:
        t = time.time()
        fixed = brain.fix_stt(text)
        out.update(raw=text, text=fixed, fixed=fixed != text, sec_fix=round(time.time() - t, 2))
    return out


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    brain._load()  # 起動時に読み込んでおく(最初の問い合わせを速く)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
