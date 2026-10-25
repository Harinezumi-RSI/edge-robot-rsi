"""声の往復テスト: 文 → 話す(TTS)→ 聞く(STT)→ 文字の誤り率。耳と口の改造の物差し。.venv-mlx で実行。

  python kitchen/voice_bench.py tts --name base                  # 声を作る(outputs/voice/bench/<name>/)。--steps 4 --bits 8 が元
  python kitchen/voice_bench.py tts --name s2 --steps 2          # 口の段数を減らす
  python kitchen/voice_bench.py tts --name b4 --bits 4           # 口を 4bit に縮める(8bit の重みを戻してから 4bit に)
  python kitchen/voice_bench.py stt --name base --tag hot        # 聞き取って誤り率(台所の言葉のヒントあり)
  python kitchen/voice_bench.py stt --name base --tag nohot --no-hot
  python kitchen/voice_bench.py stt --name base --tag e12 --enc-layers 12   # 耳の音を読む部分を 18 層 → 12 層で打ち切る

文: 発話の評価セット(調整用 30・最終 30・手順づくり 30)と見てから察するの 24 本 = 114 本。役は coach と robot を交互に。
誤り率は記号と空白を除き、「うつわ」は「器」にそろえて数える(声にするときだけ「うつわ」と読ませているため)。
結果は outputs/voice/bench/results.jsonl に 1 行ずつ。
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from kitchen import voice  # noqa: E402
from kitchen.runlog import Timer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "outputs", "voice", "bench")
FILES = ["data/utterances.jsonl", "data/utterances_test.jsonl", "data/utterances_test2.jsonl", "data/utterances_scene.jsonl",
         "data/utterances_scene_test.jsonl"]


def sentences():
    out = []
    for f in FILES:
        for l in open(os.path.join(ROOT, f)):
            d = json.loads(l)
            out.append(d.get("text") or d.get("utterance"))
    return out


def norm(s):
    s = unicodedata.normalize("NFKC", s).replace("うつわ", "器")
    return re.sub(r"[\s、。,.?？!！「」『』…ー〜~]", "", s)


def cer(ref, hyp):
    a, b = norm(ref), norm(hyp)
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[-1] / max(1, len(a))


def save_wav(path, audio, sr):
    x = np.clip(np.asarray(audio, dtype=np.float32), -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def requantize(model, bits):
    """8bit の線形層を戻して bits に量子化し直す(口を縮める)。"""
    import mlx.core as mx
    import mlx.nn as nn
    n = 0
    for name, m in list(model.named_modules()):
        for k, child in list(m.children().items()) if hasattr(m, "children") else []:
            if isinstance(child, nn.QuantizedLinear) and child.bits != bits:
                w = mx.dequantize(child.weight, child.scales, child.biases, child.group_size, child.bits)
                lin = nn.Linear(w.shape[1], w.shape[0], bias="bias" in child)
                lin.weight = w
                if "bias" in child:
                    lin.bias = child.bias
                q = nn.QuantizedLinear.from_linear(lin, group_size=child.group_size, bits=bits)
                setattr(m, k, q)
                n += 1
    return n


def param_bytes(model):
    from mlx.utils import tree_flatten
    return sum(v.nbytes for _, v in tree_flatten(model.parameters()))


def run_tts(a):
    from mlx_audio.tts.utils import load_model
    d = os.path.join(BENCH, a.name)
    os.makedirs(d, exist_ok=True)
    m = load_model(voice.TTS_MODEL)
    before = param_bytes(m)
    changed = requantize(m, a.bits) if a.bits != 8 else 0
    S = sentences()
    gen = aud = 0.0
    with Timer("other", f"声の往復テスト: 口 {a.name}(段数 {a.steps}・{a.bits}bit)", sentences=len(S)) as tm:
        for i, s in enumerate(S):
            who = "coach" if i % 2 == 0 else "robot"
            t = time.time()
            r = list(m.generate(text=voice.speakable(s), instruct=voice.VOICES[who], rng_seed=i, num_steps=a.steps))[0]
            dt = time.time() - t
            audio = np.array(r.audio)
            save_wav(os.path.join(d, f"{i:03d}.wav"), audio, m.sample_rate)
            gen += dt
            aud += len(audio) / m.sample_rate
        row = {"kind": "tts", "name": a.name, "steps": a.steps, "bits": a.bits, "n": len(S), "sec_gen": round(gen, 1), "sec_audio": round(aud, 1),
               "rtf": round(gen / aud, 3), "param_mb_before": round(before / 1e6), "param_mb": round(param_bytes(m) / 1e6), "requantized_layers": changed}
        tm.info.update(rtf=row["rtf"], param_mb=row["param_mb"])
    with open(os.path.join(BENCH, "results.jsonl"), "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False))


def find_encoder(model):
    from mlx_audio.stt.models.qwen3_asr.qwen3_asr import AudioEncoder
    stack = [model]
    while stack:
        m = stack.pop()
        if isinstance(m, AudioEncoder):
            return m
        if hasattr(m, "children"):
            for v in m.children().values():
                stack += v if isinstance(v, list) else [v]
        for v in getattr(m, "__dict__", {}).values():
            if hasattr(v, "children") and not isinstance(v, type):
                stack.append(v)
    return None


def run_stt(a):
    from mlx_audio.stt.utils import load_model
    d = os.path.join(BENCH, a.name)
    m = load_model(voice.STT_MODEL)
    if a.enc_layers:
        enc = find_encoder(m)
        enc.layers = enc.layers[: a.enc_layers]
    S = sentences()
    rows, t_all, aud = [], 0.0, 0.0
    for i, s in enumerate(S):
        p = os.path.join(d, f"{i:03d}.wav")
        t = time.time()
        r = m.generate(p, language="Japanese", hotwords=None if a.no_hot else voice.HOTWORDS, max_tokens=voice.MAX_TOKENS)
        if a.relisten and voice.leaked(re.sub(r"\s+", "", getattr(r, "text", str(r)))):
            r = m.generate(p, language="Japanese", max_tokens=voice.MAX_TOKENS)
        t_all += time.time() - t
        aud += voice.wav_seconds(p)
        hyp = re.sub(r"\s+", "", getattr(r, "text", str(r)))
        rows.append({"i": i, "ref": s, "hyp": hyp, "cer": round(cer(s, hyp), 4)})
    with open(os.path.join(d, f"stt_{a.tag}.jsonl"), "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    c = [r["cer"] for r in rows]
    row = {"kind": "stt", "name": a.name, "tag": a.tag, "hotwords": not a.no_hot, "relisten": a.relisten, "enc_layers": a.enc_layers or 18, "n": len(rows),
           "cer_mean": round(float(np.mean(c)), 4), "exact": sum(x == 0 for x in c), "rtf": round(t_all / aud, 3)}
    with open(os.path.join(BENCH, "results.jsonl"), "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tts", "stt"])
    ap.add_argument("--name", default="base")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--tag", default="hot")
    ap.add_argument("--no-hot", action="store_true")
    ap.add_argument("--enc-layers", type=int, default=0)
    ap.add_argument("--relisten", action="store_true", help="ヒントの並びが漏れたらヒントなしで聞き直す(voice.stt と同じ)")
    a = ap.parse_args()
    run_tts(a) if a.what == "tts" else run_stt(a)
