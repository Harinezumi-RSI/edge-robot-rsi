"""耳の調整(学習なし): 聞き取りに渡す前置きを変えて比べる。.venv-mlx で実行。

  python kitchen/stt_ctx_eval.py

V0: いま(台所の単語を並べたヒント + 並びが漏れたら聞き直す)
V1: 場面を説明する文の前置き(単語の並びにしない)
音声: 往復テストの 114 本(outputs/voice/bench/base)+ 未使用の 30 本(stt_fix_eval で作った声)+ 通しの評価で聞き間違えた 2 本。
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from kitchen import voice  # noqa: E402
from kitchen.voice_bench import cer, sentences  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTX = ("台所の家事ロボットへの頼みごと。よく出る言葉は、コンロの火、黒い器、白いお皿、棚の引き出し、"
       "ワインボトル、ワインラック、クリームチーズ。")


def main():
    from mlx_audio.stt.utils import load_model
    m = load_model(voice.STT_MODEL)
    items = [(s, os.path.join(ROOT, f"outputs/voice/bench/base/{i:03d}.wav")) for i, s in enumerate(sentences())]
    fresh = [json.loads(l)["text"] for l in open(os.path.join(ROOT, "data/utterances_stt_fresh.jsonl"))]
    for i, s in enumerate(fresh):
        items.append((s, voice.tts(s, "coach" if i % 2 == 0 else "robot", seed=i)[0]))
    hard = ["コンロの火、つけておいてもらえる？", "火もお願いできる？"]
    for s in hard:
        items.append((s, voice.tts(s, "coach")[0]))
    clean = lambda r: re.sub(r"\s+", "", getattr(r, "text", str(r)))  # noqa: E731

    def v0(p):
        t = clean(m.generate(p, language="Japanese", hotwords=voice.HOTWORDS, max_tokens=voice.MAX_TOKENS))
        return clean(m.generate(p, language="Japanese", max_tokens=voice.MAX_TOKENS)) if voice.leaked(t) else t

    def v1(p):
        return clean(m.generate(p, language="Japanese", system_prompt=CTX, max_tokens=voice.MAX_TOKENS))

    res = {}
    for name, fn in (("V0 単語のヒント", v0), ("V1 場面の説明", v1)):
        H = [fn(p) for _, p in items]
        c = np.array([cer(s, h) for (s, _), h in zip(items, H)])
        res[name] = {"114 本": f"{c[:114].mean():.3f}・完全一致 {(c[:114] == 0).sum()}", "未使用 30 本": f"{c[114:144].mean():.3f}・完全一致 {(c[114:144] == 0).sum()}",
                     "聞き間違えた 2 本": [h for h in H[144:]]}
        print(name, json.dumps(res[name], ensure_ascii=False), flush=True)
    json.dump(res, open(os.path.join(ROOT, "outputs/voice/stt_ctx.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
