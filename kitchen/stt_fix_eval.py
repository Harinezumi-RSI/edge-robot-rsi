"""耳の後処理(頭が聞き間違いを直す)の効き方。.venv-mlx で実行。

  python kitchen/stt_fix_eval.py old     # これまでの往復テスト 114 本の聞き取り(outputs/voice/bench/base/stt_fix.jsonl)に後処理
  python kitchen/stt_fix_eval.py fresh   # 新しい 30 本(data/utterances_stt_fresh.jsonl、未使用): 声にする → 聞き取る → 後処理

数えるもの: 文字の誤り率の平均・完全一致の数・後処理が直した数・正しかった文を壊した数。
114 本は後処理の手本を作るときに聞き間違いの型を見た集合なので、未使用の 30 本の方を本番の数字にする。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from kitchen import brain, voice  # noqa: E402
from kitchen.voice_bench import cer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def report(rows, name):
    b = np.array([r["cer_raw"] for r in rows])
    a = np.array([r["cer_fix"] for r in rows])
    out = {"set": name, "n": len(rows), "cer_raw": round(float(b.mean()), 4), "cer_fix": round(float(a.mean()), 4),
           "exact_raw": int((b == 0).sum()), "exact_fix": int((a == 0).sum()), "better": int((a < b).sum()), "worse": int((a > b).sum()),
           "broke_correct": int(((b == 0) & (a > 0)).sum()), "sec_fix_median": float(np.median([r["sec_fix"] for r in rows]))}
    print(json.dumps(out, ensure_ascii=False))
    for r in rows:
        if r["cer_fix"] != r["cer_raw"]:
            print(f"  {'改善' if r['cer_fix'] < r['cer_raw'] else '悪化'} 元「{r['ref']}」 聞き取り「{r['raw']}」→ 後処理「{r['fix']}」")
    return out


def main():
    what = sys.argv[1]
    rows = []
    if what == "old":
        for r in map(json.loads, open(os.path.join(ROOT, "outputs/voice/bench/base/stt_fix.jsonl"))):
            t = time.time()
            fx = brain.fix_stt(r["hyp"])
            rows.append({"ref": r["ref"], "raw": r["hyp"], "fix": fx, "cer_raw": cer(r["ref"], r["hyp"]), "cer_fix": cer(r["ref"], fx),
                         "sec_fix": time.time() - t})
    else:
        for i, d in enumerate(map(json.loads, open(os.path.join(ROOT, "data/utterances_stt_fresh.jsonl")))):
            who = "coach" if i % 2 == 0 else "robot"
            wav, _, _ = voice.tts(d["text"], who, seed=i)
            raw, _ = voice.stt(wav)
            t = time.time()
            fx = brain.fix_stt(raw)
            rows.append({"ref": d["text"], "raw": raw, "fix": fx, "cer_raw": cer(d["text"], raw), "cer_fix": cer(d["text"], fx),
                         "sec_fix": time.time() - t})
    out = report(rows, what)
    d = os.path.join(ROOT, "outputs", "voice")
    json.dump({"summary": out, "rows": rows}, open(os.path.join(d, f"stt_fix_{what}.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
