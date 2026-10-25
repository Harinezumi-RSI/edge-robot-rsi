"""外側のループの点検: ミニ RSI の各周の「どこが足を引っ張っているか」を数字で並べ、次の周の的を決める材料にする。.venv で実行。

  python kitchen/loop_health.py            # 全部の周
  python kitchen/loop_health.py risall1    # 指定した周

見るもの(段ごと):
  練習   : 1 試行あたりの秒・練習の成功率(シミュレータの正解は記録のためだけ)・採用できた成功の数/分
  採点   : やろうとした目標の判定の精度(採用した中の本当の成功)・拾えた割合(本当の成功の中で採用)
  学習   : 1 step の秒・学習の分
  評価   : 比べる相手からの伸び(対応比較)
最後に「いちばん効率を落としている段」を出す(人が言う前に自分で気づくため)。
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rows(p):
    p = os.path.join(ROOT, p)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def judge_quality(tag, tau):
    J = rows(f"outputs/rsi/{tag}/practice_judged.jsonl")
    if not J:
        return None
    true = [bool(r["labels"][str(r["skill"])]) for r in J]
    acc = [r["p_jev"] >= tau for r in J]
    tp = sum(a and t for a, t in zip(acc, true))
    return {"n": len(J), "true": sum(true), "accepted": sum(acc), "precision": tp / max(1, sum(acc)),
            "recall": tp / max(1, sum(true)), "wasted_true": sum(t and not a for a, t in zip(acc, true)),
            "noise": sum(a and not t for a, t in zip(acc, true))}


def main():
    want = sys.argv[1:]
    rounds = [r for r in rows("outputs/rsi/rounds.jsonl") if not want or r.get("tag", f"r{r['round']}") in want]
    issues = []
    for r in rounds:
        tag = r.get("tag", f"r{r['round']}")
        jq = judge_quality(tag, r["tau"])
        att = r.get("attempts", 0)
        per_try = 60 * r.get("practice_min", 0) / max(1, att)
        yield_per_min = r.get("accepted", 0) / max(1e-6, r.get("practice_min", 0) + r.get("judge_min", 0))
        total = sum(r.get(k, 0) for k in ("practice_min", "judge_min", "train_min", "eval_min", "eval0_min"))
        print(f"== {tag}(τ={r['tau']}、合計 {total:.0f} 分)")
        print(f"  練習: {att} 本・{per_try:.0f} 秒/本・練習の成功 {r.get('sim_ok', '?')}・採用 {r.get('accepted', '?')}(採用 {yield_per_min:.1f} 本/分)")
        if jq:
            print(f"  採点: 精度 {jq['precision']:.2f}・拾えた {jq['recall']:.2f}(捨てた本当の成功 {jq['wasted_true']}・混ざった失敗 {jq['noise']})")
        print(f"  学習: {r.get('train_min', '?')} 分({r.get('train_episodes', '?')} 本)/ 評価: {r.get('val_ok', '?')}/{r.get('val_n', '?')}"
              + (f"・ずれなし {r['val0_ok']}/{r['val0_n']}" if "val0_ok" in r else ""))
        share = {k: r.get(k, 0) / max(1e-6, total) for k in ("practice_min", "train_min", "eval_min")}
        if jq and jq["recall"] < 0.9:
            issues.append((1 - jq["recall"], f"{tag}: 採点が本当の成功の {1 - jq['recall']:.0%} を捨てている → 判定器のデータ・作り・しきい値"))
        if jq and jq["precision"] < 0.93:
            issues.append((1 - jq["precision"], f"{tag}: 採用した中の {1 - jq['precision']:.0%} が失敗 → 判定器の精度・しきい値"))
        for k, v in share.items():
            if v > 0.45:
                issues.append((v - 0.45, f"{tag}: {k} が全体の {v:.0%} → この段の速さが周の回数を決めている"))
    print("\n== いちばん効率を落としているもの(大きい順)")
    for score, msg in sorted(issues, reverse=True)[:5]:
        print(f"  {score:.2f}  {msg}")
    if not issues:
        print("  目立つものなし")


if __name__ == "__main__":
    main()
