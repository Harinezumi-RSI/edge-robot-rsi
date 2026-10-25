"""手順づくりの正確さを発話セット(data/utterances.jsonl)で測る。.venv-mlx で実行。

  python kitchen/eval_plan.py [gen|judge ...]
正解は発話ごとに許す手順の一覧(ok)。完全一致で数える。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kitchen import brain  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    modes = sys.argv[1:] or ["gen", "judge"]
    data = [json.loads(l) for l in open(os.path.join(ROOT, "data/utterances.jsonl"), encoding="utf-8")]
    out = {}
    for mode in modes:
        rows, t0 = [], time.time()
        for d in data:
            t = time.time()
            r = {"gen": brain.plan, "judge": brain.plan_judge, "rerank": brain.plan_rerank, "fast": brain.plan_fast}[mode](d["text"])
            hit = r["plan"] in d["ok"]
            rows.append({"id": d["id"], "kind": d["kind"], "plan": r["plan"], "hit": hit, "ask": r.get("ask", False),
                         "conf": r.get("conf", r.get("p")), "cands": r.get("cands"), "sec": round(time.time() - t, 2)})
            print(f"[{mode}] {d['id']} {'○' if hit else '×'}{' (聞き返す)' if r.get('ask') else ''} {r['plan']} 正解 {d['ok'][0]} "
                  f"conf {r.get('conf', r.get('p'))} 候補 {r.get('cands')}  {d['text']}", flush=True)
        by = {}
        for r in rows:
            by.setdefault(r["kind"], []).append(r["hit"])
        summ = {k: f"{sum(v)}/{len(v)}" for k, v in by.items()}
        n_ask = sum(r["ask"] for r in rows)
        wrong_acted = sum(bool((not r["hit"]) and (not r["ask"]) and r["plan"]) for r in rows)
        print(f"== {mode}: 全体 {sum(r['hit'] for r in rows)}/{len(rows)}  {summ}  聞き返す {n_ask}  "
              f"間違ったまま動く {wrong_acted}  平均 {(time.time() - t0) / len(rows):.1f} 秒/発話", flush=True)
        out[mode] = rows
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, f"outputs/eval_plan_{'_'.join(modes)}_{brain.MODEL.split('/')[-1]}.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
