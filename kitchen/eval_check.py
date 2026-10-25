"""文章方式の手順が当たりか外れかを、Jev 風の判定(plan_ok)で見分けられるかを測る。.venv-mlx で実行。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kitchen import brain  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "data/utterances.jsonl"), encoding="utf-8")}
rows = json.load(open(os.path.join(ROOT, f"outputs/eval_plan_{brain.MODEL.split('/')[-1]}.json")))["gen"]
res = []
for r in rows:
    d = data[r["id"]]
    p = brain.plan_ok(d["text"], r["plan"])
    res.append((p, r["hit"]))
    print(f"{r['id']} {'○' if r['hit'] else '×'} P(合っている)={p:.3f} {r['plan']}  {d['text']}", flush=True)
hits = [p for p, h in res if h]; miss = [p for p, h in res if not h]
auc = sum((a > b) + 0.5 * (a == b) for a in hits for b in miss) / max(1, len(hits) * len(miss))
print(f"== 当たり {len(hits)} 本の P 平均 {sum(hits)/len(hits):.3f} / 外れ {len(miss)} 本の P 平均 {sum(miss)/max(1,len(miss)):.3f} / AUC {auc:.3f}")
for th in (0.3, 0.5, 0.7, 0.9):
    act = [(p, h) for p, h in res if p >= th]
    print(f"   しきい値 {th}: 動く {len(act)} 本(うち当たり {sum(h for _, h in act)})/ 聞き返す {len(res)-len(act)} 本(うち当たり {sum(h for p, h in res if p < th)})")
