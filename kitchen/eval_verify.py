"""出来を確かめる判定(Jev 風、画像 → 「はい」の確率)を、シミュレータの正解と突き合わせて測る。.venv-mlx で実行。

  python kitchen/eval_verify.py outputs/snaps [--skills 1,7]

snaps には bench.py --snap-dir が保存した <名前>.png と <名前>.json(10 個の目標それぞれの正解)が並ぶ。
画像 1 枚につき問い 10 個(動作ごと)を判定し、動作ごとと全体の AUC・正解率(しきい値 0.5)を出す。
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kitchen import brain  # noqa: E402
from kitchen.skills import CHECK_Q  # noqa: E402


def auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def main():
    d = sys.argv[1]
    skills = [int(x) for x in sys.argv[sys.argv.index("--skills") + 1].split(",")] if "--skills" in sys.argv else list(CHECK_Q)
    out = os.path.join(d, f"verify_{brain.MODEL.split('/')[-1]}.jsonl")
    seen = set()
    if os.path.exists(out):
        seen = {(r["img"], r["skill"]) for r in map(json.loads, open(out))}
    for js in sorted(glob.glob(os.path.join(d, "*.json"))):
        img = js[:-5] + ".png"
        labels = json.load(open(js))
        for s in skills:
            if (os.path.basename(img), s) in seen:
                continue
            p, mass = brain.p_yes(CHECK_Q[s], img)
            with open(out, "a") as f:
                f.write(json.dumps({"img": os.path.basename(img), "skill": s, "p": round(p, 4),
                                    "mass": round(mass, 4), "y": bool(labels[str(s)])}) + "\n")
    rows = [json.loads(l) for l in open(out)]
    print(f"画像 {len({r['img'] for r in rows})} 枚 × 問い → {len(rows)} 組")
    for s in skills + ["全体"]:
        rr = rows if s == "全体" else [r for r in rows if r["skill"] == s]
        pos = [r["p"] for r in rr if r["y"]]
        neg = [r["p"] for r in rr if not r["y"]]
        acc = sum((r["p"] >= 0.5) == r["y"] for r in rr) / max(1, len(rr))
        a = auc(pos, neg)
        print(f"  {s!s:>4} {CHECK_Q.get(s, '')[:24]:24s} 正 {len(pos):3d} 負 {len(neg):3d}  "
              f"AUC {a if a is None else round(a, 3)}  正解率(0.5) {acc:.2f}")


if __name__ == "__main__":
    main()
