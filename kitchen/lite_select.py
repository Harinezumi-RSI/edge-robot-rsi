"""動作ごとに頭を選ぶ: outputs/lite/sel/<頭>_<条件>/val.jsonl(判断用の初期状態 0〜9 × ずれなし・小さなずれ 2 通り)から、
動作ごとに成功が一番多い頭を選び、対応表 models/lite/route.json を書く(同点は学習データの多い頭 = 候補の並びで後ろを優先)。.venv で実行。
"""
import argparse
import collections
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["l0dino", "l0f", "l0f_s1", "l0f_s2", "d1", "d1_s1", "d1_s2", "d1_s3", "d2", "d1f", "d1f_s1", "d1f_s2",
         "l0o"]  # l0o = OFT の練習を写すだけ(判断用 90/100)。評価が 300 試行そろっていなければ自動で外れる


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="あとから足した候補の頭(並びの後ろに足す)")
    ap.add_argument("--out", default="models/lite/route.json")
    a = ap.parse_args()
    order = ORDER + a.extra
    tab = collections.defaultdict(lambda: collections.Counter())
    n = collections.Counter()
    for p in glob.glob(os.path.join(ROOT, "outputs/lite/sel/*/val.jsonl")):
        h = os.path.basename(os.path.dirname(p)).rsplit("_", 1)[0]
        for r in map(json.loads, open(p)):
            tab[h][r["skill"]] += r["ok"]
            n[h] += 1
    heads = [h for h in order if n[h] >= 300]
    print("頭ごと(300 試行):", {h: sum(tab[h].values()) for h in heads})
    route, best_total = {}, 0
    for s in range(10):
        best = max(heads, key=lambda h: (tab[h][s], heads.index(h)))
        route[str(s)] = f"models/lite/{best}.npz"
        best_total += tab[best][s]
        print(f"  動作 {s}: {best}({tab[best][s]}/30)  " + " ".join(f"{h}:{tab[h][s]}" for h in heads))
    json.dump(route, open(os.path.join(ROOT, a.out), "w"), indent=1)
    single = max(heads, key=lambda h: sum(tab[h].values()))
    print(f"動作ごとに選んだ頭: 判断用で {best_total}/300(選んだ集合の上での値なので甘い)/ 1 つの頭の最良 {single} {sum(tab[single].values())}/300")


if __name__ == "__main__":
    main()
