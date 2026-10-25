"""見てから察するの評価。.venv-mlx で実行(頭と改造 Jev は同じモデルを共有する)。

  python kitchen/scene_eval.py data/utterances_scene.jsonl dev            → outputs/scene/dev/(調整に使う 12 本)
  python kitchen/scene_eval.py data/utterances_scene_test.jsonl test 5   → outputs/scene/test/(1 回だけ。画像の選び方を 5 ずらす)

場面 3 種(画像はすべて評価専用の初期状態 0〜9 のもの):
  初期      : 何もしていない場面(data/scene/init_i*.jpg、scene_render.py)
  一部済み  : 頼まれた目標の 1 つがすでに満たされている場面(SmolVLA の評価 val_all の試行後の画像)
  関係ない  : 頼まれていない目標だけが満たされている場面(同じく val_all)
正解 = 発話の正解の手順から、その場面ですでに満たされている目標を外したもの(順番は保つ)。
比べる 4 つ: 場面を見ない(plan)/ 様子を言葉で渡す(plan_seen、Jev の状態)/ 引き算(plan_subtract、Jev の状態)/ 引き算(本当の状態 = 上限)。
発話は新しく書いたもの(手順づくりの手本・ほかの評価と重ならない)。調整用 12 本で直してから、別の 12 本で 1 回だけ測る。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kitchen import brain, jev  # noqa: E402
from kitchen.runlog import Timer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rows(p):
    return [json.loads(l) for l in open(os.path.join(ROOT, p))]


def case(u, img, labels, kind):
    done = sorted(int(g) for g, t in labels.items() if t)
    ok = [[p for p in plan if p not in done] for plan in u["ok"]]
    ok = [list(x) for x in dict.fromkeys(tuple(x) for x in ok)]
    return {"id": u["id"], "text": u["text"], "img": img, "kind": kind, "done_true": done, "ok": ok}


def subtract(r, done):
    keep = [g for g in r["plan"] if g not in set(done)]
    return {**r, "plan": keep}


def cases(utt="data/utterances_scene.jsonl", shift=0):
    U, init, V = rows(utt), rows("data/scene/init.jsonl"), rows("outputs/fast/val_all.jsonl")
    out = []
    for k0, u in enumerate(U):
        k = k0 + shift
        P = set(u["ok"][0])
        r = init[k % len(init)]
        out.append(case(u, f"data/scene/{r['img']}", r["labels"], "初期"))
        for g in sorted(P):
            c = [v for v in V if v["skill"] == g and v["labels"][str(g)]]
            if c:
                v = c[k % len(c)]
                out.append(case(u, f"data/snaps/val_all/{v['img']}", v["labels"], "一部済み"))
        c = [v for v in V if v["skill"] not in P and v["labels"][str(v["skill"])] and not any(v["labels"][str(p)] for p in P)]
        v = c[(7 * k) % len(c)]
        out.append(case(u, f"data/snaps/val_all/{v['img']}", v["labels"], "関係ない"))
    return out


def main():
    utt, name = sys.argv[1], sys.argv[2]
    C = cases(utt, int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    d = os.path.join(ROOT, "outputs/scene", name)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, "results.jsonl")
    open(out, "w").close()
    with Timer("eval", f"見てから察する({name}): 場面を見ない計画 vs 改造 Jev で読んだ状態 vs 本当の状態", cases=len(C)) as tm:
        for c in C:
            t = time.time()
            p = jev.predict(os.path.join(ROOT, c["img"]))
            c["p_jev"] = [round(float(x), 3) for x in p]
            c["done_jev"] = [g for g in range(10) if p[g] >= 0.5]
            c["sec_jev"] = round(time.time() - t, 2)
            plain = brain.plan(c["text"])  # 引き算の 2 つは同じ生成を使う(温度 0 なので同じ答え)
            for name, fn in (("plain", lambda: plain), ("prompt_jev", lambda: brain.plan_seen(c["text"], c["done_jev"])),
                             ("jev", lambda: subtract(plain, c["done_jev"])), ("true", lambda: subtract(plain, c["done_true"]))):
                t = time.time()
                r = fn()
                c[name] = {"plan": r["plan"], "reply": r["reply"], "ok": r["plan"] in c["ok"], "sec": round(time.time() - t, 2),
                           "extra": [g for g in r["plan"] if g in c["done_true"]],
                           "missing": [g for g in c["ok"][0] if g not in r["plan"]]}
            with open(out, "a") as f:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
            print(f"{c['id']} {c['kind']:4s} 済み {c['done_true']} Jev {c['done_jev']} → 見ない {c['plain']['plan']}"
                  f"{'○' if c['plain']['ok'] else '×'} / 言葉で渡す {c['prompt_jev']['plan']}{'○' if c['prompt_jev']['ok'] else '×'}"
                  f" / 引き算(Jev) {c['jev']['plan']}{'○' if c['jev']['ok'] else '×'}"
                  f" / 引き算(本当) {c['true']['plan']}{'○' if c['true']['ok'] else '×'}", flush=True)
        summ = {}
        for name in ("plain", "prompt_jev", "jev", "true"):
            summ[name] = {k: f"{sum(c[name]['ok'] for c in C if c['kind'] == k)}/{sum(c['kind'] == k for c in C)}"
                          for k in ("初期", "一部済み", "関係ない")}
            summ[name]["合計"] = f"{sum(c[name]['ok'] for c in C)}/{len(C)}"
            summ[name]["やらなくてよい動作"] = sum(len(c[name]["extra"]) for c in C)
            summ[name]["抜けた動作"] = sum(len(c[name]["missing"]) for c in C)
        summ["Jev の状態の読み"] = f"{sum(c['done_jev'] == c['done_true'] for c in C)}/{len(C)} 場面で完全一致"
        tm.info.update({f"{k}_合計": v["合計"] for k, v in summ.items() if isinstance(v, dict)})
    json.dump(summ, open(os.path.join(d, "summary.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
