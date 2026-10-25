"""ループの記録: 何周・何分・何本・何 step を docs/rsi_runs.jsonl に 1 行ずつ残す(レポートの材料)。

  from kitchen.runlog import Timer
  with Timer("inner", "練習 1 周目", episodes=240) as t: ...; t.info["ok"] = 150
  python kitchen/runlog.py add outer "1 周目: まとめて実行の比較" --start "2026-10-22 16:20" --end "2026-10-22 17:05" --note "..."
  python kitchen/runlog.py table      # 種類ごとの周の数・合計の分・試行の数の表
kind: outer(Claude が回す研究のループ)/ inner(ロボットの練習・採点・学習・評価)/ practice / eval / train / label(先生が答えをつける)/ judge / other
"""
import json
import os
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.environ.get("PAI_RUNLOG") or os.path.join(ROOT, "docs", "rsi_runs.jsonl")  # 試しの実行は PAI_RUNLOG で別の場所へ


def _rel(v):
    """記録にユーザー名入りの絶対パスを残さない(公開する記録なので)。"""
    if isinstance(v, str):
        return v.replace(ROOT + os.sep, "").replace(os.path.expanduser("~"), "~")
    if isinstance(v, list):
        return [_rel(x) for x in v]
    return v


def add(kind, name, start, end, **info):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    info = {k: _rel(v) for k, v in info.items()}
    row = {"kind": kind, "name": name, "start": start.strftime("%Y-%m-%d %H:%M:%S"),
           "end": end.strftime("%Y-%m-%d %H:%M:%S"), "minutes": round((end - start).total_seconds() / 60, 1), **info}
    with open(LOG, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


class Timer:
    def __init__(self, kind, name, **info):
        self.kind, self.name, self.info = kind, name, dict(info)

    def __enter__(self):
        self.start = datetime.now()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.info["error"] = repr(exc)[:200]
        add(self.kind, self.name, self.start, datetime.now(), **self.info)


def table():
    rows = [json.loads(l) for l in open(LOG)] if os.path.exists(LOG) else []
    print("| 種類 | 回数 | 合計(分) | 試行 | 学習 step |")
    print("|---|---|---|---|---|")
    for k in ("outer", "inner", "practice", "eval", "train", "label", "judge", "other"):
        rr = [r for r in rows if r["kind"] == k]
        if rr:
            print(f"| {k} | {len(rr)} | {sum(r['minutes'] for r in rr):.0f} | {sum(r.get('episodes', 0) for r in rr)} "
                  f"| {sum(r.get('train_steps', 0) for r in rr)} |")
    print(f"\n全部で {len(rows)} 件、{sum(r['minutes'] for r in rows):.0f} 分")
    for r in rows:
        extra = {k: v for k, v in r.items() if k not in ("kind", "name", "start", "end", "minutes")}
        print(f"- [{r['kind']}] {r['start'][5:16]}〜{r['end'][11:16]}({r['minutes']} 分){r['name']} {json.dumps(extra, ensure_ascii=False) if extra else ''}")


if __name__ == "__main__":
    if sys.argv[1] == "table":
        table()
    elif sys.argv[1] == "add":
        kind, name = sys.argv[2], sys.argv[3]
        opts = dict(zip(sys.argv[4::2], sys.argv[5::2]))
        st = datetime.strptime(opts.pop("--start"), "%Y-%m-%d %H:%M")
        en = datetime.strptime(opts.pop("--end"), "%Y-%m-%d %H:%M") if "--end" in opts else datetime.now()
        opts.pop("--end", None)
        info = {k.lstrip("-"): (json.loads(v) if v[:1] in "[{0123456789" else v) for k, v in opts.items()}
        print(add(kind, name, st, en, **info))
