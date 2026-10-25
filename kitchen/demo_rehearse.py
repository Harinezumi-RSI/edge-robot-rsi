"""デモ録画の前の下見: 候補の発話を コーチの声(TTS)→ 聞き取り(STT)→ 頭の手順(場面の画像つき)に通すだけ(ロボットは動かさない)。
.venv で実行(頭のサーバ brain_server.py が要る)。枠ごとに、決めておいた条件を満たす最初の候補を選ぶ(選び方は下見の前に決める)。

  .venv/bin/python kitchen/demo_rehearse.py --spec <下見の表.json> --out <選んだ発話.json>

下見の表: [{"name": 枠の名前, "image": 場面の画像, "plan": 期待する手順 or null, "ask": 聞き返すべきか, "candidates": [発話, ...]}, ...]
動画では「発話は下見で選んだ(頭と耳が読み違えないもの)・撮るのは 1 回」と書く。
"""
import argparse
import json
import os
import urllib.request

BRAIN = os.environ.get("PAI_BRAIN", "http://127.0.0.1:8765")


def ask(path, payload):  # kitchen/demo.py の ask と同じ(重い import を避ける)
    req = urllib.request.Request(BRAIN + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    chosen, log = {}, []
    for slot in json.load(open(a.spec)):
        pick = None
        for utt in slot["candidates"]:
            r = ask("/tts", {"text": utt, "who": "coach"})
            h = ask("/stt", {"wav": r["wav"], "fix": False})
            p = ask("/plan", {"utterance": h["text"], "image": os.path.abspath(slot["image"])})
            ok = bool(p.get("ask")) == slot["ask"] and (slot["plan"] is None or sorted(p.get("plan") or []) == sorted(slot["plan"]))
            row = {"枠": slot["name"], "発話": utt, "聞き取り": h["text"], "手順": p.get("plan"), "済み": p.get("done"),
                   "聞き返す": bool(p.get("ask")), "p_ok": p.get("p_ok"), "返事": p.get("reply"), "合う": ok}
            log.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if ok:
                pick = utt
                break
        chosen[slot["name"]] = pick
    json.dump({"chosen": chosen, "log": log}, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("選んだ:", json.dumps(chosen, ensure_ascii=False))


if __name__ == "__main__":
    main()
