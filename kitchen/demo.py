"""デモ: ひとこと → 手順(頭)→ 動く前の確認(Jev 風)→ 各動作(手)→ ホームへ戻す → 出来の確認(改造 Jev)→ だめならやり直す。
録画して、右に吹き出し・手順の進み・判定の確率の棒を描いた動画を作る。.venv で実行(頭は brain_server.py に問い合わせる)。

  .venv-mlx/bin/python kitchen/brain_server.py &             # 先に頭を起動
  .venv/bin/python kitchen/demo.py --utterance "寒くなってきたね、何かあったかいもの食べたいな" --init 3 --out outputs/demo/samui
  見てから察する(続けて話しかける。場面はそのまま。話しかけられたら場面を撮って改造 Jev で済みの目標を読み、手順から引く):
  .venv/bin/python kitchen/demo.py --seen --utterance "コンロの火つけといて" "寒くなってきたね、何かあったかいもの食べたいな" --init 3 --out outputs/demo/seen

動画の速さ: シミュレーションは 20Hz。録画は 1 step 1 枚で、出力は 40fps(2 倍速、動画の中に明記)。
"""
import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from kitchen.chain_env import KitchenChainEnv, fixture_seed  # noqa: E402
from kitchen.run_chain import go_home  # noqa: E402
from kitchen.skills import CHECK_Q, INSTRUCTION, JA, STATE_JA  # noqa: E402
from kitchen.vla import VLA  # noqa: E402


class SmolHand:
    """手 = 公開の SmolVLA(先生)。"""

    def __init__(self, n_action_steps):
        self.vla = VLA(n_action_steps=n_action_steps)

    def reset(self):
        self.vla.reset()

    def act(self, obs, skill):
        return self.vla.act(obs, INSTRUCTION[skill])


class LiteHand:
    """手 = 軽い手(DINOv2-small + 頭 6MB)。Claude のループ(DAgger)で鍛えたもの。"""

    def __init__(self, head, n_exec=4):
        from kitchen.lite_eval import head_act, load_head
        if head.endswith(".json"):  # 動作ごとに頭を選ぶ対応表
            m = json.load(open(head))
            cache = {p: load_head(p) for p in set(m.values())}
            self.by_skill = {int(k): [cache[p]] for k, p in m.items()}
        else:
            self.by_skill = None
            self.hs = [load_head(h) for h in head.split(",")]  # 「,」で区切ると頭のアンサンブル(行動を平均)
        self.n, self.q, self._act = n_exec, [], head_act
        self.explore, self.rng = 0.0, None

    def set_explore(self, sigma, seed=0):
        """揺らぎ(塊ごとに同じ向き・グリッパには入れない。自己進化の練習 lite_batch.py --explore と同じ形)。sigma=0 で切る。"""
        self.explore, self.rng = sigma, np.random.default_rng(seed)

    def reset(self):
        self.q = []

    def act(self, obs, skill):
        if not self.q:
            hs = self.by_skill[skill] if self.by_skill is not None else self.hs
            q = np.mean([self._act(h, obs, skill) for h in hs], axis=0)[: self.n]
            if self.explore > 0:
                nz = self.rng.normal(0.0, self.explore, q.shape[1])
                nz[6] = 0.0
                q = q + nz
            self.q = list(q)
        return np.clip(np.asarray(self.q.pop(0), dtype=np.float32), -1, 1)

FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
FONT_B = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
BRAIN = os.environ.get("PAI_BRAIN", "http://127.0.0.1:8765")
STT_FIX = os.environ.get("PAI_STT_FIX", "0") == "1"  # 聞き取りの後処理(頭が聞き間違いを直す)。意味まで書き換えるので既定は切る(26 周目)
W, H, PANEL = 640, 480, 420


def ask(path, payload):
    req = urllib.request.Request(BRAIN + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


class Overlay:
    """右側の板: ひとこと・返事・手順の進み・最後の判定の棒・速さ。"""

    def __init__(self, utterance):
        self.f = ImageFont.truetype(FONT, 17)
        self.fs = ImageFont.truetype(FONT, 14)
        self.fb = ImageFont.truetype(FONT_B, 18)
        self.utt, self.reply, self.plan, self.status = utterance, "", [], {}
        self.bars, self.note, self.check, self.seen = None, "", None, ""
        self.speaker, self.heard = "あなた", ""

    def _wrap(self, d, text, font, width):
        lines, cur = [], ""
        for ch in text:
            if d.textlength(cur + ch, font=font) > width:
                lines.append(cur)
                cur = ch
            else:
                cur += ch
        return lines + ([cur] if cur else [])

    def draw(self, frame):
        img = Image.new("RGB", (W + PANEL, H), (248, 247, 243))
        img.paste(Image.fromarray(frame), (0, 0))
        d = ImageDraw.Draw(img)
        x, y = W + 16, 14
        d.text((x, y), self.speaker, font=self.fs, fill=(110, 110, 110))
        y += 20
        for line in self._wrap(d, f"「{self.utt}」", self.fb, PANEL - 32):
            d.text((x, y), line, font=self.fb, fill=(30, 30, 30))
            y += 24
        if self.heard:
            for line in self._wrap(d, self.heard, self.fs, PANEL - 32):
                d.text((x, y + 2), line, font=self.fs, fill=(60, 110, 60))
                y += 19
        if self.seen:
            for line in self._wrap(d, self.seen, self.fs, PANEL - 32):
                d.text((x, y + 2), line, font=self.fs, fill=(120, 90, 30))
                y += 19
        if self.reply:
            y += 6
            d.text((x, y), "ロボット", font=self.fs, fill=(110, 110, 110))
            y += 20
            for line in self._wrap(d, self.reply, self.f, PANEL - 32):
                d.text((x, y), line, font=self.f, fill=(20, 80, 160))
                y += 22
        if self.check is not None:
            d.text((x, y + 2), f"動く前の確認: 手順が望みに合う確率 {self.check:.2f}", font=self.fs, fill=(90, 90, 90))
            y += 22
        y += 8
        for i, s in enumerate(self.plan):
            st = self.status.get(i, "")
            mark, col = {"run": ("▶", (200, 120, 0)), "ok": ("✓", (30, 140, 60)), "retry": ("↻", (200, 60, 60)),
                         "ng": ("✗", (200, 60, 60))}.get(st, ("・", (120, 120, 120)))
            d.text((x, y), f"{mark} {i + 1}. {JA[s]}", font=self.f, fill=col)
            y += 24
        if self.bars:
            y += 8
            d.text((x, y), "改造 Jev(画像 1 回で判定)", font=self.fs, fill=(90, 90, 90))
            y += 20
            for label, p in self.bars:
                d.rectangle([x, y + 3, x + 160, y + 15], outline=(180, 180, 180))
                d.rectangle([x, y + 3, x + int(160 * p), y + 15], fill=(30, 140, 60) if p >= 0.5 else (200, 60, 60))
                d.text((x + 168, y), f"{p:.2f} {label}", font=self.fs, fill=(40, 40, 40))
                y += 20
        d.text((x, H - 26), self.note, font=self.fs, fill=(120, 120, 120))
        return np.array(img)


def mux(mp4, audio, fps):
    """声を動画に重ねる(それぞれ話し始めの枚目の時刻に置く)。"""
    import subprocess
    tmp = mp4.replace(".mp4", "_novoice.mp4")
    os.replace(mp4, tmp)
    ins = sum([["-i", w] for _, w in audio], [])
    fc = ";".join(f"[{i + 1}:a]adelay={int(f / fps * 1000)}|{int(f / fps * 1000)}[a{i}]" for i, (f, _) in enumerate(audio))
    fc += ";" + "".join(f"[a{i}]" for i in range(len(audio))) + f"amix=inputs={len(audio)}:normalize=0[aout]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, *ins, "-filter_complex", fc, "-map", "0:v", "-map", "[aout]",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", mp4], check=True)
    os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utterance", nargs="+", required=True, help="続けて話しかける(場面はそのまま)")
    ap.add_argument("--seen", action="store_true", help="話しかけられたら場面を撮り、改造 Jev で済みの目標を読んで手順から引く")
    ap.add_argument("--voice", action="store_true", help="話しかけるのはパソコンの Claude(コーチの声)。ロボットは声を聞き取り(Qwen3-ASR)、声で返事(Irodori-TTS)")
    ap.add_argument("--react", nargs="*", default=[], help="--voice のとき、各回の後にコーチが言うひとこと(回ごと、空文字で無言)")
    ap.add_argument("--hand", default="smolvla", help="smolvla か、軽い手の頭のパス(models/lite/d1.npz など)")
    ap.add_argument("--init", type=int, default=0)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--tau", type=float, default=0.5, help="出来の確認のしきい値(改造 Jev の確率)")
    ap.add_argument("--n-action-steps", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    ov = Overlay(a.utterance[0])
    ov.note = "Mac mini 1 台・動画は 2 倍速(話す場面は等速)" if a.voice else "Mac 1 台(Apple Silicon)・動画は 2 倍速"
    frames, log = [], {"utterance": a.utterance, "init": a.init, "seen": a.seen, "hand": a.hand, "turns": []}
    t0 = time.time()
    torch.manual_seed(a.init)
    env = KitchenChainEnv(start_task=1, init_index=a.init)
    obs, _ = env.reset()
    env.place_fixtures(fixture_seed(1, a.init))  # 備え付けを種で置く(撮り直しても同じ場面)
    home = (obs["robot_state"]["eef"]["pos"].copy(), obs["robot_state"]["eef"]["quat"].copy())
    vla = None

    def rec(e, s, t):
        frames.append(ov.draw(e.hires(width=W, height=H)))

    audio = []  # (何枚目から, wav)

    def speak(text, who):
        """声を作って流す。話している間は場面を止める(動画は 40fps で、話す場面は等速)。"""
        r = ask("/tts", {"text": text, "who": who})
        audio.append((len(frames), r["wav"]))
        still = ov.draw(env.hires(width=W, height=H))
        frames.extend([still] * int(r["sec"] * 40 + 12))
        return r

    for ui, utt in enumerate(a.utterance):
        ov.utt, ov.reply, ov.plan, ov.status, ov.bars, ov.check, ov.seen, ov.heard = utt, "", [], {}, None, None, "", ""
        turn = {"utterance": utt, "steps": []}
        log["turns"].append(turn)
        if a.voice:
            ov.speaker = "パソコンの Claude(コーチ)"
            r = speak(utt, "coach")
            heard = ask("/stt", {"wav": r["wav"], "fix": STT_FIX})
            turn["heard"] = heard
            ov.heard = f"ロボットが聞き取った({heard['sec']:.1f} 秒): 「{heard['text']}」"
            utt_for_plan = heard["text"]
        else:
            utt_for_plan = utt
        for _ in range(30):
            frames.append(ov.draw(env.hires(width=W, height=H)))
        payload = {"utterance": utt_for_plan}
        if a.seen:
            scene = a.out + f"_u{ui}_scene.jpg"
            imageio.imwrite(scene, env.hires(width=640, height=480), quality=92)
            payload["image"] = os.path.abspath(scene)
        p = ask("/plan", payload)
        turn["plan"] = p
        if a.seen:
            ov.seen = "見えている: " + ("・".join(STATE_JA[g] for g in p.get("done", [])) or "変わったところなし")
        ov.reply, ov.plan, ov.check = p["reply"], p["plan"], p.get("p_ok")
        if p.get("ask"):
            ov.reply = (p["reply"] + " ……念のため確認させてください。この手順でよいですか?").strip()
        if a.voice and ov.reply:
            speak(ov.reply, "robot")
        if not p["plan"] or p.get("ask"):
            for _ in range(80):
                frames.append(ov.draw(env.hires(width=W, height=H)))
            if a.voice and ui < len(a.react) and a.react[ui]:
                speak(a.react[ui], "coach")
            continue
        if vla is None:
            vla = SmolHand(a.n_action_steps) if a.hand == "smolvla" else LiteHand(a.hand)
        gave_up = False
        for k, s in enumerate(p["plan"]):
            for attempt in range(a.retries + 1):
                ov.status[k] = "run" if attempt == 0 else "retry"
                if attempt > 0:
                    obs, _ = go_home(env, obs, home, on_frame=rec)
                env.set_goal(s)
                vla.reset()
                for t in range(300):
                    obs, _, _, _, info = env.step(vla.act(obs, s))
                    rec(env, s, t)
                    if info["is_success"]:
                        break
                obs, _ = go_home(env, obs, home, on_frame=rec)
                snap = a.out + f"_u{ui}_k{k}_a{attempt}.jpg"
                imageio.imwrite(snap, env.hires(width=640, height=480), quality=92)
                j = ask("/jev", {"image": os.path.abspath(snap)})
                pj = j["probs"][s]
                ov.bars = [(CHECK_Q[s].replace("ですか?", "").replace("いますか?", "いる"), pj)]
                turn["steps"].append({"k": k, "skill": s, "attempt": attempt, "p_jev": pj, "sim_ok": env.goal_reached(s), "sec_jev": j["sec"]})
                for _ in range(15):
                    frames.append(ov.draw(env.hires(width=W, height=H)))
                if pj >= a.tau:
                    ov.status[k] = "ok"
                    break
                ov.status[k] = "ng"
            if ov.status.get(k) == "ng":
                # やり直しても同じ状況からでは直らないことが多い(閉ループの通し評価で 0/6)。続けずに正直に伝える
                ov.reply = f"ごめんなさい、「{JA[s]}」がうまくできませんでした。手を貸してもらえますか?"
                for _ in range(40):
                    frames.append(ov.draw(env.hires(width=W, height=H)))
                turn["gave_up_at"] = k
                gave_up = True
                if a.voice:
                    speak(ov.reply, "robot")
                break
        if gave_up:
            break
        if a.voice:
            speak("できました。", "robot")
            if ui < len(a.react) and a.react[ui]:
                ov.reply = ""
                speak(a.react[ui], "coach")
    sec = time.time() - t0
    ov.note = (f"Mac mini 1 台・動画は 2 倍速(話す場面は等速)/ かかった時間 {sec:.0f} 秒" if a.voice else
               f"Mac 1 台(Apple Silicon)・動画は 2 倍速 / かかった時間 {sec:.0f} 秒")
    for _ in range(30):
        frames.append(ov.draw(frames[-1][:, :W] if frames else np.zeros((H, W, 3), np.uint8)))
    imageio.mimwrite(a.out + ".mp4", frames, fps=40, quality=8)
    if audio:
        mux(a.out + ".mp4", audio, 40)
    log.update(sec=round(sec, 1), frames=len(frames))
    json.dump(log, open(a.out + ".json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in log.items() if k != "turns"}, ensure_ascii=False))
    for t in log["turns"]:
        print(t["utterance"], "→", t["plan"].get("plan"), t["plan"].get("reply"), "見えた", t["plan"].get("done"))


if __name__ == "__main__":
    main()
