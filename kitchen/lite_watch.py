"""目で見る: 1 試行を動かして一定間隔でコマを撮り、1 枚の一覧画像(コンタクトシート)にする。.venv で実行。

  python kitchen/lite_watch.py --hand models/lite/d1.npz --skill 0 --init 0 --out outputs/watch/d1_s0_i0.jpg
  python kitchen/lite_watch.py --hand smolvla --skill 0 --init 0 --out outputs/watch/teacher_s0_i0.jpg     # 先生

コマには step・グリッパ(開/閉)・手先の高さを書く。最後のコマはホームへ戻す前の姿。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from lerobot.envs.libero import get_task_init_states  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from kitchen.chain_env import KitchenChainEnv  # noqa: E402
from kitchen.demo import FONT, LiteHand, SmolHand  # noqa: E402
from kitchen.skills import JA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", required=True)
    ap.add_argument("--skill", type=int, required=True)
    ap.add_argument("--init", type=int, required=True)
    ap.add_argument("--every", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--camera", default="agentview")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.init)
    env = KitchenChainEnv(start_task=1, init_index=0)
    env._init_states = get_task_init_states(env._suite, a.skill)
    env.init_state_id = a.init
    obs, _ = env.reset()
    env.set_goal(a.skill)
    hand = SmolHand(5) if a.hand == "smolvla" else LiteHand(a.hand)
    hand.reset()
    f = ImageFont.truetype(FONT, 15)
    frames, ok = [], False

    def grab(t):
        im = Image.fromarray(env.hires(camera=a.camera, width=320, height=240))
        d = ImageDraw.Draw(im)
        g = obs["robot_state"]["gripper"]["qpos"]
        z = obs["robot_state"]["eef"]["pos"][2]
        d.rectangle([0, 0, 320, 20], fill=(0, 0, 0))
        d.text((4, 2), f"step {t}  指 {'閉' if g[0] < 0.03 else '開'}({g[0]:.3f})  高さ {z:.3f}", font=f, fill=(255, 255, 255))
        frames.append(im)

    for t in range(a.max_steps):
        if t % a.every == 0:
            grab(t)
        obs, _, _, _, info = env.step(hand.act(obs, a.skill))
        if info["is_success"]:
            ok = True
            break
    grab(t + 1)
    cols = 6
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 320, rows * 240 + 34), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    d.text((8, 6), f"{os.path.basename(a.hand)}・動作 {a.skill}「{JA[a.skill]}」・初期状態 {a.init} → {'成功' if ok else '失敗'}({t + 1} step)",
           font=ImageFont.truetype(FONT, 20), fill=(0, 0, 0))
    for i, im in enumerate(frames):
        sheet.paste(im, ((i % cols) * 320, 34 + (i // cols) * 240))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sheet.save(a.out, quality=85)
    print(a.out, "成功" if ok else "失敗", t + 1)


if __name__ == "__main__":
    main()
