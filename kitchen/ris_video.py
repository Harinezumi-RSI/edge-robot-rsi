"""ミニ RIS の比べ動画: 同じずれた初期姿勢から、元の手と自分で練習した手(差分)を左右に並べる。.venv で実行。

  python kitchen/ris_video.py --skill 4 --init 41 --sigma 0.1 --delta models/rsi/ris2.safetensors --out outputs/demo/ris_s4_i41

動画は説明用(数字は評価から)。1 本ずつ動かすので、まとめて回した評価とはノイズの引き方が違い、同じ初期状態でも結果が変わることがある。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from lerobot.envs.libero import get_task_init_states  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from kitchen.chain_env import KitchenChainEnv  # noqa: E402
from kitchen.skills import INSTRUCTION, JA  # noqa: E402
from kitchen.vla import VLA  # noqa: E402

FONT = "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"
W, H = 480, 360


def rollout(vla, skill, init, sigma, seed, max_steps=300):
    env = KitchenChainEnv(start_task=skill, init_index=init)
    env._init_states = get_task_init_states(env._suite, skill)
    env.init_state_id = init
    torch.manual_seed(seed)
    obs, _ = env.reset()
    env.set_goal(skill)
    obs = env.perturb_robot(sigma, np.random.default_rng(seed))
    frames = [env.hires(width=W, height=H)] * 20  # ずれた初期姿勢を少し見せる
    vla.reset()
    ok = False
    for t in range(max_steps):
        obs, _, _, _, info = env.step(vla.act(obs, INSTRUCTION[skill]))
        frames.append(env.hires(width=W, height=H))
        if info["is_success"]:
            ok = True
            break
    env.close()
    return frames, ok


def label(frame, text, sub, color):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    f, fs = ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 15)
    d.rectangle([0, 0, W, 54], fill=(255, 255, 255))
    d.text((10, 4), text, font=f, fill=color)
    d.text((10, 30), sub, font=fs, fill=(80, 80, 80))
    return np.array(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", type=int, required=True)
    ap.add_argument("--init", type=int, required=True)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--delta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else 10_000 * a.skill + a.init
    vla = VLA(n_action_steps=5)
    base, ok_b = rollout(vla, a.skill, a.init, a.sigma, seed)
    vla.policy.load_state_dict(load_file(a.delta), strict=False)
    mine, ok_m = rollout(vla, a.skill, a.init, a.sigma, seed)
    n = max(len(base), len(mine)) + 30
    base += [base[-1]] * (n - len(base))
    mine += [mine[-1]] * (n - len(mine))
    out = []
    for fb, fm in zip(base, mine):
        lb = label(fb, "元の手(公開重みのまま)", f"初期姿勢を約 10cm ずらした / {JA[a.skill]}", (60, 60, 60))
        lm = label(fm, "自分で練習して学んだ手", "練習→自分で採点→学習を 2 周(Mac 1 台)", (20, 90, 170))
        out.append(np.concatenate([lb, lm], axis=1))
    for k in range(1, 31):  # 最後の 30 枚に結果を描く
        img = Image.fromarray(out[-k])
        d = ImageDraw.Draw(img)
        f = ImageFont.truetype(FONT, 28)
        d.text((W // 2 - 40, H - 50), "成功" if ok_b else "失敗", font=f, fill=(30, 140, 60) if ok_b else (200, 60, 60))
        d.text((W + W // 2 - 40, H - 50), "成功" if ok_m else "失敗", font=f, fill=(30, 140, 60) if ok_m else (200, 60, 60))
        out[-k] = np.array(img)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    imageio.mimwrite(a.out + ".mp4", out, fps=40, quality=8)
    print({"skill": a.skill, "init": a.init, "base_ok": ok_b, "mine_ok": ok_m, "frames": len(out)})


if __name__ == "__main__":
    main()
