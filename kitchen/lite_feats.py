"""軽い手の特徴を、土台(backend)ごとに取り出す。.venv で実行。

  qwen08  : Qwen3.5-0.8B 4bit(lite_server の /feat)。層ごとの要約 F(6/12/18 層目の末尾・画像平均)+ 12 層目の 4×4 マス G [32, 1024]
  qwen08v : 同じモデルの視覚部分だけ(/vfeat、言語モデルを通さない)。4×4 マス + 画像平均 G [34, 1024] + 指示の one-hot
  dino    : DINOv2-small(言語なしの自己教師あり、0.09GB、このプロセスの PyTorch)。4×4 マス + CLS G [34, 384] + 指示の one-hot

返り値はどれも (F [種類, 次元], F の名前, G [マス, 次元])。比べたいのは「言語モデルの途中の層は制御に効くのか」。
"""
import base64
import io
import json
import os
import urllib.request

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from kitchen.lite_common import NAMES, pool_grid
from kitchen.skills import INSTRUCTION

LITE = os.environ.get("PAI_LITE", "http://127.0.0.1:8766")
ONEHOT = "指示_one-hot"
BACKENDS = ("qwen08", "qwen08v", "dino", "dino8", "dinob")
# DINOv2 の土台ごとの (モデル, マスの細かさ, 次元)。dino = small・4×4(0 周目から)、dino8 = small・8×8、dinob = base(86M)・4×4
DINO = {"dino": ("facebook/dinov2-small", 4, 384), "dino8": ("facebook/dinov2-small", 8, 384), "dinob": ("facebook/dinov2-base", 4, 768)}


def b64png(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def obs_images(obs):
    px = obs["pixels"]
    return [np.ascontiguousarray(px["image"][::-1, ::-1]), np.ascontiguousarray(px["image2"][::-1, ::-1])]


def obs_state(obs):
    e = obs["robot_state"]["eef"]
    aa = R.from_quat(e["quat"]).as_rotvec()
    return np.concatenate([e["pos"], aa, obs["robot_state"]["gripper"]["qpos"]]).astype(np.float32)


def post(path, payload):
    req = urllib.request.Request(LITE + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _arr(r, key, shape_key):
    return np.frombuffer(base64.b64decode(r[key]), dtype=np.float16).reshape(r[shape_key])


def onehot(k):
    v = np.zeros((1, 10), np.float16)
    v[0, k] = 1
    return v


_dino = {}


def dino_grid(imgs, backend="dino"):
    import torch
    name, g, _ = DINO[backend]
    if name not in _dino:
        from transformers import AutoModel
        _dino[name] = AutoModel.from_pretrained(name).to("mps").eval()
        _dino["mean"] = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        _dino["std"] = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div(255)
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
    x = ((x - _dino["mean"]) / _dino["std"]).to("mps")
    with torch.inference_mode():
        h = _dino[name](pixel_values=x).last_hidden_state.float().cpu().numpy()
    return np.concatenate([np.concatenate([pool_grid(h[i, 1:], 16, 16, g), h[i, :1]]) for i in range(len(imgs))])


def extract(backend, obs, skill, upto=None):
    imgs = obs_images(obs)
    if backend == "qwen08":
        r = post("/feat", {"images": [b64png(i) for i in imgs], "instruction": INSTRUCTION[skill], "upto": upto})
        F = _arr(r, "feat", "shape")
        return F, NAMES[:len(F)], _arr(r, "grid", "grid_shape")
    if backend == "qwen08v":
        r = post("/vfeat", {"images": [b64png(i) for i in imgs]})
        return onehot(skill), [ONEHOT], _arr(r, "grid", "grid_shape")
    if backend in DINO:
        return onehot(skill), [ONEHOT], dino_grid(imgs, backend).astype(np.float16)
    raise ValueError(backend)


def extract_batch(backend, obs_list, skills, upto=None):
    """台所 N 個分 → (F [N, 種類, 次元], 名前, G [N, マス, 次元])。DINOv2 は 2N 枚を 1 回の順伝播で、Qwen はサーバに 1 本ずつ。"""
    if backend in DINO:
        imgs = [im for o in obs_list for im in obs_images(o)]
        G = dino_grid(imgs, backend).astype(np.float16).reshape(len(obs_list), -1, DINO[backend][2])
        return np.concatenate([onehot(k)[None] for k in skills]), [ONEHOT], G
    out = [extract(backend, o, k, upto) for o, k in zip(obs_list, skills)]
    return np.stack([f for f, _, _ in out]), out[0][1], np.stack([g for _, _, g in out])
