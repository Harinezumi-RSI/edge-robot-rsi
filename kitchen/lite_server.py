"""軽い手(Qwen3.5-0.8B 4bit を固定 + 小さな行動の頭)のサーバ。.venv-mlx で実行(手を動かす側は .venv から HTTP で呼ぶ)。

  python kitchen/lite_server.py --port 8766

  POST /feat  {"images": [png_b64, png_b64], "instruction": "..."}                → 途中の層の特徴(学習データ作り)
  POST /act   {"images": [...], "instruction": "...", "state": [8 次元]}           → 8 step 分の行動(学習した頭)

改造 Jev と同じ作り方(小さな VLM の途中の層 + 小さな頭)で、判定ではなく行動を出す。
"""
import argparse
import base64
import io
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from PIL import Image  # noqa: E402
from pydantic import BaseModel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("PAI_LITE_MODEL", "mlx-community/Qwen3.5-0.8B-MLX-4bit")
HEAD = os.path.join(ROOT, "models", "lite_head.npz")
from kitchen.lite_common import GRID_LAYER, LAYERS, NAMES, head_forward, pool_grid  # noqa: E402
app = FastAPI()
_m = {}
_gpu = threading.Lock()  # FastAPI は同期の口をスレッドで並べるので、モデルの計算は 1 本ずつ


def load():
    if not _m:
        from mlx_vlm import load as vload
        _m["model"], _m["proc"] = vload(MODEL)
    return _m["model"], _m["proc"]


def features(images, instruction, upto=None):
    """各層(6/12/18、upto 指定ならその層まで)の最後のトークンと画像トークンの平均 [種類, 1024] と、
    12 層目の画像トークンを 4×4 のマスに平均したもの [32, 1024](12 層目まで計算したときだけ)。"""
    import mlx.core as mx
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs
    model, proc = load()
    cfg = model.config
    msgs = [{"role": "user", "content": f"ロボットへの指示: {instruction}\n正面とグリッパの画像から、次の動きを決める。"}]
    prompt = apply_chat_template(proc, cfg, msgs, num_images=len(images), enable_thinking=False)
    tok_id = getattr(cfg, "image_token_index", None) or getattr(cfg, "image_token_id", None)
    inp = prepare_inputs(proc, images=images, prompts=prompt, image_token_index=tok_id)
    kw = {k: v for k, v in inp.items() if k not in ("input_ids", "pixel_values", "attention_mask")}
    layers = LAYERS if upto is None else tuple(L for L in LAYERS if L <= upto)
    lm = model.language_model.model
    full = lm.layers
    try:
        lm.layers = full[: max(layers) + 1]
        out = model(inp["input_ids"], pixel_values=inp["pixel_values"], mask=inp.get("attention_mask"),
                    capture_layer_ids=list(layers), return_hidden=True, skip_logits=True, **kw)
    finally:
        lm.layers = full
    is_img = (inp["input_ids"][0] == tok_id).astype(mx.float32)
    fs, grid = [], None
    for L, h in zip(layers, out.hidden_states[: len(layers)]):
        h = h[0].astype(mx.float32)
        fs += [h[-1], (h * is_img[:, None]).sum(0) / mx.maximum(is_img.sum(), 1)]
        if L == GRID_LAYER:
            grid = h
    f = mx.stack(fs)
    mx.eval(f)
    f = np.array(f, dtype=np.float32)
    if grid is not None:
        idx = np.where(np.array(inp["input_ids"][0]) == tok_id)[0]
        toks = np.array(grid[mx.array(idx)], dtype=np.float32)
        cells, o = [], 0
        for t, hh, ww in np.array(inp["image_grid_thw"]).tolist():
            n = t * (hh // 2) * (ww // 2)
            cells.append(pool_grid(toks[o:o + n], hh // 2, ww // 2))
            o += n
        grid = np.concatenate(cells)
    return f, grid


def decode(b64s):
    return [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in b64s]


class FeatReq(BaseModel):
    images: list
    instruction: str
    upto: int | None = None


class VReq(BaseModel):
    images: list


class ActReq(BaseModel):
    images: list
    instruction: str
    state: list
    head: str = ""


_heads = {}


def head(path=""):
    path = path or HEAD
    if path not in _heads:
        _heads[path] = dict(np.load(path, allow_pickle=True))
    return _heads[path]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.float16).tobytes()).decode()


@app.post("/feat")
def feat(r: FeatReq):
    t = time.time()
    with _gpu:
        f, g = features(decode(r.images), r.instruction, upto=r.upto)
    return {"feat": b64(f), "shape": list(f.shape), "grid": b64(g), "grid_shape": list(g.shape), "sec": round(time.time() - t, 3)}


def vision_grid(images):
    """視覚部分の出力(言語モデルに入る直前の画像トークン)だけ。画像ごとに 4×4 マス + 平均 → [34, 1024]"""
    import mlx.core as mx
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import prepare_inputs
    model, proc = load()
    cfg = model.config
    prompt = apply_chat_template(proc, cfg, [{"role": "user", "content": "x"}], num_images=len(images), enable_thinking=False)
    tok_id = getattr(cfg, "image_token_index", None) or getattr(cfg, "image_token_id", None)
    inp = prepare_inputs(proc, images=images, prompts=prompt, image_token_index=tok_id)
    dtype = model.vision_tower.patch_embed.proj.weight.dtype
    h, _ = model.vision_tower(inp["pixel_values"].astype(dtype), inp["image_grid_thw"])
    h = np.array(h.astype(mx.float32))
    cells, o = [], 0
    for t, hh, ww in np.array(inp["image_grid_thw"]).tolist():
        n = t * (hh // 2) * (ww // 2)
        cells += [pool_grid(h[o:o + n], hh // 2, ww // 2), h[o:o + n].mean(0, keepdims=True)]
        o += n
    return np.concatenate(cells)


@app.post("/vfeat")
def vfeat(r: VReq):
    t = time.time()
    with _gpu:
        g = vision_grid(decode(r.images))
    return {"grid": b64(g), "grid_shape": list(g.shape), "sec": round(time.time() - t, 3)}


@app.post("/act")
def act(r: ActReq):
    t = time.time()
    h = head(r.head)
    with _gpu:
        f, g = features(decode(r.images), r.instruction, upto=int(h["upto"]))
    names = [n for n in NAMES if int(n.split("_")[0][1:]) <= int(h["upto"])]
    a = head_forward(h, {n: f[i] for i, n in enumerate(names)}, g, r.state)
    return {"actions": a.tolist(), "sec": round(time.time() - t, 3)}


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    load()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
