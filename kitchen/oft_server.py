"""先生の候補: 公開の OpenVLA-OFT(LIBERO-Goal で学習済み、moojink/openvla-7b-oft-finetuned-libero-goal)を Mac mini の GPU で動かす小さなサーバ。
OFT の環境(openvla-oft の prismatic と transformers の改造版)の python で起動し、台所(.venv)からは HTTP で呼ぶ(kitchen/oft_teacher.py)。

  <OFT の環境の python> kitchen/oft_server.py --model models/oft_goal --code <prismatic と改造版 transformers の置き場> --port 8770

前処理と後処理は公開の OFT の LIBERO 評価(experiments/robot/libero)と同じ:
  画像: 生の向き(上下左右が逆)を 180° 回す → JPEG で往復 → 224 に縮小 → 中央 90% を切り出して 224 に戻す(正面と手首の 2 枚)
  手先の状態: 位置 3 + 回転(軸角)3 + 指 2 を q01/q99 で [-1, 1] に
  グリッパ: 出力 [0, 1] を二値化して符号を反転(robosuite の規約)。ここを省くと掴む指令が出ない
1 回の推論で 8 step の塊を返す。/chunk は複数の観測をまとめて受け取り、1 つずつ推論する。
メモリは約 16GB 使うので、ほかの重い処理と同時に動かさない。
"""
import argparse
import base64
import glob
import io
import json
import math
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from PIL import Image

IMG = 224


def quat2axisangle(quat):
    """robosuite の実装と同じ(OFT の LIBERO 評価が使っているもの)。"""
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = min(1.0, max(-1.0, q[3]))
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def jpeg_roundtrip(a):
    """学習データ(RLDS)は JPEG で保存されていたので、推論でも 1 回 JPEG を通して分布を合わせる(quality 95・4:2:0)。"""
    buf = io.BytesIO()
    Image.fromarray(a).convert("RGB").save(buf, format="JPEG", quality=95, subsampling=2)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def center_crop(a, scale=0.9):
    """面積 90% の中央を切り出して 224 に戻す(TF の crop_and_resize と同じ座標の取り方、バイリニア)。"""
    h, w = a.shape[:2]
    f = math.sqrt(scale)
    lo, hi = (1.0 - f) / 2.0, (1.0 - f) / 2.0 + f
    ys = lo * (h - 1) + np.arange(IMG) * (hi - lo) * (h - 1) / (IMG - 1)
    xs = lo * (w - 1) + np.arange(IMG) * (hi - lo) * (w - 1) / (IMG - 1)
    y0 = np.floor(ys).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.floor(xs).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    p = a.astype(np.float32)
    top = p[y0][:, x0] * (1 - wx) + p[y0][:, x1] * wx
    bot = p[y1][:, x0] * (1 - wx) + p[y1][:, x1] * wx
    return np.clip(np.round(top * (1 - wy) + bot * wy), 0, 255).astype(np.uint8)


def prepare(raw):
    a = np.ascontiguousarray(raw[::-1, ::-1])
    if a.shape[:2] != (IMG, IMG):
        a = jpeg_roundtrip(a)
        a = np.asarray(Image.fromarray(a).convert("RGB").resize((IMG, IMG), Image.LANCZOS), dtype=np.uint8)
    return Image.fromarray(center_crop(a)).convert("RGB")


class OFT:
    def __init__(self, model_dir):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from prismatic.models.action_heads import L1RegressionActionHead
        from prismatic.models.projectors import ProprioProjector
        from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NUM_ACTIONS_CHUNK, NormalizationType

        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        t = time.time()
        self.proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                          trust_remote_code=True).to(self.dev).eval()
        att = {type(l.self_attn).__name__ for l in self.vla.language_model.model.layers}
        # 改造版 transformers の双方向の注意は SDPA の経路にしか入っていない。ほかに解決されると黙って精度が落ちる
        assert att == {"LlamaSdpaAttention"}, f"注意の実装が SDPA でない: {att}(transformers の改造版が読まれていない)"
        self.vla.vision_backbone.num_images_in_input = 2  # 正面と手首の 2 枚
        dim = getattr(self.vla, "llm_dim", None) or self.vla.config.text_config.hidden_size

        def load(module, pattern):
            path = sorted(glob.glob(os.path.join(model_dir, pattern)))[-1]
            sd = torch.load(path, map_location="cpu", weights_only=False)
            module.load_state_dict({k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()})
            return module.to(torch.bfloat16).to(self.dev).eval()

        self.ah = load(L1RegressionActionHead(input_dim=dim, hidden_dim=dim, action_dim=7), "action_head--*_checkpoint.pt")
        self.pp = load(ProprioProjector(llm_dim=dim, proprio_dim=8), "proprio_projector--*_checkpoint.pt")
        # 公式の get_vla と同じく、追加学習で使ったデータの統計(dataset_statistics.json)を読み込む。
        # config に入っているのは事前学習のデータセットの統計だけで、LIBERO-Goal の統計はこちらにしかない
        st = os.path.join(model_dir, "dataset_statistics.json")
        if os.path.exists(st):
            self.vla.norm_stats = {**self.vla.norm_stats, **json.load(open(st))}
        keys = list(self.vla.norm_stats)
        self.unnorm = next((k for k in keys if "libero_goal" in k), keys[0])
        self.q99 = ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99
        self.chunk = NUM_ACTIONS_CHUNK
        print(f"読み込み {time.time() - t:.0f} 秒・{self.dev}・統計 {self.unnorm}・状態の正規化 {'q01/q99' if self.q99 else 'min/max'}"
              f"・塊 {self.chunk} step", flush=True)

    def infer(self, front, wrist, pos, quat, grip, instruction):
        torch = self.torch
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.proc(prompt, prepare(front)).to(self.dev, dtype=torch.bfloat16)
        w = self.proc(prompt, prepare(wrist)).to(self.dev, dtype=torch.bfloat16)
        inputs["pixel_values"] = torch.cat([inputs["pixel_values"], w["pixel_values"]], dim=1)
        s = np.concatenate([np.asarray(pos, np.float64).reshape(3), quat2axisangle(quat), np.asarray(grip, np.float64).reshape(2)])
        ns = self.vla.norm_stats[self.unnorm]["proprio"]
        lo, hi = (np.array(ns["q01"]), np.array(ns["q99"])) if self.q99 else (np.array(ns["min"]), np.array(ns["max"]))
        s = np.clip(2 * (s - lo) / (hi - lo + 1e-8) - 1, -1, 1)
        with torch.inference_mode():
            actions, _ = self.vla.predict_action(**inputs, unnorm_key=self.unnorm, do_sample=False, proprio=s,
                                                 proprio_projector=self.pp, action_head=self.ah)
        a = np.asarray(actions, dtype=np.float32).reshape(-1, 7)
        a[:, -1] = -np.sign(2.0 * a[:, -1] - 1.0)  # [0, 1] → 二値化 → 符号反転(robosuite: -1 開く・+1 閉じる)
        return a


def decode(b, shape):
    return np.frombuffer(base64.b64decode(b), dtype=np.uint8).reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--code", required=True, help="prismatic と transformers の改造版がある置き場(sys.path の先頭に足す)")
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.code))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    oft = OFT(os.path.abspath(a.model))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, obj, code=200):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send({"ok": True, "unnorm": oft.unnorm, "chunk": oft.chunk})

        def do_POST(self):
            t = time.time()
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = []
            for it in req["items"]:
                out.append(oft.infer(decode(it["front"], it["shape"]), decode(it["wrist"], it["shape"]),
                                     it["pos"], it["quat"], it["grip"], it["instruction"]).tolist())
            self._send({"actions": out, "sec": round(time.time() - t, 3)})

    print(f"待ち受け 127.0.0.1:{a.port}", flush=True)
    HTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
