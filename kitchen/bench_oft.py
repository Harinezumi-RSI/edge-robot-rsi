"""OpenVLA-OFT(7B)の 1 回の推論の時間を、家庭用 Mac mini(M4 Pro の GPU)で測る。重みは乱数、作りと大きさは公開の OFT と同じ。

  <OFT の環境の python> kitchen/bench_oft.py <公開の OFT の設定・コードのあるディレクトリ>

- 速さは重みの値に依らない: OFT は 8 step 分の行動を 1 回の順伝播でまとめて出す(並列デコード)ので、計算の量は入力の形だけで決まる。
- 公開の重み(15GB)は手元に無く、ダウンロードせずに測るため、公開の設定(config.json)とコード(modeling_prismatic.py)から
  同じ作り(SigLIP + DINOv2 の画像の土台・LLaMA-2 7B・行動の頭・手先の状態の射影)を組み、重みは乱数で作る(bf16)。
- 入力は軽い手と同じ: 正面と手首の画像 2 枚・指示文・手先の状態 8 次元 → 行動 8 step × 7。
- 測るときは Mac のほかの重い処理を止める(OFT は 15GB 前後を使う)。
"""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image


def main():
    d = sys.argv[1]
    from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
    from prismatic.models.action_heads import L1RegressionActionHead
    from prismatic.models.projectors import ProprioProjector
    dev = "mps"
    cfg = AutoConfig.from_pretrained(d, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(d, trust_remote_code=True)
    t = time.time()
    torch.manual_seed(0)
    with torch.device(dev):
        vla = AutoModelForVision2Seq.from_config(cfg, torch_dtype=torch.bfloat16, trust_remote_code=True)
    vla = vla.to(dev).eval()
    vla.vision_backbone.num_images_in_input = 2
    llm_dim = vla.llm_dim if hasattr(vla, "llm_dim") else vla.config.text_config.hidden_size
    with torch.device(dev):
        ah = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=7).to(torch.bfloat16).eval()
        pp = ProprioProjector(llm_dim=llm_dim, proprio_dim=8).to(torch.bfloat16).eval()
    n = sum(p.numel() for m in (vla, ah, pp) for p in m.parameters())
    print(f"組み立て {time.time() - t:.0f} 秒・重みの数 {n / 1e9:.2f}B(乱数、bf16)", flush=True)
    rng = np.random.default_rng(0)
    front, wrist = (Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)) for _ in range(2))
    prompt = "In: What action should the robot take to put the bowl on the stove?\nOut:"
    keys = list(getattr(vla, "norm_stats", {}) or {})
    unnorm = keys[0] if keys else None

    def once():
        inputs = proc(prompt, front).to(dev, dtype=torch.bfloat16)
        wp = proc(prompt, wrist)["pixel_values"].to(dev, dtype=torch.bfloat16)
        inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wp], dim=1)
        with torch.inference_mode():
            actions, _ = vla.predict_action(**inputs, unnorm_key=unnorm, do_sample=False, proprio=np.zeros(8, np.float32),
                                            proprio_projector=pp, action_head=ah, use_film=False)
        torch.mps.synchronize()
        return np.asarray(actions)

    for _ in range(3):
        a = once()
    ts = []
    for _ in range(15):
        t = time.perf_counter()
        a = once()
        ts.append(time.perf_counter() - t)
    med = float(np.median(ts))
    row = {"model": "OpenVLA-OFT 作り(重みは乱数)", "device": "Mac mini M4 Pro GPU(MPS)", "params_b": round(n / 1e9, 2),
           "latency_ms_median": round(med * 1000, 1), "latency_ms_min": round(min(ts) * 1000, 1), "actions_shape": list(a.shape),
           "actions_per_sec": round(a.shape[0] / med, 1), "runs": len(ts)}
    print(json.dumps(row, ensure_ascii=False))
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "bench_oft_mac.json")
    json.dump(row, open(out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
