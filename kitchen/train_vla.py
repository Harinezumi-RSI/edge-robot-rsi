"""手(SmolVLA)を Mac で速く・低ストレージで追加学習する(.venv で実行)。

  python kitchen/train_vla.py --data data/rollouts --steps 1000 --batch 8 --k 4 --out models/vla_delta.safetensors

速さの工夫:
  1) 画像・言語の部分(VLM)は固定。学習するのは行動の専門家(と時刻・行動の射影)だけ(SmolVLA の既定と同じ方針)。
  2) 固定部分の計算(画像 2 枚と指示文と状態の読み込み = 推論時間の約 4 割)は 1 回だけ。その結果(KV)を使い回して、
     ノイズと時刻の組を k 通り専門家に通す。k 回別々に計算するのと数式上同じで、重い計算は 1/k。
低ストレージの工夫:
  3) 学習データは画像を持たない。記録したシミュレータの状態に戻して、その場で描き直す(record.py)。
  4) 保存するのは学習で変わった重みだけ(差分)。
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from kitchen.chain_env import KitchenChainEnv  # noqa: E402
from kitchen.record import restore_obs  # noqa: E402
from kitchen.runlog import Timer  # noqa: E402
from kitchen.skills import INSTRUCTION  # noqa: E402
from kitchen.vla import VLA, _batch  # noqa: E402


def stack_obs(obs_list):
    """観測の辞書のリスト → 先頭に batch 次元を持つ 1 つの辞書。"""
    o0 = obs_list[0]
    if isinstance(o0, dict):
        return {k: stack_obs([o[k] for o in obs_list]) for k in o0}
    if isinstance(o0, np.ndarray):
        return np.stack(obs_list)
    return o0


def load_data(d):
    """d = 置き場のディレクトリ(中の npz を全部)か、使う npz を 1 行ずつ並べた .txt(ミニ RSI で採点に通ったものだけ)。
    .txt の行は「パス」か「パス<TAB>重み」。重みは、その試行の step を引く確率の倍率(苦労して取った成功を重くする)。"""
    eps = []
    if d.endswith(".txt"):
        items = []
        for line in open(d):
            parts = line.strip().split("\t")
            if parts and parts[0]:
                items.append((parts[0], float(parts[1]) if len(parts) > 1 else 1.0))
    else:
        items = [(f, 1.0) for f in sorted(glob.glob(os.path.join(d, "*.npz")))]
    for f, w in items:
        z = np.load(f)
        eps.append({"states": z["states"], "actions": z["actions"], "skill": int(z["skill"]), "weight": w})
    return eps


def trainable(policy):
    """学習する重み: 行動の専門家と、時刻・行動の射影。VLM(画像・言語)と状態の射影は固定。"""
    names = []
    for n, p in policy.named_parameters():
        train = (".lm_expert." in n or "action_in_proj" in n or "action_out_proj" in n or "action_time_mlp" in n)
        p.requires_grad_(train)
        if train:
            names.append(n)
    return names


def make_batch(vla, env, eps, idx, chunk):
    obs, acts, pads, instr = [], [], [], []
    for e_i, t in idx:
        e = eps[e_i]
        obs.append(restore_obs(env, e["states"][t]))
        a = e["actions"][t:t + chunk]
        pad = np.zeros(chunk, dtype=bool)
        if len(a) < chunk:  # 試行の終わりを越える分は最後の行動で埋め、損失から外す
            pad[len(a):] = True
            a = np.concatenate([a, np.repeat(a[-1:], chunk - len(a), 0)])
        acts.append(a)
        pads.append(pad)
        instr.append(INSTRUCTION[e["skill"]])
    o = preprocess_observation(stack_obs(obs))
    o["task"] = instr
    o[ACTION] = torch.from_numpy(np.stack(acts)).float()
    o["action_is_pad"] = torch.from_numpy(np.stack(pads))
    return vla.pre(vla.env_pre(o))


def loss_prefix_reuse(policy, batch, k, autocast_dtype=None):
    """固定部分(prefix)を 1 回だけ計算し、ノイズと時刻を k 通り専門家に通した損失の平均。"""
    m = policy.model
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    lang_tokens, lang_masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
    adim = batch[ACTION].shape[-1]          # 本当の行動の次元(7)。prepare_action は 32 次元に 0 を詰める
    actions = policy.prepare_action(batch)
    with torch.no_grad(), torch.autocast("mps", dtype=autocast_dtype, enabled=autocast_dtype is not None):
        pe, ppad, patt = m.embed_prefix(images, img_masks, lang_tokens, lang_masks, state=state)
        att2d = make_att_2d_masks(ppad, patt)
        pos = torch.cumsum(ppad, dim=1) - 1
        _, kv = m.vlm_with_expert.forward(attention_mask=att2d, position_ids=pos, past_key_values=None,
                                          inputs_embeds=[pe, None], use_cache=True, fill_kv_cache=True)
    valid = (~batch["action_is_pad"]).float()[:, :, None]
    total = 0.0
    for _ in range(k):
        noise = m.sample_noise(actions.shape, actions.device)
        tt = m.sample_time(actions.shape[0], actions.device)
        x_t = tt[:, None, None] * noise + (1 - tt[:, None, None]) * actions
        with torch.autocast("mps", dtype=autocast_dtype, enabled=autocast_dtype is not None):
            v_t = m.denoise_step(ppad, kv, x_t, tt)
        # 試行の終わりを越えた step と詰め物の次元は損失から外す(LeRobot の forward は "actions_id_pad" の綴り違いで外れていない)
        err = F.mse_loss(noise - actions, v_t.float(), reduction="none")[:, :, :adim]
        total = total + (err * valid).sum() / (valid.sum() * adim).clamp(min=1)
    return total / k


def loss_plain(policy, batch):
    """比較用: LeRobot の普通の学習(毎回 prefix も計算し直す)。"""
    loss, _ = policy.forward(batch)
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rollouts")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--k", type=int, default=4, help="prefix 1 回あたりのノイズの組の数(1 で使い回しなし)")
    ap.add_argument("--plain", action="store_true", help="比較用: 使い回しなしの普通の学習")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--out", default="models/vla_delta.safetensors")
    ap.add_argument("--bench-only", type=int, default=0, help="N step だけ回して 1 step の時間を測る")
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    eps = load_data(a.data)
    idx_all = [(i, t) for i, e in enumerate(eps) for t in range(len(e["actions"]))]
    w_all = np.array([eps[i]["weight"] for i, _ in idx_all], dtype=np.float64)
    w_all /= w_all.sum()
    print(f"試行 {len(eps)} 本・サンプル {len(idx_all)}・状態の合計 {sum(e['states'].nbytes for e in eps) / 1e6:.1f} MB", flush=True)
    vla = VLA()
    policy = vla.policy
    policy.train()
    names = trainable(policy)
    params = [p for p in policy.parameters() if p.requires_grad]
    print(f"学習する重み {sum(p.numel() for p in params) / 1e6:.1f}M / 全体 {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-10)
    env = KitchenChainEnv(start_task=eps[0]["skill"], init_index=0)
    env.reset()
    chunk = policy.config.chunk_size
    dt = torch.bfloat16 if a.bf16 else None
    log, t0 = [], time.time()
    n_steps = a.bench_only or a.steps
    timer = Timer("train", f"手の追加学習 {os.path.basename(a.out)}" + ("(時間の測定)" if a.bench_only else ""),
                  data=a.data, episodes_data=len(eps), samples=len(idx_all), batch=a.batch, k=1 if a.plain else a.k,
                  plain=a.plain, bf16=a.bf16, lr=a.lr).__enter__()
    for step in range(1, n_steps + 1):
        idx = [idx_all[j] for j in rng.choice(len(idx_all), size=a.batch, p=w_all)]
        ts = time.time()
        batch = make_batch(vla, env, eps, idx, chunk)
        t_data = time.time() - ts
        loss = loss_plain(policy, batch) if a.plain else loss_prefix_reuse(policy, batch, a.k, dt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 10.0)
        opt.step()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t_step = time.time() - ts
        log.append({"step": step, "loss": float(loss), "sec": round(t_step, 3), "data_sec": round(t_data, 3)})
        if step % 20 == 0 or step == n_steps:
            rec = log[-20:]
            print(f"step {step}: loss {np.mean([r['loss'] for r in rec]):.4f}  {np.mean([r['sec'] for r in rec]):.2f} 秒/step"
                  f"(描き直し {np.mean([r['data_sec'] for r in rec]):.2f})  経過 {(time.time() - t0) / 60:.1f} 分", flush=True)
    timer.info.update(train_steps=n_steps, sec_per_step=float(np.median([r["sec"] for r in log[2:]] or [0])),
                      final_loss=float(np.mean([r["loss"] for r in log[-20:]])))
    timer.__exit__(None, None, None)
    if a.bench_only:
        print(json.dumps({"mode": "plain" if a.plain else f"reuse_k{a.k}", "batch": a.batch,
                          "sec_per_step": float(np.median([r["sec"] for r in log[2:]]))}, ensure_ascii=False))
        return
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sd = {n: p.detach().cpu().contiguous() for n, p in policy.named_parameters() if n in set(names)}
    save_file(sd, a.out)
    json.dump(log, open(a.out.replace(".safetensors", "_log.json"), "w"))
    print(f"差分を保存: {a.out}({os.path.getsize(a.out) / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
