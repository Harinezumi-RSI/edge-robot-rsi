"""VLA(画像と指示文から動作を出すモデル)の薄い包み。公開重みの SmolVLA(LIBERO で学習済み)を使う。

前処理・後処理は LeRobot の評価(lerobot/scripts/lerobot_eval.py の rollout)と同じ順番で通す:
  preprocess_observation → task を付ける → 環境の前処理(LIBERO の画像回転・状態 8 次元)
  → 方策の前処理(正規化・指示文のトークン化) → select_action → 方策の後処理 → 環境の後処理
"""
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION

DEFAULT_POLICY = "HuggingFaceVLA/smolvla_libero"


def stack_obs(obs_list):
    """観測の辞書のリスト → 先頭に batch 次元を持つ 1 つの辞書(台所を何個も同時に動かすとき)。"""
    o0 = obs_list[0]
    if isinstance(o0, dict):
        return {k: stack_obs([o[k] for o in obs_list]) for k in o0}
    if isinstance(o0, np.ndarray):
        return np.stack(obs_list)
    return o0


def _batch(x):
    """単体の環境の観測に batch 次元を足す(ベクトル化した環境と同じ形にする)。"""
    if isinstance(x, dict):
        return {k: _batch(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x[None]
    return x


class VLA:
    def __init__(self, path=DEFAULT_POLICY, device="mps", n_action_steps=None):
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path
        cfg.device = device
        if n_action_steps:
            cfg.n_action_steps = n_action_steps
        env_cfg = LiberoEnvConfig(task="libero_goal")
        self.policy = make_policy(cfg=cfg, env_cfg=env_cfg)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": device},
                                    "rename_observations_processor": {"rename_map": {}}})
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
        self.n_action_steps = cfg.n_action_steps

    def set_dtype(self, dtype):
        """改造: 演算ごとに半精度へ自動で落とす(autocast)。重みを丸ごと .to(半精度) にすると
        MPS の行列積で精度の食い違いの assert で落ちる(VLM は bf16、行動の専門家は fp32 で読み込まれるため)。"""
        self.autocast_dtype = dtype

    def reset(self):
        """指示を切り替えるときに呼ぶ(行動のキューを空にする)。"""
        self.policy.reset()

    @torch.no_grad()
    def act_batch(self, obs_list, instructions, active=None):
        """台所 N 個分の観測をまとめて 1 回で推論する → 行動 [N, 7]。"""
        o = preprocess_observation(stack_obs(obs_list))
        o["task"] = list(instructions)
        o = self.pre(self.env_pre(o))
        dt = getattr(self, "autocast_dtype", None)
        with torch.inference_mode(), torch.autocast(device_type="mps", dtype=dt, enabled=dt is not None):
            a = self.policy.select_action(o)
        a = self.env_post({ACTION: self.post(a.float())})[ACTION]
        return a.to("cpu").numpy()

    @torch.no_grad()
    def act(self, obs, instruction):
        o = preprocess_observation(_batch(obs))
        o["task"] = [instruction]
        o = self.pre(self.env_pre(o))
        dt = getattr(self, "autocast_dtype", None)
        with torch.inference_mode(), torch.autocast(device_type="mps", dtype=dt, enabled=dt is not None):
            a = self.policy.select_action(o)
        a = self.env_post({ACTION: self.post(a.float())})[ACTION]
        return a.to("cpu").numpy()[0]

    @torch.no_grad()
    def chunk_batch(self, obs_list, instructions, n=8, seed=None):
        """先生として使う: 台所 N 個分の観測から、新しく作った行動の塊の先頭 n step → [N, n, 7]。
        行動のキューは使わない(どの場面でも、その場面から動き出すときの行動を返す)。"""
        o = preprocess_observation(stack_obs(obs_list))
        o["task"] = list(instructions)
        o = self.pre(self.env_pre(o))
        if seed is not None:
            torch.manual_seed(seed)
        dt = getattr(self, "autocast_dtype", None)
        with torch.inference_mode(), torch.autocast(device_type="mps", dtype=dt, enabled=dt is not None):
            c = self.policy.predict_action_chunk(o)[:, :n]
        B = c.shape[0]
        a = self.env_post({ACTION: self.post(c.float().reshape(B * n, -1))})[ACTION]
        return a.to("cpu").numpy().reshape(B, n, -1)
