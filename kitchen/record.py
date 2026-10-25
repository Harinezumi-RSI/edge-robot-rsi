"""成功した試行を「シミュレータの状態と行動だけ」で記録する(低ストレージの学習データ。.venv で実行)。

画像は保存しない。学習のときに状態を戻してその場で描き直す(1 step あたり約 1KB、画像で持つと約 400KB)。
  python kitchen/record.py --skills 1 3 --inits 10-49 --tries 2 --out data/rollouts

1 試行 = 1 ファイル(npz): states[T, n]・actions[T, 7]・skill・init・try。成功した試行だけ残す(自分の成功から学ぶ)。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from kitchen.bench import mac_lock, parse_range  # noqa: E402
from kitchen.chain_env import KitchenChainEnv  # noqa: E402
from kitchen.skills import INSTRUCTION  # noqa: E402
from kitchen.vla import VLA  # noqa: E402

MAX_STEPS = 300


def sim_state(env):
    return env._env.env.sim.get_state().flatten().copy()


def restore_obs(env, state, fixtures=None):
    """記録した状態に戻して、そのときの観測(画像 2 枚 + 手先の状態)を描き直す。
    fixtures(備え付けの置き場所)があれば先に戻す。無いと、描き直す環境の置き場所で描かれ、元の試行と 1〜2cm ずれる。"""
    sim = env._env.env.sim
    if fixtures is not None:
        env.set_fixture_pos(fixtures)
    sim.set_state_from_flattened(state)
    sim.forward()
    return env._format_raw_obs(env._env.env._get_observations(force_update=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", nargs="+", type=int, required=True)
    ap.add_argument("--inits", default="10-49")
    ap.add_argument("--tries", type=int, default=1)
    ap.add_argument("--n-action-steps", type=int, default=0)
    ap.add_argument("--out", default="data/rollouts")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    vla = VLA(n_action_steps=a.n_action_steps or None)
    for skill in a.skills:
        env = None
        for init in parse_range(a.inits):
            for k in range(a.tries):
                name = os.path.join(a.out, f"s{skill}_i{init}_k{k}")
                if os.path.exists(name + ".npz") or os.path.exists(name + ".fail"):
                    continue
                if env is None:
                    env = KitchenChainEnv(start_task=skill, init_index=init)
                env.init_state_id = init
                with mac_lock():
                    torch.manual_seed(1000 * k + init)
                    obs, _ = env.reset()
                    env.set_goal(skill)
                    vla.reset()
                    states, actions, ok = [], [], False
                    for t in range(MAX_STEPS):
                        states.append(sim_state(env))
                        act = vla.act(obs, INSTRUCTION[skill])
                        actions.append(act)
                        obs, _, _, _, info = env.step(act)
                        if info["is_success"]:
                            ok = True
                            break
                if ok:
                    np.savez_compressed(name + ".npz", states=np.array(states, dtype=np.float32),
                                        actions=np.array(actions, dtype=np.float32), skill=skill, init=init, k=k)
                else:
                    open(name + ".fail", "w").write(str(len(actions)))
                print(f"skill {skill} init {init} try {k}: {'成功' if ok else '失敗'} {len(actions)} step", flush=True)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
