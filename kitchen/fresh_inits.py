"""どの段階でも使っていない初期状態を作る(LIBERO の置き方の乱数で、種を固定)。.venv で実行。

  python kitchen/fresh_inits.py --n 40 --out data/fresh_inits.npz
  python kitchen/fresh_inits.py --check            # 同じ種で同じ状態・別の種で別の状態になるか

LIBERO の初期状態は 1 動作 50 通り(0〜9 判断用・10〜39 練習・40〜49 最終テストで使い切り)。lerobot は番号を 50 で割った余りで
使うので、50 番以降を指定すると黙って 0〜49 の重複になる。ここでは環境の置き方(公式の初期状態を作ったのと同じ、bddl の範囲から
一様に置く乱数、引き出しやコンロのつまみの初期の開き具合も含む)で新しく作り、番号 100 番以降として使う(chain_env.init_table)。
保存するのは置いた直後の状態(公式の初期状態と同じく、落ち着かせる空の step の前)。
公式の初期状態は腕の 7 関節に約 0.02 rad の揺らぎが入っている(robosuite の initialization_noise、ガウス 0.02)。この環境は腕を揺らさずに
置くので、同じ大きさの揺らぎを足す(10/25、足す前は腕が毎回同じ姿勢だった)。物の置き場所の散らばりは公式と同じ(標準偏差の比 0.83〜0.97)。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from kitchen.chain_env import KitchenChainEnv  # noqa: E402
from kitchen.skills import INSTRUCTION  # noqa: E402


def fresh_seed(skill, k):
    """動作・番号ごとの種(備え付けの種 fixture_seed とは別の範囲)。"""
    return 500_000 + 1_000 * int(skill) + int(k)


ARM_NOISE = 0.02


def arm_index(env):
    """平らにした状態(先頭は時刻)の中の腕の 7 関節の位置。"""
    m = env._env.env.sim.model
    return [1 + int(m.jnt_qposadr[m.joint_name2id(f"robot0_joint{k}")]) for k in range(1, 8)]


def sample(env, seed):
    env._env.seed(seed)  # LIBERO は np.random.seed
    env._env.reset()     # bddl の範囲から置き直す(deterministic_reset でないとき)
    st = env._env.get_sim_state().copy()
    st[arm_index(env)] += np.random.default_rng(seed).normal(0.0, ARM_NOISE, 7)  # 公式と同じ腕の揺らぎ
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="1 動作あたりの数")
    ap.add_argument("--out", default="data/fresh_inits.npz")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    env = KitchenChainEnv(start_task=1, init_index=0)
    env.reset()
    if a.check:
        s1, s1b, s2 = sample(env, fresh_seed(0, 0)), sample(env, fresh_seed(0, 0)), sample(env, fresh_seed(0, 1))
        from lerobot.envs.libero import get_task_init_states
        f0 = get_task_init_states(env._suite, 0)
        print(f"状態の長さ {len(s1)}(公式 {f0.shape[1]})・同じ種の差 {np.abs(s1 - s1b).max():.2e}・別の種の差 {np.abs(s1 - s2).max():.3f}"
              f"・公式どうし(0 と 1)の差 {np.abs(f0[0] - f0[1]).max():.3f}")
        return
    out = {}
    for s in sorted(INSTRUCTION):
        out[str(s)] = np.stack([sample(env, fresh_seed(s, k)) for k in range(a.n)])
        print(f"動作 {s}: {a.n} 通り", flush=True)
    np.savez(a.out, **out, n=a.n, seeds="fresh_seed(skill, k) = 500000 + 1000*skill + k")
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
