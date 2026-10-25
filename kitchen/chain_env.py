"""一つの台所で動作を続けて実行するための環境。

LIBERO-Goal の 10 タスクは場面の定義が完全に同じで、違うのは目標(goal)だけ
(bddl を :language・:goal・:obj_of_interest を除いて比べると 10 本とも一致)。
そこで一つのシミュレーションのまま「いま判定する目標」と「VLA に渡す指示文」だけを差し替える。

観測の形は LeRobot の LiberoEnv(lerobot/envs/libero.py)と同じものを使う
(方策の前処理 LiberoProcessorStep が 180° 回転と状態 8 次元への変換をする)。
LiberoEnv との違い: 成功しても自動で reset しない/目標を途中で差し替えられる/動画用に高解像度の画像も撮れる。
"""
import os

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.bddl_utils import robosuite_parse_problem
from lerobot.envs.libero import LiberoEnv

from kitchen.skills import INSTRUCTION

SUITE = "libero_goal"


# LIBERO-Goal の場面の定義(10 動作で同じ)の備え付けの範囲(机の座標で x0, y0, x1, y1)
FIXTURE_REGIONS = {
    "wooden_cabinet_1_main": (0.02, -0.25, 0.04, -0.23),
    "flat_stove_1_main": (-0.42, 0.20, -0.40, 0.22),
    "wine_rack_1_main": (-0.27, -0.27, -0.25, -0.25),
}


FRESH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "fresh_inits.npz")
FRESH_FROM = 100


def init_table(suite, skill):
    """動作の初期状態の表(番号 → 状態)。0〜49 は LIBERO の公式、100 番以降は fresh_inits.py で作った新しいもの、50〜99 は NaN。
    lerobot は番号を表の長さで割った余りで使うので、公式の表のままだと 50 番以降が黙って 0〜49 の重複になる。"""
    from lerobot.envs.libero import get_task_init_states
    base = np.asarray(get_task_init_states(suite, skill), dtype=np.float64)
    if not os.path.exists(FRESH):
        return base
    fr = np.load(FRESH)[str(skill)]
    tab = np.full((FRESH_FROM + len(fr), base.shape[1]), np.nan)
    tab[:len(base)] = base
    tab[FRESH_FROM:] = fr
    return tab


def check_init(tab, init):
    assert 0 <= init < len(tab) and not np.isnan(tab[init]).any(), f"初期状態 {init} は表にない(0〜49 か、fresh_inits.py で作った 100 番以降)"


def fixture_seed(skill, init):
    """試行(動作・初期状態)ごとの備え付けの種。どの手で測っても同じ置き場所になる。"""
    return 7919 * int(skill) + int(init) + 1_000_000


class KitchenChainEnv(LiberoEnv):
    def __init__(self, start_task=1, init_index=0, **kw):
        suite = benchmark.get_benchmark_dict()[SUITE]()
        super().__init__(task_suite=suite, task_id=start_task, task_suite_name=SUITE,
                         obs_type="pixels_agent_pos", episode_index=init_index, **kw)
        self._suite = suite
        self._goals = {}
        for i in range(suite.n_tasks):
            t = suite.get_task(i)
            path = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
            self._goals[i] = robosuite_parse_problem(path)["goal_state"]
        assert set(self._goals) == set(INSTRUCTION), "動作の番号と LIBERO-Goal のタスク番号がずれている"
        self.goal_id = start_task
        # 続けて動くと 1 つのエピソードが長くなる(家事の通しは頼み 3 回 × 動作 × 最大 300 step + ホームへ戻す 80 step)。
        # robosuite の上限 1,000 step を超えると「終わったエピソード」で止まる(10/24 の通しで発生)ので外す。成否と打ち切りは呼ぶ側の step 上限で決める
        self._env.env.horizon = 10**9

    # ---- 目標の差し替え ----
    def fixture_pos(self):
        """備え付け 3 つ(棚・コンロ・ワインラック)の置き場所 [3, 3]。シミュレータの状態(位置・速度)には含まれない。"""
        m = self._env.env.sim.model
        return np.array([m.body_pos[m.body_name2id(n)] for n in FIXTURE_REGIONS], dtype=np.float64)

    def set_fixture_pos(self, P):
        sim = self._env.env.sim
        for n, p in zip(FIXTURE_REGIONS, np.asarray(P, dtype=np.float64)):
            sim.model.body_pos[sim.model.body_name2id(n)] = p
        sim.forward()

    def place_fixtures(self, seed):
        """備え付けを場面の定義の 2cm 四方から一様に置き直す(公式と同じ分布)。種で決まるので、試行ごとに再現できる。
        LIBERO は環境を作るたびに別の乱数で置き(種を決めても揃わない)、リセットでは動かさない → 台所 8 個の実行器では
        どの台所で走ったかで棚・コンロが 1〜2cm 動き、同じ試行の成否が入れ替わっていた(24 周目の目視)。"""
        rng = np.random.default_rng(seed)
        P = self.fixture_pos()
        for k, (x0, y0, x1, y1) in enumerate(FIXTURE_REGIONS.values()):
            P[k, 0], P[k, 1] = rng.uniform(x0, x1), rng.uniform(y0, y1)
        self.set_fixture_pos(P)
        return P

    def set_lazy_camera(self, on=True):
        """カメラを毎 step 撮らない(撮るのは observe() のときだけ)。1 step 20ms → 8ms(4 step に 1 回撮るとき)。
        物理は同じ。普段の画像は制御の途中で撮られ、記録した状態の描き直し(学習データの作り方)と約 3% の画素がずれるが、
        observe() の画像は描き直しとほぼ同じ(0.06%)。"""
        rs = self._env.env
        for k in [k for k in rs._observables if k.endswith("_image")]:
            rs.modify_observable(observable_name=k, attribute="sampling_rate", modifier=1e-6 if on else 1.0 / rs.control_timestep)

    def observe(self):
        """いまの状態でカメラを撮り直した観測(set_lazy_camera と組で使う)。"""
        return self._format_raw_obs(self._env.env._get_observations(force_update=True))

    def set_goal(self, skill_id):
        """以後の成功判定と VLA への指示文を skill_id の動作に切り替える(場面はそのまま)。"""
        assert self._suite.get_task(skill_id).language == INSTRUCTION[skill_id]
        self.goal_id = skill_id
        self.task_description = INSTRUCTION[skill_id]
        self._env.env.parsed_problem["goal_state"] = self._goals[skill_id]

    def goal_reached(self, skill_id=None):
        """skill_id の目標がいまの状態で満たされているか(既定はいまの目標)。"""
        if skill_id is None:
            return bool(self._env.check_success())
        inner = self._env.env
        saved = inner.parsed_problem["goal_state"]
        inner.parsed_problem["goal_state"] = self._goals[skill_id]
        try:
            return bool(self._env.check_success())
        finally:
            inner.parsed_problem["goal_state"] = saved

    # ---- 1 step(自動 reset しない) ----
    def step(self, action):
        raw_obs, reward, done, info = self._env.step(action)
        ok = bool(self._env.check_success())
        info = dict(info, is_success=ok, goal_id=self.goal_id)
        return self._format_raw_obs(raw_obs), float(ok), False, False, info

    # ---- ミニ RIS: アームの初期姿勢をずらす ----
    def perturb_robot(self, sigma, rng, settle=10, tries=20):
        """アームの 7 関節に N(0, sigma) [rad] のずれを足す(関節の可動域で切る)。机や物に触れる姿勢は引き直す。
        その後 settle step だけ動かさず(行動 0)待って落ち着かせ、観測を返す。sigma=0 なら何もしない。"""
        env = self._env.env
        robot = env.robots[0]
        idx = robot._ref_joint_pos_indexes
        sim = env.sim
        q0 = sim.data.qpos[idx].copy()
        lo, hi = sim.model.jnt_range[robot._ref_joint_indexes].T
        if sigma > 0:
            for _ in range(tries):
                q = np.clip(q0 + rng.normal(0.0, sigma, len(idx)), lo + 1e-3, hi - 1e-3)
                sim.data.qpos[idx] = q
                sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
                sim.forward()
                if not self._robot_touches_scene():
                    break
            else:
                sim.data.qpos[idx] = q0
                sim.forward()
        robot.controller.update(force=True)
        robot.controller.reset_goal()
        raw = None
        for _ in range(settle):
            raw, _, _, _ = self._env.step(np.zeros(7, dtype=np.float32) + np.array([0, 0, 0, 0, 0, 0, -1.0]))
        return self._format_raw_obs(raw if raw is not None else env._get_observations(force_update=True))

    def _robot_touches_scene(self):
        """アームの部品が机・家具・物に触れているか(ずらした直後の姿勢の点検)。"""
        env = self._env.env
        sim = env.sim
        robot_geoms = set(env.robots[0].robot_model.contact_geoms) | set(env.robots[0].gripper.contact_geoms)
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            g1, g2 = sim.model.geom_id2name(c.geom1), sim.model.geom_id2name(c.geom2)
            if (g1 in robot_geoms) != (g2 in robot_geoms):
                return True
        return False

    # ---- 動画用の高解像度画像(方策には使わない) ----
    def hires(self, camera="agentview", width=960, height=720):
        sim = self._env.env.sim
        img = sim.render(camera_name=camera, width=width, height=height)
        return np.ascontiguousarray(img[::-1])  # MuJoCo の画像は上下が逆
