"""LIBERO のほかの 3 種(Spatial・Object・Long)に、家事ロボットの手の作り方をそのまま当てる。.venv で実行。

  python kitchen/suite_loop.py teach  --suite libero_object --inits 10-39 --out data/suite/libero_object/r0.npz
  python kitchen/lite_train.py --data data/suite/libero_object/r0.npz --out models/suite/libero_object/l0.npz
  python kitchen/suite_loop.py eval   --suite libero_object --inits 0-9 --head models/suite/libero_object/l0.npz --out outputs/suite/libero_object/l0/val.jsonl
  python kitchen/suite_loop.py eval   --suite libero_object --inits 0-9 --teacher smolvla --out outputs/suite/libero_object/smolvla/val.jsonl
  python kitchen/suite_loop.py dagger --suite libero_object --inits 10-39 --head models/suite/libero_object/l0.npz --out data/suite/libero_object/d1.npz

Goal(kitchen/chain_env.py)との違い:
- 課題ごとに場面が違うので、課題ごとに環境を作る。備え付けの置き場所は環境を作るときの乱数で決まる → 作る前に課題ごとの種を決めてそろえる
  (同じ種なら置き場所が一致することを Goal・Spatial・Object・Long で確かめた。Object と Long は乱数で置く備え付けが無い)。
- 試行は「波」で回す(同じ課題の初期状態を台所の数ずつ)。手の画像は「手が見る step だけ撮り直す」撮り方(Goal の 20 周目と同じ)。
- 学習データは描き直さず、試行の途中でその場で特徴にする(先生の練習は 4 step おき、成功した試行だけ写す)。
- DAgger の答えは、手が動いている途中でその場で先生に聞く(8 step おき、全部の試行)。改造 Jev の頭は Goal の目標用なので間引かない。
- 打ち切りの step は OpenVLA の LIBERO 評価と同じ(Spatial 220・Object 280・Goal 300・Long 520)。
出力の形は Goal と同じ(評価の行 = 課題・初期状態・成否・step、学習データ = lite_data.py と同じ F・G・S・A・K・E)。
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from lerobot.envs.libero import LiberoEnv, get_task_init_states  # noqa: E402

from kitchen.bench import mac_lock, parse_range  # noqa: E402
from kitchen.lite_common import CHUNK, head_forward_batch  # noqa: E402
from kitchen.lite_eval import load_head  # noqa: E402
from kitchen.lite_feats import extract_batch, obs_state  # noqa: E402
from kitchen.runlog import Timer  # noqa: E402

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


class SuiteEnv:
    """1 つの課題の台所(LeRobot の LiberoEnv)。成功しても自動で reset しない・カメラは observe() のときだけ撮る。"""

    def __init__(self, suite_name, task):
        self.suite_name, self.task = suite_name, task
        suite = benchmark.get_benchmark_dict()[suite_name]()
        seed = 7919 * (SUITES.index(suite_name) + 1) + task  # 備え付けの置き場所を課題ごとに固定する種
        random.seed(seed)
        np.random.seed(seed)
        self.e = LiberoEnv(task_suite=suite, task_id=task, task_suite_name=suite_name, obs_type="pixels_agent_pos", episode_index=0)
        self.e._env.env.horizon = 10**9  # 打ち切りは呼ぶ側の step 上限で決める
        self.instruction = suite.get_task(task).language
        self.inits = get_task_init_states(suite, task)
        rs = self.e._env.env
        for k in [k for k in rs._observables if k.endswith("_image")]:
            rs.modify_observable(observable_name=k, attribute="sampling_rate", modifier=1e-6)

    def start(self, init):
        self.e._init_states = self.inits
        self.e.init_state_id = init
        self.e.reset()
        return self.observe()

    def observe(self):
        return self.e._format_raw_obs(self.e._env.env._get_observations(force_update=True))

    def step(self, action):
        self.e._env.step(np.clip(np.asarray(action, np.float32), -1, 1))
        return bool(self.e._env.check_success())

    def close(self):
        self.e.close()


class Kitchens:
    """台所 N 個。課題が変わったら作り直す(波は課題ごとにそろえるので、作り直しは課題の変わり目だけ)。"""

    def __init__(self, suite_name, n):
        self.suite_name, self.envs = suite_name, [None] * n

    def get(self, k, task):
        if self.envs[k] is None or self.envs[k].task != task:
            if self.envs[k] is not None:
                self.envs[k].close()
            self.envs[k] = SuiteEnv(self.suite_name, task)
        return self.envs[k]


def waves(tasks, inits, n):
    for t in tasks:
        for j in range(0, len(inits), n):
            yield t, inits[j:j + n]


class LiteHand:
    def __init__(self, path, n_exec=4):
        self.h = load_head(path)
        self.upto = int(self.h["upto"]) or None
        self.n_exec = n_exec

    def chunks(self, obs_list, tasks):
        """→ 行動の塊 [N, 8, 7] と、その観測の特徴(F, 名前, G)。特徴は DAgger の記録にも使う。"""
        F, names, G = extract_batch(self.h["backend"], obs_list, tasks, self.upto)
        st = np.stack([obs_state(o) for o in obs_list])
        A = head_forward_batch(self.h, {nm: F[:, j] for j, nm in enumerate(names)}, G, st)
        return A, (F, names, G)


def teacher(name, n_action_steps=5):
    from kitchen.vla import VLA  # SmolVLA(LIBERO の 4 種で学習済みの公開重み)。5 step ずつ(Goal で 74%、毎 step は 86% だが 5 倍遅い)
    assert name == "smolvla", "ほかの 3 種の先生は今は SmolVLA だけ(公開の OpenVLA-OFT は Goal 用の重みしか手元に無い)"
    return VLA(n_action_steps=n_action_steps)


def run(a):
    tasks = list(range(10)) if a.tasks == "all" else [int(x) for x in a.tasks.split(",")]
    inits = parse_range(a.inits)
    max_steps = MAX_STEPS[a.suite]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if a.mode == "eval" and os.path.exists(a.out):
        done = {(r["task"], r["init"]) for r in map(json.loads, open(a.out))}
    K = Kitchens(a.suite, a.n_envs)
    hand = LiteHand(a.head) if a.head else None
    vla = teacher(a.teacher, a.teacher_steps) if (a.mode in ("teach", "dagger") or not a.head) else None
    if a.mode in ("teach", "dagger"):
        assert vla is not None
    rows, data = [], {k: [] for k in "FGSAKE"}
    names = None
    n_ok = n_all = n_ep = 0
    t0 = time.time()
    who = os.path.basename(a.head) if a.head else f"{a.teacher}_n{a.teacher_steps}"
    with Timer("practice" if a.mode != "eval" else "eval", f"{a.suite} {a.mode} {who}", suite=a.suite, inits=a.inits, tasks=a.tasks,
               who=who, mode=a.mode, n_envs=a.n_envs) as tm:
        for task, ii in waves(tasks, inits, a.n_envs):
            ii = [i for i in ii if (task, i) not in done]
            if not ii:
                continue
            with mac_lock():
                envs = [K.get(k, task) for k in range(len(ii))]
                obs = [e.start(i) for e, i in zip(envs, ii)]
                instr = [envs[0].instruction] * len(ii)
                if vla is not None:
                    vla.reset()
                n = len(ii)
                ok, steps, live = [False] * n, [0] * n, [True] * n
                queue = [[] for _ in range(n)]
                rec = [{"samples": [], "actions": []} for _ in range(n)]
                for t in range(max_steps):
                    act = [None] * n
                    if hand is not None:  # 軽い手: 塊を出して 4 step 実行
                        need = [k for k in range(n) if live[k] and not queue[k]]
                        if need:
                            ob = [envs[k].observe() for k in need]
                            A, (F, nm, G) = hand.chunks(ob, [task] * len(need))
                            names = nm
                            label = a.mode == "dagger" and t % a.every == 0
                            if label:  # その場の観測に先生の答え(その場面からの 8 step)
                                L = vla.chunk_batch(ob, [instr[0]] * len(need), n=CHUNK, seed=t)
                            for j, k in enumerate(need):
                                queue[k] = list(A[j, :hand.n_exec])
                                if label:
                                    rec[k]["samples"].append((F[j], G[j], obs_state(ob[j]), L[j].astype(np.float32)))
                        for k in range(n):
                            if live[k]:
                                act[k] = queue[k].pop(0)
                    else:  # 先生(SmolVLA、teacher_steps ずつ実行。観測は毎 step 渡し、塊が尽きたときだけ使われる)
                        ob = [envs[k].observe() if live[k] else obs[k] for k in range(n)]
                        obs = ob
                        A = vla.act_batch(ob, instr, active=live)
                        if a.mode == "teach" and t % a.every == 0:
                            idx = [k for k in range(n) if live[k]]
                            F, nm, G = extract_batch("dino", [ob[k] for k in idx], [task] * len(idx))
                            names = nm
                            for j, k in enumerate(idx):
                                rec[k]["samples"].append((t, F[j], G[j], obs_state(ob[k])))
                        for k in range(n):
                            if live[k]:
                                act[k] = A[k]
                    for k in range(n):
                        if not live[k]:
                            continue
                        if a.mode == "teach":
                            rec[k]["actions"].append(np.asarray(act[k], np.float32))
                        steps[k] = t + 1
                        if envs[k].step(act[k]):
                            ok[k], live[k] = True, False
                    if not any(live):
                        break
            for k, i in enumerate(ii):
                n_all += 1
                n_ok += ok[k]
                rows.append({"suite": a.suite, "task": task, "init": i, "ok": ok[k], "steps": steps[k], "who": who})
                if a.mode == "eval":
                    with open(a.out, "a") as f:
                        f.write(json.dumps(rows[-1]) + "\n")
                elif a.mode == "teach" and ok[k]:  # 先生の成功だけ写す
                    acts = np.stack(rec[k]["actions"])
                    for (t, F, G, S) in rec[k]["samples"]:
                        ch = acts[t:t + CHUNK]
                        if len(ch) < CHUNK:
                            ch = np.concatenate([ch, np.repeat(ch[-1:], CHUNK - len(ch), 0)])
                        for key, v in zip("FGSAKE", (F, G, S, ch, task, n_ep)):
                            data[key].append(v)
                    n_ep += 1
                elif a.mode == "dagger":
                    for (F, G, S, L) in rec[k]["samples"]:
                        for key, v in zip("FGSAKE", (F, G, S, L, task, n_ep)):
                            data[key].append(v)
                    n_ep += 1
            print(f"課題 {task} 初期状態 {ii[0]}〜{ii[-1]}: 成功 {sum(ok)}/{n}(合計 {n_ok}/{n_all}・サンプル {len(data['F'])}・"
                  f"{(time.time() - t0) / 60:.1f} 分)", flush=True)
        tm.info.update(ok=n_ok, episodes=n_all, samples=len(data["F"]))
    if a.mode != "eval":
        np.savez(a.out, F=np.stack(data["F"]), G=np.stack(data["G"]), S=np.stack(data["S"]), A=np.stack(data["A"]),
                 K=np.array(data["K"]), E=np.array(data["E"]), F_names=np.array(names), backend="dino",
                 suite=a.suite, mode=a.mode, who=who)
    print(f"合計 {n_ok}/{n_all}" + (f"・サンプル {len(data['F'])} → {a.out}" if a.mode != "eval" else f" → {a.out}"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["teach", "eval", "dagger"])
    ap.add_argument("--suite", required=True, choices=SUITES)
    ap.add_argument("--tasks", default="all", help="all か「0,3,5」")
    ap.add_argument("--inits", default="0-9")
    ap.add_argument("--head", default="", help="軽い手の頭(無ければ先生で動かす)")
    ap.add_argument("--teacher", default="smolvla")
    ap.add_argument("--teacher-steps", type=int, default=5, help="先生の 1 回の推論で実行する step(1 = 毎 step 推論、論文の作法だが 5 倍遅い)")
    ap.add_argument("--every", type=int, default=0, help="記録の間隔(既定: 先生の練習 4・DAgger 8)")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.every = a.every or (8 if a.mode == "dagger" else 4)
    run(a)


if __name__ == "__main__":
    main()
