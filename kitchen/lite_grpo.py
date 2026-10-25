"""先生なしのオンライン RL(GRPO)で軽い手を鍛える。.venv で実行(改造 Jev の採点は頭のサーバ brain_server.py の /jev に聞く)。

  .venv-mlx/bin/python kitchen/brain_server.py &
  python kitchen/lite_grpo.py --init models/lite/c1o.npz --tag g1 --skills 0 1 3 5 6 9 --chain --iters 200 --workers 3

LLM のオンライン RL と同じ回し方: いまの手で少し試し(場面 × 揺らぎつきの組)→ 自分の目(改造 Jev)で採点 →
組の中で平均より良かった動きを増やし悪かった動きを減らす(GRPO の優位、PPO の上限つきの比)→ 出発点の手から離れすぎない罰(KL)→ 少しだけ更新、を何百回も。
方策: 「頭の塊(平均)+ 塊ごとの揺らぎ N(0, σ)」のガウス方策(4 step 同じ向き・グリッパには入れない)。比は実行した 4 step × 6 次元の尤度から。
比べる尤度は練習の時と同じく dropout なしで出す(dropout ありだと動かす前から比が 1 からずれ、9 割がクリップされていた)。

報酬(自分の目と自分の動きの記録だけ。シミュレータの正解は記録するだけで使わない):
  r = p(狙いの目標, 動いた後)
      + w_fast · p · (1 - step / 打ち切り)                           … 早くできた
      - w_undo · Σ_{ほかの目標} max(0, p(動く前) - p(動いた後))       … できていたことを壊さない(改造 Jev は 10 目標を 1 回で読む)
      - w_jerk · ガクつき / 基準                                      … 滑らか(行動の 2 階差の 2 乗の平均)
速さ: 練習は --workers 本の処理に分ける(台所 8 個ずつ)。変わった場面の「前の動作」は場面ごとに 1 回だけ動かして状態を覚え、
あとは戻すだけ。練習の打ち切りは --train-steps(評価は 300 のまま)。
--eval-every 回ごとに、揺らぎなしで判断用の初期状態 0〜9 を続けて動く評価で測り、頭を保存する。
"""
import argparse
import json
import shutil
import multiprocessing as mp
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRAIN = os.environ.get("PAI_BRAIN", "http://127.0.0.1:8765")
PRE = {0: 9, 1: 2, 2: 7, 3: 9, 4: 7, 5: 2, 6: 5, 7: 1, 8: 9, 9: 0}  # 続けて動く評価の前の動作(lite_batch.py と同じ)
EXEC, EVAL_STEPS = 4, 300


def ask(path, payload):
    req = urllib.request.Request(BRAIN + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


# ---------------- 練習の処理(別プロセス) ----------------
_W = {}


def _init_worker(n_envs, snap_dir, keep_snaps=False):
    import torch
    torch.set_num_threads(2)
    from kitchen.chain_env import KitchenChainEnv, init_table
    from kitchen.skills import INSTRUCTION
    envs = [KitchenChainEnv(start_task=1, init_index=0) for _ in range(n_envs)]
    for e in envs:
        e.set_lazy_camera(True)
    _W.update(envs=envs, inits={s: init_table(envs[0]._suite, s) for s in INSTRUCTION}, pre={}, snap=snap_dir, keep_snaps=keep_snaps)


def _act(h, ob, skills):
    from kitchen.lite_common import head_forward_batch
    from kitchen.lite_feats import extract_batch, obs_state
    F, names, G = extract_batch("dino", ob, skills)
    S8 = np.stack([obs_state(o) for o in ob])
    mu = head_forward_batch(h, {nm: F[:, j] for j, nm in enumerate(names)}, G, S8)[:, :EXEC]
    return F, G, S8, mu


def _start(env, s, i, h, chain, tag, cache=True):
    """場面を用意する。変わった場面は「前の動作 → ホーム」を場面ごとに 1 回だけ動かして覚え、2 回目からは戻すだけ。
    → (観測, ホーム, 動く前の Jev の 10 目標の確率)"""
    from kitchen.chain_env import check_init, fixture_seed
    from kitchen.lite_batch import home_recorded
    check_init(_W["inits"][s], i)
    env._init_states = _W["inits"][s]
    env.init_state_id = i
    o, _ = env.reset()
    home = (o["robot_state"]["eef"]["pos"].copy(), o["robot_state"]["eef"]["quat"].copy())
    env.place_fixtures(fixture_seed(s, i))
    key = (s, i, chain)
    if not cache or key not in _W["pre"]:  # 評価は覚えた状態を使わず、その手で前の動作から動かす(ほかの評価と同じ条件)
        if chain:
            g = PRE[s]
            env.set_goal(g)
            q = []
            for t in range(EVAL_STEPS):
                if not q:
                    q = list(_act(h, [env.observe()], [g])[3][0])
                o, _, _, _, info = env.step(np.clip(q.pop(0), -1, 1))
                if info["is_success"]:
                    break
            o = home_recorded(env, o, home, None)  # lite_batch.py の続けて動く評価と同じ戻し方(go_home は着いてから 1 step 多い)
        if not cache:
            env.set_goal(s)
            return env.observe(), home, None
        p = os.path.join(_W["snap"], f"before_s{s}_i{i}_{int(chain)}_{os.getpid()}.jpg")
        import imageio.v2 as imageio
        imageio.imwrite(p, env.hires(width=640, height=480), quality=92)
        before = ask("/jev", {"image": os.path.abspath(p)})["probs"]
        if not _W.get("keep_snaps"):
            os.remove(p)
        _W["pre"][key] = (env._env.env.sim.get_state().flatten().copy(), before)
    else:
        state, before = _W["pre"][key]
        sim = env._env.env.sim
        sim.set_state_from_flattened(state)
        sim.forward()
        robot = env._env.env.robots[0]
        robot.controller.update(force=True)
        robot.controller.reset_goal()
    env.set_goal(s)
    return env.observe(), home, _W["pre"][key][1]


def _rollout(job):
    """job = (頭, 試行 [(動作, 初期状態, 組の番号, 種)], 続けて動くか, σ, 打ち切り, 記録するか) → 試行ごとの記録"""
    import imageio.v2 as imageio
    from kitchen.run_chain import go_home
    h, trials, chain, sigma, max_steps, record, tag = job
    envs = _W["envs"]
    out = []
    for w in range(0, len(trials), len(envs)):
        wave = trials[w:w + len(envs)]
        n = len(wave)
        homes, befores, rec = [], [], []
        sigs = [t[4] if len(t) > 4 else sigma for t in wave]  # 試行ごとの揺らぎ(成功の少ない場面は大きく)
        chs = [t[5] if len(t) > 5 else chain for t in wave]   # 試行ごとに普段/続けて動く
        for env, t, sg, ch in zip(envs, wave, sigs, chs):
            s, i, k = t[0], t[1], t[2]
            o, home, before = _start(env, s, i, h, ch, tag, cache=record)
            homes.append(home)
            befores.append(before)
            rec.append({"skill": s, "init": i, "k": k, "F": [], "G": [], "S": [], "mu": [], "a": [], "acts": [], "ok": False, "steps": 0,
                        "p_before": before, "sigma": float(sg), "chain": bool(ch)})
        rngs = [np.random.default_rng(t[3]) for t in wave]
        live, queue = [True] * n, [[] for _ in range(n)]
        for t in range(max_steps):
            need = [j for j in range(n) if live[j] and not queue[j]]
            if need:
                F, G, S8, mu = _act(h, [envs[j].observe() for j in need], [wave[j][0] for j in need])
                for b, j in enumerate(need):
                    a = mu[b].astype(np.float32).copy()
                    if sigs[j] > 0:
                        nz = rngs[j].normal(0.0, sigs[j], 7).astype(np.float32)
                        nz[6] = 0.0
                        a = a + nz
                    queue[j] = list(a)
                    if record:
                        r = rec[j]
                        r["F"].append(F[b]); r["G"].append(G[b]); r["S"].append(S8[b]); r["mu"].append(mu[b].astype(np.float32)); r["a"].append(a)
            for j in range(n):
                if not live[j]:
                    continue
                act = np.clip(queue[j].pop(0), -1, 1)
                rec[j]["acts"].append(act[:6])
                _, _, _, _, info = envs[j].step(act)
                rec[j]["steps"] = t + 1
                if info["is_success"]:
                    live[j], rec[j]["ok"] = False, True
            if not any(live):
                break
        for j, env in enumerate(envs[:n]):
            s = wave[j][0]
            o, _ = go_home(env, env.observe(), homes[j])
            rec[j]["sim_ok"] = bool(env.goal_reached(s))
            if record:
                p = os.path.join(_W["snap"], f"{tag}_{os.getpid()}_{w}_{j}.jpg")
                imageio.imwrite(p, env.hires(width=640, height=480), quality=92)
                rec[j]["p_after"] = ask("/jev", {"image": os.path.abspath(p)})["probs"]
                if not _W.get("keep_snaps"):  # 採点が済んだ画像は使わない(点数は trials.jsonl に残る)。g1〜g6 で 0.58GB たまった
                    os.remove(p)
            A = np.asarray(rec[j].pop("acts"), np.float32)
            rec[j]["jerk"] = float(np.mean(np.sum(np.diff(A, n=2, axis=0) ** 2, 1))) if len(A) > 2 else 0.0
        out += rec
    return out


# ---------------- 学ぶ側 ----------------
def main():
    import torch
    from kitchen.bench import mac_lock, parse_range
    from kitchen.lite_common import state_feats
    from kitchen.lite_train import Head
    from kitchen.runlog import Timer
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--ref", default="", help="KL の基準の頭(既定は --init)")
    ap.add_argument("--eval-only", action="store_true", help="--init を続けて動く評価で測るだけ(lite_batch.py と同じ手続き)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--skills", nargs="+", type=int, default=[0, 1, 3, 5, 6, 9])
    ap.add_argument("--inits", default="10-39", help="練習の初期状態")
    ap.add_argument("--chain", action="store_true", help="練習は別の動作をやり終えた後の場面から")
    ap.add_argument("--scenes", type=int, default=6, help="1 回の更新で試す場面の数")
    ap.add_argument("--mixed-only", action="store_true",
                    help="DAPO の dynamic sampling: 組の中に Jev の確率 0.5 以上と未満が両方ある組だけで更新する(全部失敗・全部成功の組の雑音を捨てる)")
    ap.add_argument("--eval-normal", action="store_true", help="評価で普段(前の動作なし)も測る")
    ap.add_argument("--keep-snaps", action="store_true", help="採点に使った画像を残す(既定は採点後に消す)")
    ap.add_argument("--recycle", type=int, default=0, help="練習の処理を何回の仕事ごとに作り直すか(0 = 作り直さない)。前の動作の覚えは消える")
    ap.add_argument("--modes", default="", choices=["", "chain", "normal", "both"],
                    help="練習の場面: both = 普段と続けて動くの両方(普段を距離の罰だけで守ると少しずつ崩れた、10/25 の新しい初期状態での比べ)")
    ap.add_argument("--adaptive-sigma", action="store_true",
                    help="場面ごとに揺らぎを決める: 組に揺らぎなしの 1 本を足し、それが当たるのに揺らぎありが外れるなら ×0.7、両方ほとんど外れるか揺らぎありもほぼ当たるなら ×1.4(Jev だけで)")
    ap.add_argument("--sigma-min", type=float, default=0.015)
    ap.add_argument("--sigma-max", type=float, default=0.15)
    ap.add_argument("--hard-below", type=float, default=0.0,
                    help="この成功率(Jev、場面ごとの移動平均)未満の場面は --hard-sigma・--hard-group で試す(0 なら使わない)。全部失敗の組からは学べない")
    ap.add_argument("--hard-sigma", type=float, default=0.12)
    ap.add_argument("--hard-group", type=int, default=16)
    ap.add_argument("--reuse", type=int, default=1, help="直近何回分の試行で更新するか(1 = いまの回だけ)。古い試行は集めたときの手との比で補正")
    ap.add_argument("--min-free-gb", type=float, default=1.5, help="ディスクの空きがこれ未満なら頭を保存して止める")
    ap.add_argument("--seq-ratio", action="store_true",
                    help="GSPO(Qwen): 比を試行全体で(塊の対数の比の平均 → exp)、クリップも試行単位。塊ごとの比は σ が小さいと振れが大きい")
    ap.add_argument("--no-std", action="store_true", help="Dr. GRPO: 組の優位を標準偏差で割らない")
    ap.add_argument("--curriculum", action="store_true",
                    help="場面ごとの成功率(Jev 0.5 以上の割合、指数移動平均)を覚え、p(1-p) + 0.05 に比例して引く(五分五分に近い場面を多めに)")
    ap.add_argument("--group", type=int, default=8, help="1 場面を何回試すか(GRPO の組)")
    ap.add_argument("--sigma", type=float, default=0.06)
    ap.add_argument("--train-steps", type=int, default=220, help="練習の打ち切り(評価は 300)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--clip-high", type=float, default=0.0, help="DAPO・SimpleVLA-RL の clip-higher: 比の上側の幅(0 なら --clip と同じ)")
    ap.add_argument("--replay", nargs="*", default=[], help="出発点の手の学習データ(全動作の場面)。ここでも KL の基準からの距離に罰をつける")
    ap.add_argument("--replay-kl", type=float, default=0.0,
                    help="PPO-ptx と同じ考え: 練習していない動作の場面でも基準の手から離れない罰(KL は練習の場面でしか効かず、火をつけるが 9→5 に崩れた)")
    ap.add_argument("--gdpo", action="store_true", help="GDPO: 報酬の項(狙い・早さ・壊さない・滑らかさ)ごとに組の中で比べてから足す")
    ap.add_argument("--kl", type=float, default=1.0, help="出発点の手からの距離の罰(正規化した出力の 2 乗)")
    ap.add_argument("--inner", type=int, default=2, help="1 回の更新で同じ試行を何周なめるか")
    ap.add_argument("--w-fast", type=float, default=0.3)
    ap.add_argument("--w-undo", type=float, default=0.5)
    ap.add_argument("--w-jerk", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--eval-skills", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--eval-inits", default="0-9")
    a = ap.parse_args()
    torch.set_num_threads(4)
    out = os.path.join(ROOT, "outputs", "lite", a.tag)
    snap = os.path.join(out, "snaps")
    os.makedirs(snap, exist_ok=True)
    if not a.eval_only:  # 評価だけなら Jev を使わない(頭のサーバなしで回せる)
        urllib.request.urlopen(BRAIN + "/health", timeout=10).read()
    h0 = dict(np.load(a.init, allow_pickle=True))
    assert str(h0["arch"]) == "grid"

    def build(path):
        hh = dict(np.load(path, allow_pickle=True))
        for k in ("s_mu", "s_sd", "l_mu", "l_sd", "g_mu", "g_sd", "a_mu", "a_sd"):
            assert np.allclose(hh[k], h0[k]), f"正規化の値が違う頭は KL の基準にできない: {k}"
        m = Head("grid", hh["L_W"].shape[0], 34, hh["P_W"].shape[0])
        with torch.no_grad():
            m.P.weight.copy_(torch.from_numpy(hh["P_W"].T)); m.P.bias.copy_(torch.from_numpy(hh["P_b"]))
            m.L.weight.copy_(torch.from_numpy(hh["L_W"].T)); m.L.bias.copy_(torch.from_numpy(hh["L_b"]))
            for i, l in enumerate([l for l in m.net if isinstance(l, torch.nn.Linear)]):
                l.weight.copy_(torch.from_numpy(hh[f"W{i}"].T)); l.bias.copy_(torch.from_numpy(hh[f"b{i}"]))
        m.eval()  # dropout は使わない(練習の時と同じ計算で比をとる)
        return m

    pol, ref = build(a.init), build(a.ref or a.init)
    for p in ref.parameters():
        p.requires_grad_(False)

    def head_dict():
        lin = [l for l in pol.net if isinstance(l, torch.nn.Linear)]
        d = {k: v for k, v in h0.items() if not (k.startswith("W") or (k.startswith("b") and k[1:].isdigit())) and k not in ("data", "init", "iters")}
        d.update({f"W{i}": l.weight.detach().numpy().T.astype(np.float32) for i, l in enumerate(lin)})
        d.update({f"b{i}": l.bias.detach().numpy().astype(np.float32) for i, l in enumerate(lin)})
        d.update(P_W=pol.P.weight.detach().numpy().T.astype(np.float32), P_b=pol.P.bias.detach().numpy().astype(np.float32),
                 L_W=pol.L.weight.detach().numpy().T.astype(np.float32), L_b=pol.L.bias.detach().numpy().astype(np.float32))
        return d

    a_mu, a_sd = torch.tensor(h0["a_mu"]), torch.tensor(h0["a_sd"])

    def inputs(F, G, S8):
        L = torch.tensor(((np.asarray(F, np.float32).reshape(len(F), -1) - h0["l_mu"]) / h0["l_sd"]).astype(np.float32))
        S = torch.tensor(((state_feats(np.asarray(S8, np.float32)) - h0["s_mu"]) / h0["s_sd"]).astype(np.float32))
        Gt = torch.tensor(((np.asarray(G, np.float32) - h0["g_mu"]) / h0["g_sd"]).astype(np.float32))
        return L, S, Gt

    opt = torch.optim.AdamW(pol.parameters(), lr=a.lr, weight_decay=0.0)
    rep_data = None
    if a.replay and a.replay_kl > 0:
        from kitchen.lite_train import load
        rF, rG, rS8, _, _, _, rnames, _ = load(a.replay)
        rep_data = (rF[:, rnames.index("指示_one-hot")], rG, rS8)
        print(f"復習の場面 {len(rF)}(全動作)で基準からの距離に罰 ×{a.replay_kl}", flush=True)
    ctx = mp.get_context("spawn")
    # 練習の処理は回を重ねると大きくなる(g6 で 1 本 6.0 → 7.2GB、スワップがディスクを食う)→ --recycle 回ごとに作り直す
    pool = ctx.Pool(a.workers, initializer=_init_worker, initargs=(a.n_envs, snap, a.keep_snaps), maxtasksperchild=a.recycle or None)
    modes = {"chain": [True], "normal": [False], "both": [False, True]}[a.modes or ("chain" if a.chain else "normal")]
    plan = [(s, i, c) for s in a.skills for i in parse_range(a.inits) for c in modes]  # 場面 = (動作, 初期状態, 続けて動くか)
    rng = np.random.default_rng(0)
    succ = {}  # 場面 → 成功率(Jev 0.5 以上の割合)
    sig_scene = {}  # 場面 → 揺らぎ(--adaptive-sigma)
    jerk_ref = None
    stopped = False
    from collections import deque
    buf = deque(maxlen=max(1, a.reuse))
    logf = open(os.path.join(out, "grpo_log.jsonl"), "a")
    t0 = time.time()

    def run(trials, sigma, max_steps, record, tag, chain=None):
        hd = head_dict()
        chain = a.chain if chain is None else chain
        key_ = lambda t: (t[0], t[1], bool(t[5]) if len(t) > 5 else chain)  # noqa: E731
        scenes_ = sorted({key_(t) for t in trials})  # 場面ごとに 1 本の処理へ(組と前の動作の覚えを 1 本の中に収める)
        owner = {sc: j % a.workers for j, sc in enumerate(scenes_)}
        chunks = [[t for t in trials if owner[key_(t)] == k] for k in range(a.workers)]
        res = pool.map(_rollout, [(hd, c, chain, sigma, max_steps, record, f"{tag}w{k}") for k, c in enumerate(chunks) if c])
        return [r for rr in res for r in rr]

    if a.eval_only:  # 評価だけ(続けて動く、--eval-normal なら普段も)。1 試行ずつの結果を eval_only_{chain,normal}.jsonl に
        ev = [(s, i, 0, 0) for s in a.eval_skills for i in parse_range(a.eval_inits)]
        for name, ch in [("chain", True)] + ([("normal", False)] if a.eval_normal else []):
            with mac_lock():
                E = run(ev, 0.0, EVAL_STEPS, False, f"eval_only_{name}", chain=ch)
            by = {str(s): sum(r["ok"] for r in E if r["skill"] == s) for s in a.eval_skills}
            print(f"{'続けて動く' if ch else '普段'} {sum(r['ok'] for r in E)}/{len(E)} {by}", flush=True)
            with open(os.path.join(out, f"eval_only_{name}.jsonl"), "w") as fo:
                for r in E:
                    fo.write(json.dumps({"head": a.init, "skill": r["skill"], "init": r["init"], "sigma": 0.0, "ok": bool(r["ok"]),
                                         "steps": int(r["steps"]), "chain": ch}, ensure_ascii=False) + "\n")
        pool.close()
        return
    with Timer("inner", f"オンライン RL(GRPO){a.tag}", init=os.path.basename(a.init), skills=a.skills, chain=a.chain, sigma=a.sigma,
               group=a.group, scenes=a.scenes, lr=a.lr, clip=a.clip, kl=a.kl, workers=a.workers) as tm:
        for it in range(1, a.iters + 1):
            ti = time.time()
            free_gb = shutil.disk_usage(ROOT).free / 2**30
            if free_gb < a.min_free_gb:  # メモリが詰まるとスワップがディスクを食い、OS に黙って止められる(10/25 g5)。自分で保存して降りる
                np.savez(os.path.join(ROOT, "models", "lite", f"{a.tag}_it{it - 1}_stop.npz"),
                         **head_dict(), init=a.init, iters=it - 1, data=np.array([f"grpo {a.tag}"]))
                print(f"ディスクの空き {free_gb:.1f}GB({a.min_free_gb}GB 未満)→ {it - 1} 回目の頭を保存して止める", flush=True)
                stopped = True
                break
            if a.curriculum:
                wts = np.array([succ.get(sc, 0.3) * (1 - succ.get(sc, 0.3)) + 0.05 for sc in plan])
                scenes = [plan[j] for j in rng.choice(len(plan), size=min(a.scenes, len(plan)), replace=False, p=wts / wts.sum())]
            else:
                scenes = [plan[j] for j in rng.choice(len(plan), size=min(a.scenes, len(plan)), replace=False)]
            trials = []
            for n, (s, i, c) in enumerate(scenes):
                hard = a.hard_below > 0 and (s, i, c) in succ and succ[(s, i, c)] < a.hard_below  # 試したことがあり、ほとんど当たらない場面
                sg, ng = (a.hard_sigma, a.hard_group) if hard else (a.sigma, a.group)
                if a.adaptive_sigma:
                    sg = sig_scene.get((s, i, c), a.sigma)
                trials += [(s, i, k, it * 10_000 + n * 100 + k, sg, c) for k in range(ng)]
                if a.adaptive_sigma:  # 揺らぎなしの 1 本(学びには使わず、揺らぎの決め方と練習の場面での素の成否に)
                    trials.append((s, i, -1, it * 10_000 + n * 100 + 99, 0.0, c))
            with mac_lock():
                R = run(trials, a.sigma, a.train_steps, True, f"i{it}")
            jerks = np.array([r["jerk"] for r in R])
            jerk_ref = float(np.mean(jerks)) if jerk_ref is None else 0.9 * jerk_ref + 0.1 * float(np.mean(jerks))
            for r in R:  # 報酬(自分の目と自分の動きの記録だけ)
                s, pa, pb = r["skill"], np.asarray(r["p_after"]), np.asarray(r["p_before"])
                undo = float(np.sum(np.maximum(0.0, np.delete(pb, s) - np.delete(pa, s))))
                r["comp"] = [float(pa[s]), float(pa[s] * (1 - r["steps"] / a.train_steps)), -undo, -r["jerk"] / max(jerk_ref, 1e-8)]
                r["reward"] = float(r["comp"][0] + a.w_fast * r["comp"][1] + a.w_undo * r["comp"][2] + a.w_jerk * r["comp"][3])
                r["undo"] = undo
            groups, greedy = {}, {}
            for r in R:
                if r["k"] < 0:  # 揺らぎなし(学びに使わない)
                    r["adv"] = 0.0
                    greedy[(r["skill"], r["init"], r["chain"])] = r
                    continue
                groups.setdefault((r["skill"], r["init"], r["chain"]), []).append(r)
            for g in groups.values():
                rw = np.array([r["reward"] for r in g])
                C = np.array([x["comp"] for x in g])
                sd_ = C.std(0)
                z = np.where(sd_ > 1e-3, (C - C.mean(0)) / (sd_ + 1e-6), 0.0)
                wts_ = np.array([1.0, a.w_fast, a.w_undo, a.w_jerk])
                for gi, r in enumerate(g):
                    if a.gdpo:  # GDPO(NVIDIA): 報酬の項ごとに組の中で比べ(平均を引いて標準偏差で割る)、重みをつけて足す
                        r["adv"] = float((z[gi] * wts_).sum()) if rw.std() > 1e-3 else 0.0
                    elif a.no_std:  # Dr. GRPO: 組の平均を引くだけ(標準偏差で割ると、ほぼ同じ結果の組の小さな差を引き伸ばす)
                        r["adv"] = float(r["reward"] - rw.mean()) if rw.std() > 1e-3 else 0.0
                    else:
                        r["adv"] = float((r["reward"] - rw.mean()) / (rw.std() + 1e-6)) if rw.std() > 1e-3 else 0.0
            for key, g in groups.items():  # 場面ごとの成功率を覚える(引き方に使う)
                hit = np.mean([r["p_after"][r["skill"]] >= 0.5 for r in g])
                succ[key] = hit if key not in succ else 0.5 * succ[key] + 0.5 * hit
                if a.adaptive_sigma and key in greedy:
                    # 揺らぎなしは当たるのに揺らぎありの多くが外れる → 揺らぎが大きすぎる、揺らぎなしも外れてほとんど当たらない → 探りが足りない、
                    # 揺らぎありもほぼ全部当たる → 比べる差が出ない。いずれも Jev の判定だけで決める(シミュレータの成否は使わない)
                    sg0 = sig_scene.get(key, a.sigma)
                    g_ok = greedy[key]["p_after"][key[0]] >= 0.5
                    if g_ok and hit < 0.5:
                        sg0 *= 0.7
                    elif (not g_ok and hit < 0.1) or hit > 0.9:
                        sg0 *= 1.4
                    sig_scene[key] = float(np.clip(sg0, a.sigma_min, a.sigma_max))
                if a.mixed_only and not (0 < hit < 1):
                    for r in g:
                        r["adv"] = 0.0
            with open(os.path.join(out, "trials.jsonl"), "a") as ft:  # 1 試行ずつ(報酬の項と、比べるためだけのシミュレータの成否)
                for r in R:
                    ft.write(json.dumps({"iter": it, "skill": r["skill"], "init": r["init"], "k": r["k"], "ok": bool(r["ok"]), "sim_ok": bool(r["sim_ok"]),
                                         "p_target": round(float(r["p_after"][r["skill"]]), 4), "steps": int(r["steps"]), "undo": round(r["undo"], 4),
                                         "jerk": round(r["jerk"], 5), "reward": round(r["reward"], 4), "adv": round(r["adv"], 4), "sigma": r["sigma"], "chain": r["chain"]}) + "\n")
            use_now = [r for r in R if r["adv"] != 0.0 and r["F"] and r["k"] >= 0]  # 揺らぎなし(k=-1)は比が定まらないので使わない
            buf.append(use_now)  # 試行の再利用(PPO の同一データでの複数回更新と同じ考え): 試すのが高く学ぶのは安いので、直近 --reuse 回分の試行も比(GSPO)で補正して使う
            use = [r for u in buf for r in u]
            stat = {"iter": it, "trials": len(R), "p_target": round(float(np.mean([r["p_after"][r["skill"]] for r in R])), 3),
                    "jev_ok": int(sum(r["p_after"][r["skill"]] >= 0.8 for r in R)), "sim_ok": int(sum(r["sim_ok"] for r in R)),
                    "reward": round(float(np.mean([r["reward"] for r in R])), 3), "undo": round(float(np.mean([r["undo"] for r in R])), 3),
                    "jerk": round(float(np.mean(jerks)), 5), "sec_roll": round(time.time() - ti, 1),
                    "groups_used": len({(r["skill"], r["init"], r["chain"]) for r in use_now}), "groups": len(groups), "trials_update": len(use),
                    "chain_trials": int(sum(r["chain"] for r in R))}
            if greedy:
                stat.update(greedy_jev=f"{sum(g['p_after'][g['skill']] >= 0.5 for g in greedy.values())}/{len(greedy)}",
                            greedy_sim=f"{sum(g['ok'] for g in greedy.values())}/{len(greedy)}",
                            sigma_med=round(float(np.median(list(sig_scene.values()))), 4) if sig_scene else a.sigma)
            if use:
                F = np.concatenate([np.stack(r["F"]) for r in use]); G = np.concatenate([np.stack(r["G"]) for r in use])
                S8 = np.concatenate([np.stack(r["S"]) for r in use])
                MU = torch.tensor(np.concatenate([np.stack(r["mu"]) for r in use]))[:, :, :6]
                A_ = torch.tensor(np.concatenate([np.stack(r["a"]) for r in use]))[:, :, :6]
                ADV = torch.tensor(np.concatenate([[r["adv"]] * len(r["F"]) for r in use]), dtype=torch.float32)
                TID = torch.tensor(np.concatenate([[k] * len(r["F"]) for k, r in enumerate(use)]))  # 塊 → 試行の番号
                TADV = torch.tensor([r["adv"] for r in use], dtype=torch.float32)
                SIG2 = torch.tensor(np.concatenate([[r["sigma"] ** 2] * len(r["F"]) for r in use]), dtype=torch.float32)  # 塊ごとの σ²
                L, S, Gt = inputs(F, G, S8)
                with torch.no_grad():
                    lp_old = -((A_ - MU) ** 2).sum((1, 2)) / (2 * SIG2)
                    y_ref = ref(L, S, Gt)
                n = len(L)
                cf, kls, pgs = [], [], []
                nt = len(use)
                for _ in range(a.inner):
                    if a.seq_ratio:  # GSPO: 試行ごとに束ねて、塊の対数の比の平均で 1 つの比
                        tperm = torch.randperm(nt)
                        batches_ = [torch.nonzero(torch.isin(TID, tperm[t0:t0 + 8])).squeeze(1) for t0 in range(0, nt, 8)]
                    else:
                        perm = torch.randperm(n)
                        batches_ = [perm[b0:b0 + 256] for b0 in range(0, n, 256)]
                    for b in batches_:
                        y = pol(L[b], S[b], Gt[b])
                        mu_new = (y.reshape(len(b), -1, 7) * a_sd + a_mu)[:, :EXEC, :6]
                        lp = -((A_[b] - mu_new) ** 2).sum((1, 2)) / (2 * SIG2[b])
                        if a.seq_ratio:
                            tid = TID[b]
                            uniq, inv = torch.unique(tid, return_inverse=True)
                            d = (lp - lp_old[b]).clamp(-10, 10)
                            m = torch.zeros(len(uniq)).index_add(0, inv, d) / torch.bincount(inv, minlength=len(uniq)).float()
                            ratio = torch.exp(m)
                            adv_t = TADV[uniq]
                            pg = -torch.min(ratio * adv_t, torch.clamp(ratio, 1 - a.clip, 1 + (a.clip_high or a.clip)) * adv_t).mean()
                        else:
                            ratio = torch.exp(torch.clamp(lp - lp_old[b], -10, 10))
                            pg = -torch.min(ratio * ADV[b], torch.clamp(ratio, 1 - a.clip, 1 + (a.clip_high or a.clip)) * ADV[b]).mean()
                        kl = ((y - y_ref[b]) ** 2).mean()
                        loss = pg + a.kl * kl
                        if rep_data is not None:  # 全動作の場面で基準の手から離れない
                            ri = np.random.randint(0, len(rep_data[0]), 128)
                            rL, rS, rGt = inputs(rep_data[0][ri][:, None, :], rep_data[1][ri], rep_data[2][ri])
                            with torch.no_grad():
                                ry_ref = ref(rL, rS, rGt)
                            loss = loss + a.replay_kl * ((pol(rL, rS, rGt) - ry_ref) ** 2).mean()
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                        opt.step()
                        cf.append(float(((ratio - 1).abs() > a.clip).float().mean())); kls.append(float(kl)); pgs.append(float(pg))
                stat.update(samples=int(n), pg=round(float(np.mean(pgs)), 4), kl=round(float(np.mean(kls)), 6), clipfrac=round(float(np.mean(cf)), 3))
            if it % a.eval_every == 0 or it == a.iters:
                ev = [(s, i, 0, 0) for s in a.eval_skills for i in parse_range(a.eval_inits)]
                with mac_lock():  # 続けて動くは必ず前の動作から(--modes both では a.chain が偽で、10/25 の g6 は 2 回とも普段を測っていた)
                    E = run(ev, 0.0, EVAL_STEPS, False, f"e{it}", chain=True)
                stat["val_chain"] = f"{sum(r['ok'] for r in E)}/{len(E)}"  # ほかの評価と同じく試行中に一度でも成功したか
                stat["val_by_skill"] = {str(s): sum(r["ok"] for r in E if r["skill"] == s) for s in a.eval_skills}
                evals = [("chain", E)]
                if a.eval_normal:  # 普段(前の動作なし)も同じ処理で(評価を別に並べるとメモリが足りない)
                    with mac_lock():
                        N = run(ev, 0.0, EVAL_STEPS, False, f"n{it}", chain=False)
                    stat["val_normal"] = f"{sum(r['ok'] for r in N)}/{len(N)}"
                    stat["val_normal_by_skill"] = {str(s): sum(r["ok"] for r in N if r["skill"] == s) for s in a.eval_skills}
                    evals.append(("normal", N))
                for name, rows in evals:  # 対応のある比べのために 1 試行ずつ
                    with open(os.path.join(out, f"eval_it{it}_{name}.jsonl"), "w") as fo:
                        for r in rows:
                            fo.write(json.dumps({"head": f"{a.tag}_it{it}", "skill": r["skill"], "init": r["init"], "sigma": 0.0, "ok": bool(r["ok"]),
                                                 "steps": int(r["steps"]), "chain": name == "chain"}) + "\n")
                np.savez(os.path.join(ROOT, "models", "lite", f"{a.tag}_it{it}.npz"),
                         **head_dict(), init=a.init, iters=it, data=np.array([f"grpo {a.tag}"]))
            stat["sec"] = round(time.time() - ti, 1)
            logf.write(json.dumps(stat, ensure_ascii=False) + "\n"); logf.flush()
            print(f"{it} 回目: 狙いの確率 {stat['p_target']}・報酬 {stat['reward']}・Jev の当たり {stat['jev_ok']}/{stat['trials']}"
                  + (f"・比のクリップ {stat.get('clipfrac')}・KL {stat.get('kl')}" if use else "")
                  + (f"・続けて動く {stat['val_chain']}" if "val_chain" in stat else "")
                  + (f"・普段 {stat['val_normal']}" if "val_normal" in stat else "") + f"・{stat['sec']} 秒(合計 {(time.time() - t0) / 60:.0f} 分)", flush=True)
        tm.info.update(iters=a.iters)
    if not stopped:
        np.savez(os.path.join(ROOT, "models", "lite", f"{a.tag}.npz"), **head_dict(), init=a.init, iters=a.iters, data=np.array([f"grpo {a.tag}"]))
    pool.close()


if __name__ == "__main__":
    main()
