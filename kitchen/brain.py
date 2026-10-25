"""ロボットの「頭」: ひとことから意図を推し量って手順を組む(plan)/カメラ画像で出来を確かめる(verify)。

ローカルの視覚言語モデル(Qwen3.5、MLX 4bit)で動かす。.venv-mlx で実行する。
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kitchen.skills import JA, POLITE, SCENE_JA, SKILLS, STATE_JA  # noqa: E402

MODEL = os.environ.get("PAI_BRAIN_MODEL", "mlx-community/Qwen3.5-4B-MLX-4bit")

PLAN_SYS = f"""あなたは家庭用ロボットの頭脳です。人のひとことから、その人が本当に望んでいることを推し量り、
ロボットができる動作だけを組み合わせて手順を作ります。

場面: {SCENE_JA}

ロボットができる動作(番号: 内容):
""" + "\n".join(f"{i}: {ja}" for i, _, ja in SKILLS) + """

決まり:
- 手順は上の番号だけで作る。必要な動作だけを、実行する順に並べる(多くても 4 つ)。
- 直接の命令でなくても、気持ちや状況から望みを推し量る(例:「寒い」なら温かいものを用意する)。
- 上の動作ではかなえられない頼み(火を消す、物を捨てる など)や、何もしなくてよい場合は手順を空にし、理由を返事で伝える。
- 返事は話し言葉で 1〜2 文。

次の JSON だけを出力する:
{"intent": "推し量った望み", "reply": "人への返事", "plan": [番号, ...]}"""

# 手順の手本(会話の形で見せる)。評価に使う発話とは重ならないものだけ
FEWSHOT = [
    ("ワインをラックに戻しておいて", {"intent": "ワインボトルをラックに戻してほしい", "reply": "ワインをラックに戻しますね。", "plan": [9]}),
    ("おなかすいた、器あっためといて", {"intent": "器の中身を温めたい", "reply": "器をコンロにかけて火をつけますね。", "plan": [1, 7]}),
    ("パンに塗るチーズ、器に出しておいて", {"intent": "クリームチーズを器に用意してほしい", "reply": "クリームチーズを器に入れておきます。", "plan": [6]}),
    ("食器しまっといて", {"intent": "器を片付けてほしい", "reply": "器を引き出しにしまいますね。", "plan": [3]}),
    ("冷蔵庫からビール取って", {"intent": "ビールを飲みたい", "reply": "ごめんなさい、冷蔵庫の物は扱えません。", "plan": []}),
    ("コンロの前に皿を寄せて、器をのせておいて", {"intent": "コンロの前で皿と器を使いたい", "reply": "皿をコンロの前に寄せて、器をのせます。", "plan": [5, 8]}),
    ("ありがとう、もう大丈夫だよ", {"intent": "頼みごとはない", "reply": "どういたしまして。また呼んでくださいね。", "plan": []}),
    ("真ん中の引き出しから何か出したいんだけど", {"intent": "真ん中の引き出しを開けてほしい", "reply": "真ん中の引き出しを開けますね。", "plan": [0]}),
]

VERIFY_SYS = """あなたは家庭用ロボットの確認係です。ロボットのカメラ画像を見て、指定した動作が完了しているかを判定します。
次の JSON だけを出力する: {"done": true または false, "reason": "見えたこと(短く)"}"""

_model = _processor = None


def _load():
    global _model, _processor
    if _model is None:
        from mlx_vlm import load
        _model, _processor = load(MODEL)
    return _model, _processor


THINK = os.environ.get("PAI_THINK", "0") == "1"


def _chat(system, user, images=None, max_tokens=256, think=None, shots=(), temperature=0.0):
    from mlx_vlm import generate
    model, processor = _load()
    think = THINK if think is None else think
    messages = [{"role": "system", "content": system}]
    for u, a in shots:
        messages += [{"role": "user", "content": u}, {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)}]
    messages.append({"role": "user", "content": user})
    kw = {}
    if think:
        max_tokens = max(max_tokens, 2048)
    try:  # Qwen3 系の考える時間(thinking)は既定で切る(返事を速く・JSON だけにする)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=think)
    except TypeError:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if images:
        from mlx_vlm.prompt_utils import apply_chat_template
        prompt = apply_chat_template(processor, model.config, messages,
                                     num_images=len(images), enable_thinking=think)
        kw["image"] = images
    out = generate(model, processor, prompt, max_tokens=max_tokens, temperature=temperature, verbose=False, **kw)
    return getattr(out, "text", out)


def _json(text):
    text = text.split("</think>")[-1]
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def plan(utterance, history=()):
    """history: この会話のこれまでの頼み [(ひとこと, {"intent", "reply", "plan"}), ...]。「その上に」のような前の頼みを指す言葉のため。"""
    raw = _chat(PLAN_SYS, utterance, shots=list(FEWSHOT) + list(history))
    d = _json(raw) or {}
    ids = [int(x) for x in d.get("plan", []) if str(x).isdigit() and int(x) in JA]
    return {"intent": d.get("intent", ""), "reply": d.get("reply", ""), "plan": ids, "raw": raw}


# 見てから察する: 計画の前にカメラで場面を読み(改造 Jev で 10 目標の状態)、すでにできていることを手順から外す。
# 場面の説明から「火はついていない」を外す(火の状態はカメラで読む)
PLAN_SYS_SEEN = PLAN_SYS.replace("コンロ(火はついていない)", "コンロ").replace("- 返事は話し言葉で 1〜2 文。", """- 「いまの様子」はカメラで確かめた、いまそうなっていること。そうなっている目標の動作は手順に入れない。
  頼まれたことが全部そうなっていれば手順を空にし、そのことを返事で伝える。
- 返事は話し言葉で 1〜2 文。""")
NOTHING = "特に変わったところはない(物は最初の置き場所、引き出しは閉まっていて、火はついていない)"


def seen_text(done_ids):
    return "。".join(STATE_JA[g] for g in sorted(done_ids)) + "。" if done_ids else NOTHING


def seen_user(utterance, done_ids):
    return f"いまの様子(カメラで確認): {seen_text(done_ids)}\n人のひとこと: {utterance}"


# 手本: いつもの 8 本(様子は「特になし」)+ すでにできている場面 2 本(評価の発話とは別の言い方)
FEWSHOT_SEEN = [(seen_user(u, []), a) for u, a in FEWSHOT] + [
    (seen_user("コンロの前に皿を寄せて、器をのせておいて", [5]),
     {"intent": "コンロの前で皿と器を使いたい", "reply": "お皿はもうコンロの前にあるので、器をのせますね。", "plan": [8]}),
    (seen_user("食器しまっといて", [3]),
     {"intent": "器を片付けてほしい", "reply": "器はもう引き出しにしまってありますよ。", "plan": []}),
]


def plan_subtract(utterance, done_ids, history=()):
    """見てから察する(採用版): 頭は望み(どの目標か)だけを推し量り、すでに満たされている目標を引き算する。
    動作 = 目標の述語を満たすこと なので、満たされている目標の動作は要らない。返事は引いた分を言い添える。
    言葉で様子を渡して頭に任せる版(plan_seen)は、様子に書いた目標を手順に足してしまった(17 周目)。"""
    r = plan(utterance, history)
    done = set(done_ids)
    keep = [g for g in r["plan"] if g not in done]
    gone = [g for g in r["plan"] if g in done]
    if gone:
        why = "、".join(STATE_JA[g] for g in gone)
        r["reply"] = (f"もう{why}ので、このままで大丈夫ですよ。" if not keep else
                      f"もう{why}ので、" + "。それから、".join(POLITE[g] for g in keep) + "ね。")
    return {**r, "plan": keep, "skipped": gone}


def plan_seen(utterance, done_ids):
    """(不採用)done_ids: いまの場面ですでに満たされている目標の番号。様子を言葉で渡して頭に任せる。"""
    raw = _chat(PLAN_SYS_SEEN, seen_user(utterance, done_ids), shots=FEWSHOT_SEEN)
    d = _json(raw) or {}
    ids = [int(x) for x in d.get("plan", []) if str(x).isdigit() and int(x) in JA]
    return {"intent": d.get("intent", ""), "reply": d.get("reply", ""), "plan": ids, "raw": raw}


# 耳の後処理: 音声認識の聞き間違い(同じ音・似た音の別の言葉)を、台所の場面を知っている頭が直す。
# 手本の例文は評価に使った文とは別に作った(型は往復テストで見えたもの: コンロ→今度・ラック→楽・器→器物)。
STT_FIX_SYS = """あなたは台所の家事ロボットの耳の後処理係です。音声認識の結果には、同じ音や似た音の別の言葉への聞き間違いがあります。
台所の言葉(コンロ・火・器・お皿・棚・引き出し・ワインボトル・ワインラック・クリームチーズ)を手がかりに、
ロボットへの頼みとして自然になるよう聞き間違いだけを直し、直した文を 1 行だけ出力する。
直す所がなければ、そのまま出力する。言い換え・付け足し・説明はしない。"""
STT_FIX_SHOTS = [
    ("今度の上に器を置いといて", "コンロの上に器を置いといて"),
    ("器物を棚にしまって", "器を棚にしまって"),
    ("ワインを楽に置いてくれる？", "ワインをラックに置いてくれる？"),
    ("チーズを器に入れて、お皿で出して", "チーズを器に入れて、お皿で出して"),
    ("明日の天気を教えて", "明日の天気を教えて"),
]


def fix_stt(text):
    """聞き取った文 → 聞き間違いを直した文(1 行)。"""
    from mlx_vlm import generate
    model, processor = _load()
    messages = [{"role": "system", "content": STT_FIX_SYS}]
    for u, a in STT_FIX_SHOTS:
        messages += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": text})
    try:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    out = generate(model, processor, prompt, max_tokens=64, temperature=0.0, verbose=False)
    line = getattr(out, "text", out).split("</think>")[-1].strip().splitlines()
    fixed = (line[0] if line else "").strip().strip("「」\"")
    return fixed or text


def verify(image, skill_id):
    raw = _chat(VERIFY_SYS, f"動作「{JA[skill_id]}」は完了していますか?", images=[image], max_tokens=96)
    d = _json(raw) or {}
    return {"done": bool(d.get("done", False)), "reason": d.get("reason", ""), "raw": raw}


if __name__ == "__main__":
    import time
    for u in sys.argv[1:]:
        t = time.time()
        r = plan(u)
        print(f"{time.time() - t:5.1f}s  {u} → {r['plan']}  意図: {r['intent']} / 返事: {r['reply']}")


# ---------------------------------------------------------------------------------------------
# Jev 風の判定器(System 1): 文章を生成させず、答えの記号 1 文字の確率を 1 回の順伝播で読む。
#   choice: 選択肢から 1 つ(確率つき)/ p_yes: はい・いいえの「はい」の確率
# 生成の言い回しに左右されず、確信度が数字で出る(低ければ聞き返す・やり直すに使える)。
LABELS = "ABCDEFGHIJKLMNOP"


def _label_ids(processor, n):
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    ids = []
    for c in LABELS[:n]:
        t = tok.encode(c, add_special_tokens=False)
        assert len(t) == 1, (c, t)
        ids.append(t[0])
    return ids


def choice(system, question, options, images=None, shots=()):
    """options のどれかを選ばせ、各選択肢の確率(記号のトークン確率を選択肢の中で正規化)を返す。"""
    import mlx.core as mx
    from mlx_vlm import stream_generate
    model, processor = _load()
    body = question + "\n\n" + "\n".join(f"{LABELS[k]}. {o}" for k, o in enumerate(options)) + \
        "\n\n記号 1 文字だけで答える。"
    messages = [{"role": "system", "content": system}]
    for u, a in shots:
        messages += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": body})
    if images:
        from mlx_vlm.prompt_utils import apply_chat_template
        prompt = apply_chat_template(processor, model.config, messages, num_images=len(images), enable_thinking=False)
    else:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    res = next(iter(stream_generate(model, processor, prompt, image=images, max_tokens=1, temperature=0.0)))
    lp = res.logprobs
    lp = lp.reshape(-1) if hasattr(lp, "reshape") else mx.array(lp).reshape(-1)
    ids = _label_ids(processor, len(options))
    sel = mx.array([lp[i].item() for i in ids])
    p = mx.softmax(sel).tolist()
    mass = float(mx.exp(sel).sum().item())  # 記号に乗った確率の合計(小さければ答え方がずれている)
    best = max(range(len(options)), key=lambda k: p[k])
    return {"probs": p, "best": best, "conf": p[best], "label_mass": mass}


def p_yes(question, image, system=VERIFY_SYS.split("\n")[0]):
    r = choice(system, question, ["はい", "いいえ"], images=[image])
    return r["probs"][0], r["label_mass"]


END = "終わり(これ以上は何もしない/頼みはかなえられない)"
JUDGE_SYS = PLAN_SYS.split("決まり:")[0] + """決まり:
- 人のひとことから望みを推し量り、ロボットが次にするべき動作を 1 つ選ぶ。
- 望みがすでにかなっている、上の動作ではかなえられない、頼みごとではない、のどれかなら「終わり」を選ぶ。
- 必要な動作だけを選ぶ。関係のない動作は選ばない。

例(ひとこと → 選ぶ動作の順):
""" + "\n".join(f"「{u}」→ " + " → ".join([JA[i] for i in a["plan"]] + ["終わり"]) for u, a in FEWSHOT)


def plan_judge(utterance, max_steps=4):
    """Jev 風の判定を繰り返して手順を作る(次の一手を選ぶ分類 × 数回)。各手の確信度も返す。"""
    done, confs = [], []
    for _ in range(max_steps):
        hist = "、".join(JA[i] for i in done) if done else "まだ何もしていない"
        q = f"ひとこと:「{utterance}」\nここまでにした動作: {hist}\n次にするべき動作はどれ?"
        r = choice(JUDGE_SYS, q, [JA[i] for i in range(10)] + [END])
        confs.append(round(r["conf"], 3))
        if r["best"] == 10 or r["best"] in done:
            break
        done.append(r["best"])
    return {"plan": done, "conf": confs}


CHECK_SYS = PLAN_SYS.split("決まり:")[0] + "あなたは手順の確認係です。人のひとことと、ロボットがこれから実行する手順を見て、" \
    "その手順がその人の望みにちょうど合っているか(足りない動作も余計な動作もないか)を判定します。"


def plan_ok(utterance, plan_ids):
    """手順がひとことの望みに合っている確率(Jev 風の判定。低ければ動く前に聞き返す)。"""
    steps = " → ".join(JA[i] for i in plan_ids) if plan_ids else "何もしない"
    q = f"ひとこと:「{utterance}」\nロボットの手順: {steps}\nこの手順はその人の望みにちょうど合っていますか?"
    r = choice(CHECK_SYS, q, ["はい", "いいえ"])
    return r["probs"][0]


def plan_rerank(utterance, n_samples=4, temp=0.7, ask_below=0.15):
    """生成で手順の候補を出し(System 2)、Jev 風の判定で一番合うものを選ぶ(System 1)。
    候補 = 決定的な生成 1 本 + 温度つきの生成 n_samples 本 + 次の一手を選ぶ判定の手順。
    生成が「何もしない」と言うときはそのまま返す(できない頼み・頼みでないもの)。
    選んだ手順の確信度が ask_below 未満なら、動く前に聞き返す(ask=True)。"""
    import mlx.core as mx
    base = plan(utterance)
    if not base["plan"]:
        return {**base, "cands": [], "p": None, "ask": False}
    cands = {tuple(base["plan"])}
    for k in range(n_samples):
        mx.random.seed(k)
        d = _json(_chat(PLAN_SYS, utterance, shots=FEWSHOT, temperature=temp)) or {}
        ids = [int(x) for x in d.get("plan", []) if str(x).isdigit() and int(x) in JA]
        if ids:
            cands.add(tuple(ids))
    cands.add(tuple(plan_judge(utterance)["plan"]) or tuple(base["plan"]))
    scored = sorted(((plan_ok(utterance, list(c)), list(c)) for c in cands if c), reverse=True)
    p, best = scored[0]
    return {"intent": base["intent"], "reply": base["reply"], "plan": best, "p": round(p, 3),
            "ask": p < ask_below, "cands": [(round(q, 3), c) for q, c in scored]}


# ---- 速さの改造: 手順を先に出させ、手順の配列が閉じた瞬間に返す(返事の文章は待たない) ----
FEWSHOT_FAST = [(u, {"plan": a["plan"], "reply": a["reply"]}) for u, a in FEWSHOT]
PLAN_SYS_FAST = PLAN_SYS.replace('{"intent": "推し量った望み", "reply": "人への返事", "plan": [番号, ...]}',
                                 '{"plan": [番号, ...], "reply": "人への返事"}')


def plan_fast(utterance, max_tokens=96):
    """手順の配列が閉じた時点で生成を打ち切る(ロボットはすぐ動き出せる)。返事は打ち切りまでに出た分だけ。"""
    from mlx_vlm import stream_generate
    model, processor = _load()
    messages = [{"role": "system", "content": PLAN_SYS_FAST}]
    for u, a in FEWSHOT_FAST:
        messages += [{"role": "user", "content": u}, {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)}]
    messages.append({"role": "user", "content": utterance})
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    text = ""
    for r in stream_generate(model, processor, prompt, max_tokens=max_tokens, temperature=0.0):
        text += r.text
        m = re.search(r'"plan"\s*:\s*\[([^\]]*)\]', text)
        if m:
            ids = [int(x) for x in re.findall(r"\d+", m.group(1)) if int(x) in JA]
            return {"plan": ids, "raw": text}
    return {"plan": [], "raw": text}
