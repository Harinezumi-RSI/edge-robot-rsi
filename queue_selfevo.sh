#!/bin/zsh
# 先生なしの自己進化 1 周目 x1o(10/24 夜、方針: LLM の強化学習のように多数の試行から自己改善)。
#  0. Spatial の DAgger(走っている suite_loop)が終わったら、d1 の学習と判断用の評価(queue_suites.sh の続き。Long は後回し)
#  1. 練習を 3 本並べる(鍵はこの列が持ち、中の練習は鍵なしで走らせる。手は c1o、すべて変わった場面 = 別の動作の後から):
#     A 混ざる組がある 4 動作(中の引き出し・引き出し(上)+器・皿を寄せる・チーズ)を σ 0.06 で 8 回ずつ(GRPO の組)
#     B・C ほぼ解けない 2 動作(器をコンロ・ワインをラック)を σ 0.1 で 1 場面 48 回ずつ(大量に試して当たりを探す)
#  2. 改造 Jev で採点(シミュレータの正解は使わない)→ 組(動作・初期状態)の中で混ざった組の成功だけ、組の成功率が低いほど重く
#  3. 学び直し(c1o のデータ + 当たり)→ 判断用の普段の場面と変わった場面で評価(c1o は 91・63)
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1
say() { echo "$(date '+%m/%d %H:%M') $*"; }
okc() { echo "$(grep -c '"ok": true' $1)/$(wc -l < $1 | tr -d ' ')"; }
LOCK=$PAI_MAC_LOCK; ME="paifinal $$"
take() { [ -n "$LOCK" ] || return 0; until mkdir $LOCK 2>/dev/null; do sleep 30; done; echo "$ME" > $LOCK/owner; say "鍵を持った"; }
release() { [ -n "$LOCK" ] && [ "$(cat $LOCK/owner 2>/dev/null)" = "$ME" ] && { rm -f $LOCK/owner; rmdir $LOCK; say "鍵を返した"; }; return 0; }
trap release EXIT; trap 'release; exit 1' INT TERM

# 0. Spatial の続き
SU=libero_spatial; O=outputs/suite/$SU; D=data/suite/$SU; M=models/suite/$SU
while kill -0 13268 2>/dev/null; do sleep 30; done
if [ -f $D/d1_dagger.npz ]; then
  .venv/bin/python kitchen/lite_train.py --data $D/r0.npz $D/d1_dagger.npz --out $M/d1.npz > logs/suite_${SU}_train_d1.log 2>&1
  .venv/bin/python kitchen/suite_loop.py eval --suite $SU --inits 0-9 --head $M/d1.npz --out $O/d1/val.jsonl > logs/suite_${SU}_d1.log 2>&1
  say "$SU d1(DAgger)$(okc $O/d1/val.jsonl)"
else
  say "$SU の DAgger のデータが無い"
fi

# 1. 練習を 3 本並べる
T=x1o; R=data/lite_rsi/$T; P=outputs/lite/$T
mkdir -p $R/demos $R/snaps $P
take
practice() {  # $1 = 名前, $2 = 動作, $3 = 初期状態, $4 = 回数, $5 = 揺らぎ
  PAI_MAC_LOCK= .venv/bin/python kitchen/lite_batch.py --skills ${=2} --inits $3 --chain --samples $4 --explore $5 --head models/lite/c1o.npz \
    --record $R/demos --snap $R/snaps --tag $T$1 --kind practice --name "自己進化の練習 $T$1(σ $5・$4 回ずつ)" --out $P/practice_$1.jsonl \
    > logs/selfevo_${T}_$1.log 2>&1
}
practice A "0 3 5 6" 10-29 8 0.06 &
practice B "1" 10-19 48 0.1 &
practice C "9" 10-19 48 0.1 &
wait
for k in A B C; do say "練習 $k $(okc $P/practice_$k.jsonl)"; done

# 2. 改造 Jev で採点 → 組の中の当たりを選ぶ
for k in A B C; do
  .venv-mlx/bin/python kitchen/judge_runs.py $P/practice_$k.jsonl $R/snaps > logs/selfevo_${T}_judge_$k.log 2>&1 || say "採点 $k が落ちた"
done
.venv/bin/python - $T <<'PY' > logs/selfevo_${T}_select.log 2>&1
import json, sys
T = sys.argv[1]
P, R = f"outputs/lite/{T}", f"data/lite_rsi/{T}"
TAU = 0.8
summary, lines = {}, []
for k in "ABC":
    J = [json.loads(l) for l in open(f"{P}/practice_{k}_judged.jsonl")]
    grp = {}
    for r in J:
        grp.setdefault((r["skill"], r["init"]), []).append(r)
    acc_n = acc_true = w_sum = 0
    for key, rs in grp.items():
        rate = sum(r["p_jev"] >= TAU for r in rs) / len(rs)
        if rate >= 1.0:
            continue
        w = 1 + round(2 * (1 - rate))  # 組の成功率が低いほど重く(GRPO の優位 1 - 成功率)
        for r in rs:
            if r["p_jev"] >= TAU:
                lines += [f"{R}/demos/{r['demo']}\n"] * w
                acc_n += 1
                acc_true += bool(r["labels"][str(r["skill"])])
                w_sum += w
    rates = [sum(r["p_jev"] >= TAU for r in rs) / len(rs) for rs in grp.values()]
    summary[k] = {"trials": len(J), "jev_ok": sum(r["p_jev"] >= TAU for r in J), "sim_ok": sum(bool(r["labels"][str(r["skill"])]) for r in J),
                  "groups": len(grp), "mixed": sum(0 < x < 1 for x in rates), "all_ok": sum(x == 1 for x in rates), "all_ng": sum(x == 0 for x in rates),
                  "accepted": acc_n, "accepted_true": acc_true, "weighted": w_sum}
open(f"{P}/accept.txt", "w").writelines(lines)
json.dump(summary, open(f"{P}/select.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False))
PY
say "選んだ当たり $(tail -1 logs/selfevo_${T}_select.log | cut -c1-600)"

# 3. 学び直し → 評価
.venv/bin/python kitchen/lite_data.py --demos $R/demos --accept-list $P/accept.txt --every 4 --backend dino --out data/lite/${T}_explore.npz > logs/selfevo_${T}_data.log 2>&1 || { say "データが作れない"; exit 1; }
.venv/bin/python kitchen/lite_train.py --data data/lite/r0o_dino.npz data/lite/d1o_dagger.npz data/lite/c1o_dagger.npz data/lite/${T}_explore.npz \
  --out models/lite/$T.npz > logs/selfevo_${T}_train.log 2>&1 || { say "学習が落ちた"; exit 1; }
PAI_MAC_LOCK= .venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --head models/lite/$T.npz --name "自己進化 $T の評価" \
  --out $P/val.jsonl > logs/selfevo_${T}_val.log 2>&1
PAI_MAC_LOCK= .venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --chain --head models/lite/$T.npz --name "自己進化 $T の評価(続けて動く)" \
  --out $P/val_chain.jsonl > logs/selfevo_${T}_val_chain.log 2>&1
say "自己進化 $T: 普段 $(okc $P/val.jsonl)(c1o 91)・続けて動く $(okc $P/val_chain.jsonl)(c1o 63)"
release
say "SELFEVO_DONE"
