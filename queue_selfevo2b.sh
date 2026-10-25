#!/bin/zsh
# 先生なしの自己進化 2 周目 x2o(10/24 夜、作り直し b)。1 周目 x1o は普段 90・続けて動く 60(c1o 91・63)で上がらず、家事の通しでは 14/30(c1o 20)と下向き。
# 練習 A(混ざる 4 動作)の揺らぎつきの当たりをなぞって皿を寄せるなどが崩れたと見て、練習の手は c1o に戻し、x1o の当たりは難しい場面の B・C だけ使う。
# 難しい 3 動作(器をコンロ・チーズ・ワインをラック)だけに絞り、新しい初期状態 20〜39 で 1 場面 32 回ずつ(約 1,900 試行)。
#  0. queue_retry.sh の本番のやり直しの比べ(家事の通し)が終わったら、その後の Long を止める(x2o の後に回す)
#  1. 練習 3 本並べ(鍵はこの列が持つ): 器をコンロ・チーズ・ワインをラック、すべて変わった場面、σ 0.1、32 回ずつ
#  2. 改造 Jev で採点 → 組の中の当たりを重みつきで  3. 学び直し(c1o のデータ + x1o・x2o の当たり)→ 評価  4. Long
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1
say() { echo "$(date '+%m/%d %H:%M') $*"; }
okc() { echo "$(grep -c '"ok": true' $1)/$(wc -l < $1 | tr -d ' ')"; }
room() { local d=$(df -g ~ | awk 'NR==2{print $4}'); local s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6/1024)}'); [ $d -ge 6 ] && [ $s -le 9 ]; }
LOCK=$PAI_MAC_LOCK; ME="paifinal $$"
take() { [ -n "$LOCK" ] || return 0; until mkdir $LOCK 2>/dev/null; do sleep 30; done; echo "$ME" > $LOCK/owner; say "鍵を持った"; }
release() { [ -n "$LOCK" ] && [ "$(cat $LOCK/owner 2>/dev/null)" = "$ME" ] && { rm -f $LOCK/owner; rmdir $LOCK; say "鍵を返した"; }; return 0; }
trap release EXIT; trap 'release; exit 1' INT TERM

# 0. 本番のやり直しの比べが終わるのを待ち、Long を止める
until grep -q "dev4_x1o_r2" logs/queue_retry.out 2>/dev/null || ! kill -0 46959 2>/dev/null; do sleep 20; done
kill 46959 2>/dev/null; sleep 3
pkill -f "suite_loop.py eval --suite libero_10"; sleep 2
say "本番のやり直しの比べが終わった。Long は x2o の後に回す"

# 1. 練習 3 本並べ
T=x2o; R=data/lite_rsi/$T; P=outputs/lite/$T
mkdir -p $R/demos $R/snaps $P
until room; do sleep 120; done
take
practice() {  # $1 = 名前, $2 = 動作
  PAI_MAC_LOCK= .venv/bin/python kitchen/lite_batch.py --skills $2 --inits 20-39 --chain --samples 32 --explore 0.1 --head models/lite/c1o.npz \
    --record $R/demos --snap $R/snaps --tag $T$1 --kind practice --name "自己進化の練習 $T$1(σ 0.1・32 回ずつ)" --out $P/practice_$1.jsonl \
    > logs/selfevo_${T}_$1.log 2>&1
}
practice B 1 &
practice D 6 &
practice C 9 &
wait
for k in B D C; do say "練習 $k $(okc $P/practice_$k.jsonl)"; done

# 2. 採点 → 当たり
for k in B D C; do
  .venv-mlx/bin/python kitchen/judge_runs.py $P/practice_$k.jsonl $R/snaps > logs/selfevo_${T}_judge_$k.log 2>&1 || say "採点 $k が落ちた"
done
.venv/bin/python - $T <<'PY' > logs/selfevo_${T}_select.log 2>&1
import json, sys
T = sys.argv[1]
P, R = f"outputs/lite/{T}", f"data/lite_rsi/{T}"
TAU = 0.8
summary, lines = {}, []
for k in "BDC":
    J = [json.loads(l) for l in open(f"{P}/practice_{k}_judged.jsonl")]
    grp = {}
    for r in J:
        grp.setdefault((r["skill"], r["init"]), []).append(r)
    acc_n = acc_true = w_sum = 0
    for key, rs in grp.items():
        rate = sum(r["p_jev"] >= TAU for r in rs) / len(rs)
        if rate >= 1.0:
            continue
        w = 1 + round(2 * (1 - rate))
        for r in rs:
            if r["p_jev"] >= TAU:
                lines += [f"{R}/demos/{r['demo']}\n"] * w
                acc_n += 1
                acc_true += bool(r["labels"][str(r["skill"])])
                w_sum += w
    rates = [sum(r["p_jev"] >= TAU for r in rs) / len(rs) for rs in grp.values()]
    summary[k] = {"trials": len(J), "jev_ok": sum(r["p_jev"] >= TAU for r in J), "sim_ok": sum(bool(r["labels"][str(r["skill"])]) for r in J),
                  "groups": len(grp), "mixed": sum(0 < x < 1 for x in rates), "all_ng": sum(x == 0 for x in rates),
                  "accepted": acc_n, "accepted_true": acc_true, "weighted": w_sum}
open(f"{P}/accept.txt", "w").writelines(lines)
json.dump(summary, open(f"{P}/select.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False))
PY
say "選んだ当たり $(tail -1 logs/selfevo_${T}_select.log | cut -c1-500)"

# 3. 学び直し → 評価(x1o の当たりは難しい場面の B・C だけ)
grep -E "_x1o[BC]\.npz" outputs/lite/x1o/accept.txt > outputs/lite/x1o/accept_hard.txt
[ -f data/lite/x1o_hard.npz ] || .venv/bin/python kitchen/lite_data.py --demos data/lite_rsi/x1o/demos --accept-list outputs/lite/x1o/accept_hard.txt --every 4 --backend dino \
  --out data/lite/x1o_hard.npz > logs/selfevo_x1o_hard_data.log 2>&1 || { say "x1o の難しい場面のデータが作れない"; exit 1; }
.venv/bin/python kitchen/lite_data.py --demos $R/demos --accept-list $P/accept.txt --every 4 --backend dino --out data/lite/${T}_explore.npz > logs/selfevo_${T}_data.log 2>&1 || { say "データが作れない"; exit 1; }
.venv/bin/python kitchen/lite_train.py --data data/lite/r0o_dino.npz data/lite/d1o_dagger.npz data/lite/c1o_dagger.npz data/lite/x1o_hard.npz data/lite/${T}_explore.npz \
  --out models/lite/$T.npz > logs/selfevo_${T}_train.log 2>&1 || { say "学習が落ちた"; exit 1; }
PAI_MAC_LOCK= .venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --head models/lite/$T.npz --name "自己進化 $T の評価" \
  --out $P/val.jsonl > logs/selfevo_${T}_val.log 2>&1
PAI_MAC_LOCK= .venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --chain --head models/lite/$T.npz --name "自己進化 $T の評価(続けて動く)" \
  --out $P/val_chain.jsonl > logs/selfevo_${T}_val_chain.log 2>&1
say "自己進化 $T: 普段 $(okc $P/val.jsonl)(c1o 91・x1o 90)・続けて動く $(okc $P/val_chain.jsonl)(c1o 63・x1o 60)"
release
# 家事の通し(判断用、c1o 20/30・x1o 14/30)
pkill -f "kitchen/brain_server.py"; sleep 2
nohup .venv-mlx/bin/python kitchen/brain_server.py > logs/brain_server_dev4_$T.log 2>&1 &
BP=$!
until curl -s -o /dev/null http://127.0.0.1:8765/health; do sleep 3; kill -0 $BP 2>/dev/null || { say "頭のサーバが落ちた"; break; }; done
PAI_HISTORY=1 .venv/bin/python kitchen/e2e_session.py --hand models/lite/$T.npz --inits 0-9 --tag dev4_$T > logs/e2e_dev4_$T.log 2>&1 || say "通し dev4_$T が落ちた"
pkill -f "kitchen/brain_server.py"; sleep 2
[ -f outputs/e2e/dev4_$T/summary.json ] && say "通し(dev4_$T)$(tr '\n' ' ' < outputs/e2e/dev4_$T/summary.json | cut -c1-300)"

# 4. Long
SU=libero_10; O=outputs/suite/$SU; D=data/suite/$SU; M=models/suite/$SU
mkdir -p $O/smolvla_n5 $O/l0 $O/d1 $D $M
until room; do sleep 120; done
.venv/bin/python kitchen/suite_loop.py eval --suite $SU --inits 0-9 --teacher smolvla --teacher-steps 5 --out $O/smolvla_n5/val.jsonl > logs/suite_${SU}_smolvla.log 2>&1
say "$SU 先生 SmolVLA(5 step ずつ)$(okc $O/smolvla_n5/val.jsonl)"
[ -f $D/r0.npz ] || .venv/bin/python kitchen/suite_loop.py teach --suite $SU --inits 10-39 --out $D/r0.npz > logs/suite_${SU}_teach.log 2>&1 || { say "$SU 先生の練習が落ちた"; exit 1; }
say "$SU 先生の練習 $(tail -1 logs/suite_${SU}_teach.log | cut -c1-120)"
.venv/bin/python kitchen/lite_train.py --data $D/r0.npz --out $M/l0.npz > logs/suite_${SU}_train_l0.log 2>&1 || { say "$SU l0 の学習が落ちた"; exit 1; }
.venv/bin/python kitchen/suite_loop.py eval --suite $SU --inits 0-9 --head $M/l0.npz --out $O/l0/val.jsonl > logs/suite_${SU}_l0.log 2>&1
say "$SU l0(先生を写す)$(okc $O/l0/val.jsonl)"
[ -f $D/d1_dagger.npz ] || .venv/bin/python kitchen/suite_loop.py dagger --suite $SU --inits 10-39 --head $M/l0.npz --out $D/d1_dagger.npz > logs/suite_${SU}_dagger.log 2>&1 || { say "$SU DAgger が落ちた"; exit 1; }
.venv/bin/python kitchen/lite_train.py --data $D/r0.npz $D/d1_dagger.npz --out $M/d1.npz > logs/suite_${SU}_train_d1.log 2>&1 || { say "$SU d1 の学習が落ちた"; exit 1; }
.venv/bin/python kitchen/suite_loop.py eval --suite $SU --inits 0-9 --head $M/d1.npz --out $O/d1/val.jsonl > logs/suite_${SU}_d1.log 2>&1
say "$SU d1(DAgger)$(okc $O/d1/val.jsonl)"
say "SELFEVO2_LONG_DONE"
