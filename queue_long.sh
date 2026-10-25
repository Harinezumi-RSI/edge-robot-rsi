#!/bin/zsh
# LIBERO-Long(libero_10)を先生 SmolVLA で最後まで(10/25、今夜回す)。先生の評価 44/100 は済み(10/24)。
# 自己進化の 3 回目の確かめ(lite_grpo.py --eval-only、test3_*)が終わってから: 先生の練習 → 写す(l0)→ 評価 → DAgger → d1 → 評価
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1
say() { echo "$(date '+%m/%d %H:%M') $*"; }
okc() { echo "$(grep -c '"ok": true' $1)/$(wc -l < $1 | tr -d ' ')"; }
room() { local d=$(df -g ~ | awk 'NR==2{print $4}'); local s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6/1024)}'); [ $d -ge 3 ] && [ $s -le 12 ]; }
until [ -f outputs/lite/test3_g6c_it20/eval_only_normal.jsonl ] && ! ps -axo command | grep -q "[l]ite_grpo.py --eval-only"; do sleep 30; done
say "自己進化の 3 回目の確かめが終わった。Long を始める"
SU=libero_10; O=outputs/suite/$SU; D=data/suite/$SU; M=models/suite/$SU
mkdir -p $O/l0 $O/d1 $D $M
until room; do sleep 120; done
[ -f $D/r0.npz ] || .venv/bin/python kitchen/suite_loop.py teach --suite $SU --inits 10-39 --out $D/r0.npz > logs/suite_${SU}_teach.log 2>&1 || { say "$SU 先生の練習が落ちた"; exit 1; }
say "$SU 先生の練習 $(tail -1 logs/suite_${SU}_teach.log | cut -c1-120)"
.venv/bin/python kitchen/lite_train.py --data $D/r0.npz --out $M/l0.npz > logs/suite_${SU}_train_l0.log 2>&1 || { say "$SU l0 の学習が落ちた"; exit 1; }
.venv/bin/python kitchen/suite_loop.py eval --suite $SU --inits 0-9 --head $M/l0.npz --out $O/l0/val.jsonl > logs/suite_${SU}_l0.log 2>&1
say "$SU l0(先生を写す)$(okc $O/l0/val.jsonl)"
[ -f $D/d1_dagger.npz ] || .venv/bin/python kitchen/suite_loop.py dagger --suite $SU --inits 10-39 --head $M/l0.npz --out $D/d1_dagger.npz > logs/suite_${SU}_dagger.log 2>&1 || { say "$SU DAgger が落ちた"; exit 1; }
.venv/bin/python kitchen/lite_train.py --data $D/r0.npz $D/d1_dagger.npz --out $M/d1.npz > logs/suite_${SU}_train_d1.log 2>&1 || { say "$SU d1 の学習が落ちた"; exit 1; }
.venv/bin/python kitchen/suite_loop.py eval --suite $SU --inits 0-9 --head $M/d1.npz --out $O/d1/val.jsonl > logs/suite_${SU}_d1.log 2>&1
say "$SU d1(DAgger)$(okc $O/d1/val.jsonl)"
say "LONG_DONE"
