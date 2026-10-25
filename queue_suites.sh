#!/bin/zsh
# LIBERO のほかの 3 種に、家事ロボットの手の作り方をそのまま当てる(10/24、デモの完成後に他の 3 種へ)。
# 先生は SmolVLA(4 種で学習済みの公開重み、手元にある)を 5 step ずつ(毎 step は 1 波 10 分で 1 種 7 時間かかるので、Goal の SmolVLA 時代と同じ作法に)。公開の OpenVLA-OFT は Goal 用の重みしか無く、ほかの種の重みは約 15GB を落とす必要がある(人の確認待ち)。
# 種ごとに: 先生の判断用(初期状態 0〜9)→ 先生の練習 300(10〜39)を写す l0 → 判断用 → DAgger(8 step おきにその場で先生の答え)d1 → 判断用。
# 最終テスト(40〜49)は最後に 1 回だけ(この列ではしない)。
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
room() { local d=$(df -g ~ | awk 'NR==2{print $4}'); local s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6/1024)}'); [ $d -ge 6 ] && [ $s -le 9 ]; }
okc() { echo "$(grep -c '"ok": true' $1)/$(wc -l < $1 | tr -d ' ')"; }
for SU in libero_object libero_spatial libero_10; do
  O=outputs/suite/$SU; D=data/suite/$SU; M=models/suite/$SU
  mkdir -p $O/smolvla_n5 $O/l0 $O/d1 $D $M
  until room; do sleep 120; done
  .venv/bin/python kitchen/suite_loop.py eval --suite $SU --inits 0-9 --teacher smolvla --teacher-steps 5 --out $O/smolvla_n5/val.jsonl > logs/suite_${SU}_smolvla.log 2>&1
  say "$SU 先生 SmolVLA(5 step ずつ)$(okc $O/smolvla_n5/val.jsonl)"
  until room; do sleep 120; done
  [ -f $D/r0.npz ] || .venv/bin/python kitchen/suite_loop.py teach --suite $SU --inits 10-39 --out $D/r0.npz > logs/suite_${SU}_teach.log 2>&1 || { say "$SU 先生の練習が落ちた"; continue; }
  say "$SU 先生の練習 $(tail -1 logs/suite_${SU}_teach.log | cut -c1-120)"
  .venv/bin/python kitchen/lite_train.py --data $D/r0.npz --out $M/l0.npz > logs/suite_${SU}_train_l0.log 2>&1 || { say "$SU l0 の学習が落ちた"; continue; }
  .venv/bin/python kitchen/suite_loop.py eval --suite $SU --inits 0-9 --head $M/l0.npz --out $O/l0/val.jsonl > logs/suite_${SU}_l0.log 2>&1
  say "$SU l0(先生を写す)$(okc $O/l0/val.jsonl)"
  until room; do sleep 120; done
  [ -f $D/d1_dagger.npz ] || .venv/bin/python kitchen/suite_loop.py dagger --suite $SU --inits 10-39 --head $M/l0.npz --out $D/d1_dagger.npz > logs/suite_${SU}_dagger.log 2>&1 || { say "$SU DAgger が落ちた"; continue; }
  .venv/bin/python kitchen/lite_train.py --data $D/r0.npz $D/d1_dagger.npz --out $M/d1.npz > logs/suite_${SU}_train_d1.log 2>&1 || { say "$SU d1 の学習が落ちた"; continue; }
  .venv/bin/python kitchen/suite_loop.py eval --suite $SU --inits 0-9 --head $M/d1.npz --out $O/d1/val.jsonl > logs/suite_${SU}_d1.log 2>&1
  say "$SU d1(DAgger)$(okc $O/d1/val.jsonl)"
done
say "SUITES_DONE"
