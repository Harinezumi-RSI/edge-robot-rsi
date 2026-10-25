#!/bin/zsh
# 95% に向けて: 特徴を細かく・大きく(同じ学習データ = 先生の成功 + DAgger 1 周目、答えはそのまま。
# DAgger 2 周目は引き出しが 6→0・8→2 と崩れた(79→70)ので外し、比べる相手は d1 を同じ撮り方で = outputs/lite/d1_lazy_s0)
#   dino8: DINOv2-small・8×8 のマス / dinob: DINOv2-base・4×4。評価は手が見る step だけ撮る(20 周目)。
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
room() { local d=$(df -g ~ | awk 'NR==2{print $4}'); local s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6/1024)}'); [ $d -ge 5 ] && [ $s -le 9 ]; }
until grep -q "LITE_LOOP2_DONE" logs/queue_lite_loop2.out 2>/dev/null; do sleep 60; done
for b in dino8 dinob; do
  until room; do say "空きを待つ"; sleep 120; done
  .venv/bin/python kitchen/lite_refeat.py --src data/lite/r0_dino.npz --every 4 --backend $b --out data/lite/r0_$b.npz > logs/refeat_r0_$b.log 2>&1 || { say "r0 $b が落ちた"; continue; }
  .venv/bin/python kitchen/lite_refeat.py --src data/lite/d1_dagger.npz --every 8 --backend $b --out data/lite/d1_dagger_$b.npz > logs/refeat_d1_$b.log 2>&1 || { say "d1 $b が落ちた"; continue; }
  say "特徴 $b $(tail -1 logs/refeat_d1_$b.log)"
  .venv/bin/python kitchen/lite_train.py --data data/lite/r0_$b.npz data/lite/d1_dagger_$b.npz --out models/lite/d1_$b.npz > logs/lite_train_d1_$b.log 2>&1 && say "$(tail -1 logs/lite_train_d1_$b.log)"
  .venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --head models/lite/d1_$b.npz --n-envs 8 --lazy-camera \
    --name "軽い手 d1 の特徴 $b(撮り方 4 回に 1 回)" --out outputs/lite/d1_${b}_lazy_s0/val.jsonl > logs/lite_eval_d1_${b}.log 2>&1
  say "$b $(tail -1 logs/lite_eval_d1_${b}.log)"
done
say "FEAT_DONE"
