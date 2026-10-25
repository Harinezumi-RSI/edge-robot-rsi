#!/bin/zsh
# 備え付けを試行ごとの種で置く(再現する評価)・記録に残して描き直しで戻す(画像と行動が食い違わない学習データ)で、95% を目指し直す
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
room() { local d=$(df -g ~ | awk 'NR==2{print $4}'); local s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6/1024)}'); [ $d -ge 4 ] && [ $s -le 9 ]; }
until grep -q "E2E_DONE" logs/queue_e2e.out 2>/dev/null && grep -q "KNOBS_DONE" logs/queue_knobs.out 2>/dev/null; do sleep 60; done
ALL=(0 1 2 3 4 5 6 7 8 9)
# A: 固定した条件の基準
until room; do sleep 120; done
.venv/bin/python kitchen/lite_batch.py --skills $ALL --inits 0-9 --head models/lite/d1.npz --name "基準: d1(備え付け固定・撮り直し)" --out outputs/lite/d1_fix/val.jsonl > logs/fix_d1.log 2>&1
say "A1 d1 $(tail -1 logs/fix_d1.log)"
.venv/bin/python kitchen/fast_rollout.py --skills $ALL --inits 0-9 --n-envs 8 --n-action-steps 5 --name "基準: SmolVLA 5 step(備え付け固定)" --out outputs/fast/val_all_fix.jsonl > logs/fix_smolvla.log 2>&1
say "A2 SmolVLA $(grep '合計' logs/fix_smolvla.log | tail -1)"
# B: 学習データを作り直す(先生の練習 300 本、備え付けの位置つき)
until room; do sleep 120; done
mkdir -p data/rsi/r0f/demos
.venv/bin/python kitchen/fast_rollout.py --skills $ALL --inits 10-39 --n-envs 8 --n-action-steps 5 --record data/rsi/r0f/demos --record-all \
  --kind practice --name "先生の練習(備え付けの位置つき)" --out outputs/rsi/r0f_practice.jsonl > logs/fix_r0f.log 2>&1
say "B1 先生の練習 $(grep '合計' logs/fix_r0f.log | tail -1)"
.venv/bin/python kitchen/lite_data.py --demos data/rsi/r0f/demos --every 4 --backend dino --out data/lite/r0f_dino.npz > logs/fix_r0f_data.log 2>&1
say "B2 $(tail -1 logs/fix_r0f_data.log)"
.venv/bin/python kitchen/lite_train.py --data data/lite/r0f_dino.npz --out models/lite/l0f.npz > logs/fix_l0f_train.log 2>&1 && say "B3 $(tail -1 logs/fix_l0f_train.log)"
.venv/bin/python kitchen/lite_batch.py --skills $ALL --inits 0-9 --head models/lite/l0f.npz --name "l0f(正しいデータで写すだけ)" --out outputs/lite/l0f/val.jsonl > logs/fix_l0f.log 2>&1
say "B3 l0f $(tail -1 logs/fix_l0f.log)"
until room; do sleep 120; done
.venv/bin/python kitchen/lite_rsi.py --tag d1f --mode dagger --prev models/lite/l0f.npz --pool data/lite/r0f_dino.npz > logs/lite_rsi_d1f.out 2>&1 || say "d1f が落ちた"
say "B4 d1f $(tail -1 logs/lite_rsi_d1f.out | cut -c1-400)"
say "FIX_DONE"
