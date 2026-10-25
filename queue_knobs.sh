#!/bin/zsh
# 学習し直さない調整: 予測した 8 step のうち何 step 実行するか(いまは 4)を 2・8 に。d1・普段の撮り方(79/100 と比べる)
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
for x in 2 8; do
  .venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --head models/lite/d1.npz --n-envs 8 --exec $x \
    --name "軽い手 d1 を $x step ずつ実行" --out outputs/lite/d1_exec$x/val.jsonl > logs/lite_eval_d1_exec$x.log 2>&1
  say "exec $x $(tail -1 logs/lite_eval_d1_exec$x.log)"
done
say "KNOBS_DONE"
