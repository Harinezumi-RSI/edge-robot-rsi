#!/bin/zsh
# 見直しの間隔: SmolVLA は 5 step ずつ → 毎 step 推論で 74 → 9 割近くに上がった。軽い手(推論 15ms)も 8 step の予測のうち 1・2 step だけ実行して見直す。
# d1f(4 step ずつで 77/100)と同じ固定した条件で
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
for x in 1 2; do
  .venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --head models/lite/d1f.npz --exec $x \
    --name "軽い手 d1f を $x step ずつ見直す" --out outputs/lite/d1f_exec$x/val.jsonl > logs/d1f_exec$x.log 2>&1
  say "d1f exec $x $(tail -1 logs/d1f_exec$x.log)"
done
say "EXEC_DONE"
