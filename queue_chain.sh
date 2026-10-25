#!/bin/zsh
# 続けて動く評価(狙いの動作の前に、邪魔しない別の動作を同じ手で): 変わった場面への強さ。d1 と d1f(正しいデータで学び直した手)
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
until grep -q "FIX_DONE" logs/queue_fix.out 2>/dev/null; do sleep 60; done
for h in d1 d1f; do
  [ -f models/lite/$h.npz ] || continue
  .venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --head models/lite/$h.npz --chain \
    --name "続けて動く評価 $h" --out outputs/lite/${h}_chain/val.jsonl > logs/chain_$h.log 2>&1
  say "続けて $h $(tail -1 logs/chain_$h.log)"
done
say "CHAIN_DONE"
