#!/bin/zsh
# 13 周目: 手を実時間より速く(10 step まとめて実行 + ノイズ除去 3 回)しても成功が落ちないか。11 周目の後に回す
cd ~/dev/pai-final; source env.sh
# 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ、env.sh が読む)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
until grep -q "IT11_DONE" logs/it11.out 2>/dev/null; do sleep 120; done
.venv/bin/python kitchen/fast_rollout.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --n-envs 8 --n-action-steps 10 --num-steps 3 \
  --snap data/snaps/speed --tag n10k3 --name "13 周目: 10 step まとめて + 反復 3 回(全 10 動作 × 検証)" --out outputs/fast/speed_n10k3.jsonl > logs/it13.log 2>&1
say "n10k3 $(tail -1 logs/it13.log)" | tee -a logs/it13.out
say "IT13_DONE" | tee -a logs/it13.out
