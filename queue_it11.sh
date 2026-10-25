#!/bin/zsh
# 11 周目: 全 10 動作で立て直しを練習する
cd ~/dev/pai-final; source env.sh
# 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ、env.sh が読む)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
.venv/bin/python kitchen/fast_rollout.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --n-envs 8 --n-action-steps 5 --sigma 0.1 \
  --snap data/snaps/ris_val --tag base_all --name "11 周目: 元の重み σ=0.1 戻すなし・全 10 動作(比べる相手)" --out outputs/ris/val_all_s10_base.jsonl > logs/it11_base.log 2>&1
say "(1) 元 $(tail -1 logs/it11_base.log)" | tee -a logs/it11.out
.venv/bin/python kitchen/rsi_round.py --round 11 --tag risall1 --skills 0 1 2 3 4 5 6 7 8 9 --sigma 0.1 --practice-home-first \
  --practice-policy base --tau 0.8 --pool-tags risall1 ris1 ris2 --train-steps 2000 --eval-also-sigma0 --seed-offset 5000 > logs/rsi_risall1_main.out 2>&1
say "(2) risall1 $(tail -1 logs/rsi_risall1_main.out | cut -c1-400)" | tee -a logs/it11.out
say "IT11_DONE" | tee -a logs/it11.out
