#!/bin/zsh
# 8 周目: (a) ずれなしを同じ並びで元の重み(壊していないか)(b) 練習していない 4 動作で σ=0.1 の立て直しが移るか(元 / ris1)
# (c) ris2: 学習後の手(ris1)で戻す動きなしに練習 → 改造 Jev で採点 → ris1 の手本と合わせて学び直す → 評価
cd ~/dev/pai-final; source env.sh
# 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ、env.sh が読む)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
.venv/bin/python kitchen/fast_rollout.py --skills 1 7 4 2 --inits 0-9 --n-envs 8 --n-action-steps 5 --sigma 0 --snap data/snaps/val_rsi --tag base_s0 \
  --name "8 周目(a): ずれなし・元の重み(ris1 と同じ並び)" --out outputs/ris/val_s0_base.jsonl > logs/it8a.log 2>&1
say "(a) $(tail -1 logs/it8a.log)" | tee -a logs/it8.out
.venv/bin/python kitchen/fast_rollout.py --skills 8 6 0 5 --inits 0-9 --n-envs 8 --n-action-steps 5 --sigma 0.1 --snap data/snaps/ris_val --tag tr_base \
  --name "8 周目(b): 練習していない 4 動作 σ=0.1・元の重み" --out outputs/ris/transfer_base.jsonl > logs/it8b1.log 2>&1
say "(b) 元 $(tail -1 logs/it8b1.log)" | tee -a logs/it8.out
.venv/bin/python kitchen/fast_rollout.py --skills 8 6 0 5 --inits 0-9 --n-envs 8 --n-action-steps 5 --sigma 0.1 --snap data/snaps/ris_val --tag tr_ris1 \
  --delta models/rsi/ris1.safetensors --name "8 周目(b): 練習していない 4 動作 σ=0.1・ris1" --out outputs/ris/transfer_ris1.jsonl > logs/it8b2.log 2>&1
say "(b) ris1 $(tail -1 logs/it8b2.log)" | tee -a logs/it8.out
.venv/bin/python kitchen/rsi_round.py --round 8 --tag ris2 --skills 1 7 4 2 --sigma 0.1 --practice-policy prev --tau 0.8 \
  --pool-tags ris1 ris2 --seed-offset 4000 --prev-delta models/rsi/ris1.safetensors > logs/rsi_ris2_main.out 2>&1
say "(c) ris2 $(tail -1 logs/rsi_ris2_main.out | cut -c1-300)" | tee -a logs/it8.out
say "IT8_DONE" | tee -a logs/it8.out
