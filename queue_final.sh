#!/bin/zsh
# 9 周目 = 最終テスト(初期状態 40〜49、ループ中に一度も使っていない)。1 回だけ回す。
cd ~/dev/pai-final; source env.sh
# 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ、env.sh が読む)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
R() { .venv/bin/python kitchen/fast_rollout.py --skills 1 7 4 2 --inits 40-49 --n-envs 8 --n-action-steps 5 --snap data/snaps/final "$@"; }
R --sigma 0.1 --tag base_s10 --name "最終テスト: σ=0.1 戻すなし・元の重み" --out outputs/final/s10_base.jsonl > logs/final_a1.log 2>&1
say "(a) 元 $(tail -1 logs/final_a1.log)" | tee -a logs/final.out
R --sigma 0.1 --tag ris2_s10 --delta models/rsi/ris2.safetensors --name "最終テスト: σ=0.1 戻すなし・ris2" --out outputs/final/s10_ris2.jsonl > logs/final_a2.log 2>&1
say "(a) ris2 $(tail -1 logs/final_a2.log)" | tee -a logs/final.out
R --sigma 0 --tag base_s0 --name "最終テスト: ずれなし・元の重み" --out outputs/final/s0_base.jsonl > logs/final_b1.log 2>&1
say "(b) 元 $(tail -1 logs/final_b1.log)" | tee -a logs/final.out
R --sigma 0 --tag ris2_s0 --delta models/rsi/ris2.safetensors --name "最終テスト: ずれなし・ris2" --out outputs/final/s0_ris2.jsonl > logs/final_b2.log 2>&1
say "(b) ris2 $(tail -1 logs/final_b2.log)" | tee -a logs/final.out
S=$(date '+%Y-%m-%d %H:%M')
.venv-mlx/bin/python kitchen/jev.py from_runs data/judge_final outputs/final/s10_base.jsonl data/snaps/final outputs/final/s10_ris2.jsonl data/snaps/final outputs/final/s0_base.jsonl data/snaps/final outputs/final/s0_ris2.jsonl data/snaps/final >> logs/jev_final.log 2>&1
.venv-mlx/bin/python kitchen/jev.py extract data/judge_final >> logs/jev_final.log 2>&1
.venv-mlx/bin/python kitchen/jev.py evaltest data/judge_train0 data/judge_final --zero >> logs/jev_final.log 2>&1
.venv/bin/python kitchen/runlog.py add judge "最終テスト: 改造 Jev とゼロショット(初期状態 40〜49 の画像)" --start "$S" > /dev/null
say "FINAL_DONE" | tee -a logs/final.out
