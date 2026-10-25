#!/bin/zsh
# 耳の後処理の効き方(114 本の聞き取り + 未使用の 30 本)。通しの評価が終わってから
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1
say() { echo "$(date '+%m/%d %H:%M') $*"; }
until grep -q "E2E_DONE" logs/queue_e2e.out 2>/dev/null; do sleep 30; done
for w in old fresh; do
  .venv-mlx/bin/python kitchen/stt_fix_eval.py $w > logs/stt_fix_$w.log 2>&1
  say "後処理 $w $(grep '^{' logs/stt_fix_$w.log | tail -1 | cut -c1-300)"
done
say "STTFIX_DONE"
