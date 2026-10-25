#!/bin/zsh
# 家事ロボットの通しの評価のやり直し(備え付けを種で置く・会話の流れを頭に渡す)。手は固定した条件の評価で良いほう(d1 / d1f)
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
until grep -q "SELECT_DONE" logs/queue_select.out 2>/dev/null; do sleep 60; done
n() { [ -f $1 ] && grep -c '"ok": true' $1 || echo 0; }
HAND=models/lite/d1.npz
[ -f models/lite/d1f.npz ] && [ $(n outputs/lite/d1f/val.jsonl) -gt $(n outputs/lite/d1_fix/val.jsonl) ] && HAND=models/lite/d1f.npz
[ -f models/lite/route.json ] && HAND=models/lite/route.json  # 動作ごとに選んだ頭(queue_select)
say "手 $HAND(d1 $(n outputs/lite/d1_fix/val.jsonl)・d1f $(n outputs/lite/d1f/val.jsonl))"
pkill -f "kitchen/brain_server.py"; sleep 2
nohup .venv-mlx/bin/python kitchen/brain_server.py > logs/brain_server_e2e2.log 2>&1 &
until curl -s -o /dev/null http://127.0.0.1:8765/health; do sleep 3; done
for h in 1 0; do
  PAI_HISTORY=$h .venv/bin/python kitchen/e2e_session.py --hand $HAND --inits 0-9 --tag dev2_h$h > logs/e2e_dev2_h$h.log 2>&1 || say "通し h$h が落ちた"
  say "通し(会話の流れ $h)$(tr '\n' ' ' < outputs/e2e/dev2_h$h/summary.json | cut -c1-400)"
done
pkill -f "kitchen/brain_server.py"
say "E2E2_DONE"
