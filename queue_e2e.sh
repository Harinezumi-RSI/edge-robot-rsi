#!/bin/zsh
# 家事ロボットの通しの評価(開発中、初期状態 0〜9): 頭のアンサンブルの評価の後、SmolVLA の測り直しの前に
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
until grep -q "ENS_DONE" logs/queue_ens.out 2>/dev/null; do sleep 30; done
n() { [ -f $1 ] && grep -c '"ok": true' $1 || echo 0; }
HAND=models/lite/d1.npz; BEST=$(n outputs/lite/d1/val.jsonl)
for e in d1x4 d1x4_8; do
  v=$(n outputs/lite/ens_$e/val.jsonl)
  if [ $v -gt $BEST ]; then BEST=$v; HAND=$(grep -o '"head": "[^"]*"' logs/lite_eval_ens_$e.log 2>/dev/null | head -1 | cut -d'"' -f4); fi
done
[ $HAND = models/lite/d1.npz ] || HAND=$( [ $(n outputs/lite/ens_d1x4_8/val.jsonl) -ge $(n outputs/lite/ens_d1x4/val.jsonl) ] && echo models/lite/d1.npz,models/lite/d1_s1.npz,models/lite/d1_s2.npz,models/lite/d1_s3.npz,models/lite/d1_dino8.npz || echo models/lite/d1.npz,models/lite/d1_s1.npz,models/lite/d1_s2.npz,models/lite/d1_s3.npz )
say "手 $HAND(評価 $BEST/100)"
pkill -f "kitchen/brain_server.py"; sleep 2
nohup .venv-mlx/bin/python kitchen/brain_server.py > logs/brain_server_e2e.log 2>&1 &
until curl -s -o /dev/null http://127.0.0.1:8765/health; do sleep 3; done
.venv/bin/python kitchen/e2e_session.py --hand $HAND --inits 0-9 --tag dev > logs/e2e_dev.log 2>&1 || say "通しの評価が落ちた"
say "通し $(tail -12 logs/e2e_dev.log | tr '\n' ' ' | cut -c1-500)"
pkill -f "kitchen/brain_server.py"
say "E2E_DONE"
