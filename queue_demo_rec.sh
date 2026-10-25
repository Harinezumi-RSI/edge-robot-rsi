#!/bin/zsh
# 動画のデモを撮る(10/24)。最終テストの列(queue_final2.sh)が終わってから、Mac の鍵を持って(ほかの重い処理と重ねない):
#  1. 下見: 候補の発話を 声 → 聞き取り → 手順 に通し、決めておいた条件(data/demo_rehearse.json)に合う最初の候補を選ぶ(ロボットは動かさない)
#  2. 撮る: Wi-Fi なしの囲いの中(offline_demo.sh)で、手 c1o・初期状態 0 で 1 回ずつ
#     v3_oneshot = 火をつけて → 寒くなってきたね(見てから察する)/ v3_askback = あいまいな頼み → 聞き返す → 言い直し
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
say() { echo "$(date '+%m/%d %H:%M') $*"; }
LOCK=$PAI_MAC_LOCK  # 鍵(env.local.sh)。空なら取らない
ME="paifinal-demo $$"
while kill -0 16907 2>/dev/null; do sleep 30; done  # queue_final2.sh
if [ -n "$LOCK" ]; then
  until mkdir $LOCK 2>/dev/null; do sleep 30; done
  echo "$ME" > $LOCK/owner
  release() { [ "$(cat $LOCK/owner 2>/dev/null)" = "$ME" ] && { rm -f $LOCK/owner; rmdir $LOCK; }; }
  trap release EXIT; trap 'release; exit 1' INT TERM
fi
say "鍵を持った"
pkill -f "kitchen/brain_server.py"; sleep 2
nohup .venv-mlx/bin/python kitchen/brain_server.py > logs/brain_server_rehearse.log 2>&1 &
BP=$!
until curl -s -o /dev/null http://127.0.0.1:8765/health; do sleep 3; kill -0 $BP 2>/dev/null || { say "頭のサーバが落ちた"; exit 1; }; done
.venv/bin/python kitchen/demo_rehearse.py --spec data/demo_rehearse.json --out outputs/demo/v3_rehearse.json > logs/demo_rehearse.log 2>&1 || { say "下見が落ちた"; exit 1; }
say "$(tail -1 logs/demo_rehearse.log)"
pkill -f "kitchen/brain_server.py"; sleep 2
pick() { .venv/bin/python -c "import json,sys; print(json.load(open('outputs/demo/v3_rehearse.json'))['chosen'][sys.argv[1]] or '')" $1; }
O1=$(pick oneshot1); O2=$(pick oneshot2); A1=$(pick askback1); A2=$(pick askback2)
if [ -n "$O1" ] && [ -n "$O2" ]; then
  zsh offline_demo.sh v3_oneshot models/lite/c1o.npz "$O1" "$O2" > logs/demo_v3_oneshot.out 2>&1
  say "一度で決まる回 $(tail -4 logs/demo_v3_oneshot.out | tr '\n' ' ' | cut -c1-300)"
else
  say "一度で決まる回の発話が選べない(下見の表を見直す)"
fi
if [ -n "$A1" ] && [ -n "$A2" ]; then
  zsh offline_demo.sh v3_askback models/lite/c1o.npz "$A1" "$A2" > logs/demo_v3_askback.out 2>&1
  say "聞き返す回 $(tail -4 logs/demo_v3_askback.out | tr '\n' ' ' | cut -c1-300)"
else
  say "聞き返す回の発話が選べない(下見の表を見直す)"
fi
say "DEMO_REC_DONE"
