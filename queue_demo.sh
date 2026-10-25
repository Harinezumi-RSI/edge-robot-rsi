#!/bin/zsh
# 家庭用ロボットのデモを先に仕上げる(10/24 方針: デモの完成後に他の LIBERO へ)。OFT の一周(queue_oft.sh)の後に自動で:
#  1. 変わった場面に強くする: 別の動作をやり終えた後の場面から練習し、OFT が答えをつける DAgger(c1o)。出発点は OFT から作った頭の種 3 つで一番のもの
#  2. c1o も頭を選ぶ評価に足して、動作ごとの対応表を作り直す(route_oft2.json)
#  3. 家事ロボットの通し(判断用のセッション、会話の流れあり): 対応表の手 と c1o 単独
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
room() { local d=$(df -g ~ | awk 'NR==2{print $4}'); local s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6/1024)}'); [ $d -ge 6 ] && [ $s -le 9 ]; }
ALL=(0 1 2 3 4 5 6 7 8 9)
until grep -q "OFT_DONE" logs/queue_oft.out 2>/dev/null; do sleep 120; done
[ -f models/lite/route_oft.json ] || { say "OFT の一周が最後まで行っていないので止める"; say "DEMO_DONE"; exit 0; }
best=""; bn=-1
for h in d1o d1o_s1 d1o_s2; do
  k=$(cat outputs/lite/sel/${h}_*/val.jsonl 2>/dev/null | grep -c '"ok": true'); [ $k -gt $bn ] && { bn=$k; best=$h; }
done
say "出発点 $best($bn/300)"
until room; do sleep 120; done
.venv/bin/python kitchen/lite_rsi.py --tag c1o --mode dagger --teacher oft --dagger-sparse-ok --practice-chain --prev models/lite/$best.npz \
  --pool data/lite/r0o_dino.npz data/lite/d1o_dagger.npz --eval-also-chain > logs/lite_rsi_c1o.out 2>&1 || say "c1o が落ちた"
say "c1o $(tail -1 logs/lite_rsi_c1o.out | cut -c1-500)"
H=c1o
if [ -f models/lite/$H.npz ]; then
  for v in "s0 0 0" "a 0.02 1000" "b 0.02 2000"; do
    set -- ${=v}
    o=outputs/lite/sel/${H}_$1/val.jsonl
    [ -f $o ] && [ $(wc -l < $o) -ge 100 ] && continue
    until room; do sleep 120; done
    .venv/bin/python kitchen/lite_batch.py --skills $ALL --inits 0-9 --head models/lite/$H.npz --sigma $2 --seed-offset $3 \
      --name "頭を選ぶ評価 $H($1)" --out $o > logs/sel_${H}_$1.log 2>&1
  done
  say "候補 $H $(cat outputs/lite/sel/${H}_*/val.jsonl | grep -c '"ok": true')/300"
fi
.venv/bin/python kitchen/lite_select.py --extra d1o d1o_s1 d1o_s2 c1o --out models/lite/route_oft2.json > logs/lite_select_oft2.log 2>&1
say "$(tail -1 logs/lite_select_oft2.log | cut -c1-300)"
pkill -f "kitchen/brain_server.py"; sleep 2
nohup .venv-mlx/bin/python kitchen/brain_server.py > logs/brain_server_e2e3.log 2>&1 &
until curl -s -o /dev/null http://127.0.0.1:8765/health; do sleep 3; done
for spec in "route models/lite/route_oft2.json" "c1o models/lite/c1o.npz"; do
  set -- ${=spec}
  [ -f $2 ] || continue
  PAI_HISTORY=1 .venv/bin/python kitchen/e2e_session.py --hand $2 --inits 0-9 --tag dev3_$1 > logs/e2e_dev3_$1.log 2>&1 || say "通し $1 が落ちた"
  say "通し($1)$(tr '\n' ' ' < outputs/e2e/dev3_$1/summary.json | cut -c1-400)"
done
pkill -f "kitchen/brain_server.py"
say "DEMO_DONE"
