#!/bin/zsh
# 家庭用ロボットのデモの続き(10/24 朝)。queue_demo.sh の通しは robosuite の上限 1,000 step で落ちた(失敗が重なるとセッションが 1,000 step を超える。
# chain_env.py で上限を外した)。変わった場面の DAgger(c1o)は答えつけが 6,121 場面・約 3.2 時間に膨らんだので止めて、順番を入れ替える:
#  1. 家事ロボットの通し(対応表 route_oft2 = OFT から作った頭を含めて動作ごとに選んだ手)。落ちる前の E01・E02 の記録はそのまま使う
#  2. c1o: 練習と採点は済んでいるので使い回し、Jev が成功と見なかった試行は 16 step おきに答えをつける(3,279 場面・約 1.7 時間)
#  3. c1o の頭を選ぶ評価 → 対応表 route_oft3 → 通し(route_oft3 が route_oft2 と違えば)と c1o 単独
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
room() { local d=$(df -g ~ | awk 'NR==2{print $4}'); local s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6/1024)}'); [ $d -ge 6 ] && [ $s -le 9 ]; }
ALL=(0 1 2 3 4 5 6 7 8 9)
e2e() {  # $1 = 記録の名前、$2 = 手
  pkill -f "kitchen/brain_server.py"; sleep 2
  nohup .venv-mlx/bin/python kitchen/brain_server.py > logs/brain_server_$1.log 2>&1 &
  local bp=$!
  until curl -s -o /dev/null http://127.0.0.1:8765/health; do sleep 3; kill -0 $bp 2>/dev/null || { say "頭のサーバが落ちた($1)"; return 1; }; done
  PAI_HISTORY=1 .venv/bin/python kitchen/e2e_session.py --hand $2 --inits 0-9 --tag $1 > logs/e2e_$1.log 2>&1 || say "通し $1 が落ちた"
  pkill -f "kitchen/brain_server.py"; sleep 2
  [ -f outputs/e2e/$1/summary.json ] && say "通し($1)$(tr '\n' ' ' < outputs/e2e/$1/summary.json | cut -c1-400)"
}
until room; do sleep 120; done
e2e dev3_route models/lite/route_oft2.json
until room; do sleep 120; done
.venv/bin/python kitchen/lite_rsi.py --tag c1o --mode dagger --teacher oft --dagger-sparse-ok --practice-chain --prev models/lite/d1o.npz \
  --pool data/lite/r0o_dino.npz data/lite/d1o_dagger.npz --eval-also-chain --reuse-practice --dagger-every 16 > logs/lite_rsi_c1o.out 2>&1 || say "c1o が落ちた"
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
  say "候補 $H $(cat outputs/lite/sel/${H}_{s0,a,b}/val.jsonl | grep -c '"ok": true')/300"
  .venv/bin/python kitchen/lite_select.py --extra d1o d1o_s1 d1o_s2 c1o --out models/lite/route_oft3.json > logs/lite_select_oft3.log 2>&1
  say "$(tail -1 logs/lite_select_oft3.log | cut -c1-300)"
  cmp -s models/lite/route_oft2.json models/lite/route_oft3.json || e2e dev3_route3 models/lite/route_oft3.json
  e2e dev3_c1o models/lite/c1o.npz
fi
say "DEMO2_DONE"
