#!/bin/zsh
# 軽い手の立て直しの RSI(先生なし): ずらした姿勢 → 手続きでホームへ戻す → 軽い手。改造 Jev が成功と見た試行だけで学び直す。
# 出発点は DAgger の 1・2 周目の良いほう。撮り方は手が見る step だけ(20 周目)。比べる相手 = 出発点のずれあり・なしの評価(同じ撮り方)。
cd ~/dev/pai-final; source env.sh
# 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ、env.sh が読む)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
room() { local d=$(df -g ~ | awk 'NR==2{print $4}'); local s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6/1024)}'); [ $d -ge 4 ] && [ $s -le 9 ]; }
until grep -q "LITE_LOOP2_DONE" logs/queue_lite_loop2.out 2>/dev/null; do sleep 60; done
n() { [ -f outputs/lite/$1/val.jsonl ] && grep -c '"ok": true' outputs/lite/$1/val.jsonl || echo 0; }
P=d1; POOL=(data/lite/r0_dino.npz data/lite/d1_dagger.npz)
if [ $(n d2) -ge $(n d1) ] && [ -f models/lite/d2.npz ]; then P=d2; POOL+=(data/lite/d2_dagger.npz); fi
say "出発点 $P(d1 $(n d1)・d2 $(n d2))"
B() { .venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --head models/lite/$P.npz --n-envs 8 --lazy-camera --sigma $1 \
       --name "軽い手 $P の評価(σ=$1、比べる相手)" --out outputs/lite/${P}_lazy_s$2/val.jsonl > logs/lite_eval_${P}_lazy_s$2.log 2>&1; say "$P σ=$1 $(tail -1 logs/lite_eval_${P}_lazy_s$2.log)"; }
until room; do sleep 120; done; B 0 0
until room; do sleep 120; done; B 0.1 10
# h1(立て直しの RSI)は 95% を超えてから(10/23 方針: 95% を超えてから比較する)
say "LITE_LOOP3_DONE"
