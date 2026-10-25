#!/bin/zsh
# 軽い手のループ 1 周目を 2 通り(同じ出発点 l0dino = DINOv2-small + 頭、先生の成功 227 本)
#   d1: DAgger(練習で行った場面に先生 SmolVLA が正しい 8 step をつける)
#   s1: 自己改善(先生なし。改造 Jev が練習の画像を採点し τ=0.8 以上の自分の成功だけ学ぶ)
cd ~/dev/pai-final; source env.sh
# 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ、env.sh が読む)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
disk() { df -g ~ | awk 'NR==2{print $4}'; }
until grep -q "^合計\|Traceback" logs/lite_eval_l0v.log 2>/dev/null; do sleep 30; done
for t in d1 s1; do
  [ $(disk) -lt 3 ] && { say "ディスクの空きが 3GB 未満なので止める"; exit 1; }
  m=dagger; [ $t = s1 ] && m=self
  .venv/bin/python kitchen/lite_rsi.py --tag $t --mode $m --prev models/lite/l0dino.npz --pool data/lite/r0_dino.npz > logs/lite_rsi_$t.out 2>&1 || say "$t が落ちた"
  say "$t $(tail -1 logs/lite_rsi_$t.out | cut -c1-400)"
done
say "LITE_LOOP_DONE"
