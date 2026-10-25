#!/bin/zsh
# 軽い手のループの続き: s1(自己改善)の後に、撮り方の比較の続き → DAgger 2 周目(d1 から、Jev が成功と見た試行は答えを間引く)
cd ~/dev/pai-final; source env.sh
# 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ、env.sh が読む)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
room() { local d=$(df -g ~ | awk 'NR==2{print $4}'); local s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6/1024)}'); [ $d -ge 4 ] && [ $s -le 9 ]; }
until grep -q "LITE_LOOP_DONE\|止める" logs/queue_lite_loop.out 2>/dev/null; do sleep 60; done
until room; do say "空きを待つ"; sleep 120; done
.venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --head models/lite/l0dino.npz --n-envs 8 --lazy-camera \
  --name "軽い手 l0dino をカメラ 4 回に 1 回で評価(続き)" --out outputs/lite/l0dino_lazy/val.jsonl >> logs/lite_eval_l0dino_lazy.log 2>&1
say "撮り方 $(tail -1 logs/lite_eval_l0dino_lazy.log)"
until room; do say "空きを待つ"; sleep 120; done
.venv/bin/python kitchen/lite_rsi.py --tag d2 --mode dagger --dagger-sparse-ok --prev models/lite/d1.npz --pool data/lite/r0_dino.npz data/lite/d1_dagger.npz > logs/lite_rsi_d2.out 2>&1 || say "d2 が落ちた"
say "d2 $(tail -1 logs/lite_rsi_d2.out | cut -c1-400)"
say "LITE_LOOP2_DONE"
