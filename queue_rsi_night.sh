#!/bin/zsh
# 夜の RSI(方針: RSI を継続): 新しい情報が入る練習を 2 通り。出発点は頭を選ぶ評価で一番良かった単独の頭。評価はすべて備え付けを固定した条件。
#   h1: 立て直しの RSI(先生なし)= ずらした姿勢 → 手続きでホームへ → 軽い手。改造 Jev が成功と見た試行だけで学び直す
#   c1: 変わった場面の DAgger = 別の動作をやり終えた後の場面から練習し、先生(SmolVLA)が答えをつける
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
room() { local d=$(df -g ~ | awk 'NR==2{print $4}'); local s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6/1024)}'); [ $d -ge 6 ] && [ $s -le 9 ]; }
until grep -q "BENCH_DONE" logs/queue_bench.out 2>/dev/null && grep -q "E2E2_DONE" logs/queue_e2e2.out 2>/dev/null; do sleep 120; done
P=$(grep -o "1 つの頭の最良 [A-Za-z0-9_]*" logs/lite_select.log | awk '{print $2}')
[ -z "$P" ] && P=d1f
case $P in
  l0dino) POOL=(data/lite/r0_dino.npz) ;;
  l0f*) POOL=(data/lite/r0f_dino.npz) ;;
  d1f*) POOL=(data/lite/r0f_dino.npz data/lite/d1f_dagger.npz) ;;
  d2) POOL=(data/lite/r0_dino.npz data/lite/d1_dagger.npz data/lite/d2_dagger.npz) ;;
  *) POOL=(data/lite/r0_dino.npz data/lite/d1_dagger.npz) ;;
esac
say "出発点 $P(学習データ $POOL)"
B() { .venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --head models/lite/$P.npz $@ > /dev/null 2>&1; }
until room; do sleep 120; done
B --sigma 0.1 --name "出発点 $P(σ=0.1)" --out outputs/lite/night_base/${P}_s10.jsonl
B --chain --name "出発点 $P(続けて動く)" --out outputs/lite/night_base/${P}_chain.jsonl
say "出発点 σ=0.1 $(grep -c '"ok": true' outputs/lite/night_base/${P}_s10.jsonl)/100・続けて $(grep -c '"ok": true' outputs/lite/night_base/${P}_chain.jsonl)/100"
until room; do sleep 120; done
.venv/bin/python kitchen/lite_rsi.py --tag h1 --mode self --prev models/lite/$P.npz --pool $POOL --sigma 0.1 --practice-home-first \
  --eval-sigma 0.1 --eval-also-sigma0 --eval-also-chain > logs/lite_rsi_h1.out 2>&1 || say "h1 が落ちた"
say "h1 $(tail -1 logs/lite_rsi_h1.out | cut -c1-500)"
until room; do sleep 120; done
.venv/bin/python kitchen/lite_rsi.py --tag c1 --mode dagger --dagger-sparse-ok --prev models/lite/$P.npz --pool $POOL --practice-chain \
  --eval-also-chain > logs/lite_rsi_c1.out 2>&1 || say "c1 が落ちた"
say "c1 $(tail -1 logs/lite_rsi_c1.out | cut -c1-500)"
say "RSI_NIGHT_DONE"
