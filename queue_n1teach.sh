#!/bin/zsh
# 毎 step の先生の手本を足す(10/23 夜に決めて、夜の RSI の後に回す)。
# 先生を毎 step 推論にすると同じ 92 試行で 70 → 79(15・6、p=0.078)。差が出た動作(0 引き出し・5 皿を押す・6 チーズ・9 ワインをラック)だけ、
# 毎 step の先生で練習の初期状態 10〜39 を回して手本を足す。学習のラベルは「実行した行動の続き 8 step」なので、閉じたループで直した動きを写せる。
# 軽い手そのものを毎 step 見直すと 77 → 62(p=0.0015、引き出しで止まる)なので、手は 4 step ずつのまま、手本の側だけ変える。
# 学習ごとに動作が丸ごと振れるので種 3 つで作り、頭を選ぶ評価と同じ条件(ずれなし + σ=0.02 × 2)で d1f の種 3 つと比べる。
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
room() { local d=$(df -g ~ | awk 'NR==2{print $4}'); local s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6/1024)}'); [ $d -ge 6 ] && [ $s -le 9 ]; }
until grep -q "RSI_NIGHT_DONE" logs/queue_rsi_night.out 2>/dev/null; do sleep 120; done
until room; do sleep 120; done
mkdir -p data/rsi/r0n1/demos
.venv/bin/python kitchen/fast_rollout.py --skills 0 5 6 9 --inits 10-39 --n-envs 8 --n-action-steps 1 --record data/rsi/r0n1/demos --record-all \
  --kind practice --name "毎 step の先生の練習(差が出た 4 動作)" --out outputs/rsi/r0n1_practice.jsonl > logs/n1t_practice.log 2>&1
say "毎 step の先生の練習 $(grep '合計' logs/n1t_practice.log | tail -1)"
.venv/bin/python kitchen/lite_data.py --demos data/rsi/r0n1/demos --every 4 --backend dino --out data/lite/r0n1_dino.npz > logs/n1t_data.log 2>&1 || { say "手本の特徴が作れない"; exit 1; }
for s in 0 1 2; do
  n=d1n; [ $s -gt 0 ] && n=d1n_s$s
  .venv/bin/python kitchen/lite_train.py --data data/lite/r0f_dino.npz data/lite/r0n1_dino.npz data/lite/d1f_dagger.npz --seed $s --out models/lite/$n.npz > logs/train_$n.log 2>&1
done
for h in d1n d1n_s1 d1n_s2; do
  for v in "s0 0 0" "a 0.02 1000" "b 0.02 2000"; do
    set -- ${=v}
    o=outputs/lite/sel/${h}_$1/val.jsonl
    [ -f $o ] && [ $(wc -l < $o) -ge 100 ] && continue
    until room; do sleep 120; done
    .venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --head models/lite/$h.npz --sigma $2 --seed-offset $3 \
      --name "頭を選ぶ評価 $h($1)" --out $o > logs/sel_${h}_$1.log 2>&1
  done
  say "候補 $h $(cat outputs/lite/sel/${h}_*/val.jsonl | grep -c '"ok": true')/300"
done
.venv/bin/python kitchen/lite_select.py --extra d1n d1n_s1 d1n_s2 --out models/lite/route2.json > logs/lite_select2.log 2>&1; say "$(tail -1 logs/lite_select2.log | cut -c1-300)"
say "N1TEACH_DONE"
