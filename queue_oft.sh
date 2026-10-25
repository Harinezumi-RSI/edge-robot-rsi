#!/bin/zsh
# 先生を公開の OpenVLA-OFT(LIBERO-Goal)にする(10/23 夜、方針: より強い公開の教師)。
# 1. 先生の腕前: 判断用の 100 試行(SmolVLA の 5 step ずつ 74・毎 step 86 と同じ試行)。初期状態 0 の 10 試行で先に動作確認
# 2. 85 以上なら: 先生の練習(初期状態 10〜39 × 10 動作)→ 写すだけの軽い手 l0o → DAgger 1 周(答えは OFT、Jev が成功と見た試行は間引く)
#    → 種 3 つ → 頭を選ぶ評価と同じ条件(ずれなし + σ=0.02 × 2)で SmolVLA が先生の d1f の種 3 つと比べ、動作ごとの対応表も作り直す
# OFT は約 16GB。同じ Mac のほかの評価も同じ鍵で OFT を載せるので、OFT を使う間は鍵を持ち続ける(kitchen/oft_session.py)。
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
room() { local d=$(df -g ~ | awk 'NR==2{print $4}'); local s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6/1024)}'); [ $d -ge 6 ] && [ $s -le 9 ]; }
ALL=(0 1 2 3 4 5 6 7 8 9)
until grep -q "SELECT_DONE" logs/queue_select.out 2>/dev/null; do sleep 60; done
until room; do sleep 120; done
.venv/bin/python kitchen/oft_session.py -- zsh oft_val_inner.sh > logs/oft_val.log 2>&1
N=$(grep -c '"ok": true' outputs/fast/val_all_oft.jsonl 2>/dev/null); M=$(wc -l < outputs/fast/val_all_oft.jsonl 2>/dev/null)
say "OFT 判断用 ${N:-0}/${M:-0} $(grep '合計' logs/oft_val.log | tail -1)"
[ ${M:-0} -ge 100 ] && [ ${N:-0} -ge 85 ] || { say "OFT が 100 試行そろわないか 85 未満なので止める"; say "OFT_DONE"; exit 0; }
mkdir -p data/rsi/r0o/demos
until room; do sleep 120; done
.venv/bin/python kitchen/oft_session.py -- .venv/bin/python kitchen/fast_rollout.py --teacher oft --skills $ALL --inits 10-39 --n-envs 4 \
  --record data/rsi/r0o/demos --record-all --kind practice --name "先生(OFT)の練習" --out outputs/rsi/r0o_practice.jsonl > logs/oft_r0o.log 2>&1
say "先生(OFT)の練習 $(grep '合計' logs/oft_r0o.log | tail -1)"
.venv/bin/python kitchen/lite_data.py --demos data/rsi/r0o/demos --every 4 --backend dino --out data/lite/r0o_dino.npz > logs/oft_r0o_data.log 2>&1 || { say "手本の特徴が作れない"; exit 1; }
.venv/bin/python kitchen/lite_train.py --data data/lite/r0o_dino.npz --out models/lite/l0o.npz > logs/train_l0o.log 2>&1
.venv/bin/python kitchen/lite_batch.py --skills $ALL --inits 0-9 --head models/lite/l0o.npz --name "l0o(OFT を写すだけ)" --out outputs/lite/l0o/val.jsonl > logs/l0o.log 2>&1
say "l0o $(tail -1 logs/l0o.log)"
until room; do sleep 120; done
.venv/bin/python kitchen/lite_rsi.py --tag d1o --mode dagger --teacher oft --dagger-sparse-ok --prev models/lite/l0o.npz --pool data/lite/r0o_dino.npz \
  > logs/lite_rsi_d1o.out 2>&1 || say "d1o が落ちた"
say "d1o $(tail -1 logs/lite_rsi_d1o.out | cut -c1-400)"
for s in 1 2; do
  .venv/bin/python kitchen/lite_train.py --data data/lite/r0o_dino.npz data/lite/d1o_dagger.npz --seed $s --out models/lite/d1o_s$s.npz > logs/train_d1o_s$s.log 2>&1
done
for h in d1o d1o_s1 d1o_s2; do
  [ -f models/lite/$h.npz ] || continue
  for v in "s0 0 0" "a 0.02 1000" "b 0.02 2000"; do
    set -- ${=v}
    o=outputs/lite/sel/${h}_$1/val.jsonl
    [ -f $o ] && [ $(wc -l < $o) -ge 100 ] && continue
    until room; do sleep 120; done
    .venv/bin/python kitchen/lite_batch.py --skills $ALL --inits 0-9 --head models/lite/$h.npz --sigma $2 --seed-offset $3 \
      --name "頭を選ぶ評価 $h($1)" --out $o > logs/sel_${h}_$1.log 2>&1
  done
  say "候補 $h $(cat outputs/lite/sel/${h}_*/val.jsonl | grep -c '"ok": true')/300"
done
.venv/bin/python kitchen/lite_select.py --extra d1o d1o_s1 d1o_s2 --out models/lite/route_oft.json > logs/lite_select_oft.log 2>&1
say "$(tail -1 logs/lite_select_oft.log | cut -c1-300)"
say "OFT_DONE"
