#!/bin/zsh
# 動作ごとに頭を選ぶ: 候補の頭を、判断用の初期状態 0〜9 × (ずれなし + σ=0.02 のずれ 2 通り) = 動作ごとに 30 試行で比べ、動作ごとに一番良い頭の対応表を作る。
# 学習のたびに動作が丸ごと振れる(27 周目)のを、頭を選んで避ける。数字は最終テスト(40〜49)で 1 回だけ。
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
until grep -q "CHAIN_DONE" logs/queue_chain.out 2>/dev/null; do sleep 60; done
for s in 1 2; do
  .venv/bin/python kitchen/lite_train.py --data data/lite/r0f_dino.npz --seed $s --out models/lite/l0f_s$s.npz > logs/train_l0f_s$s.log 2>&1
  [ -f data/lite/d1f_dagger.npz ] && .venv/bin/python kitchen/lite_train.py --data data/lite/r0f_dino.npz data/lite/d1f_dagger.npz --seed $s --out models/lite/d1f_s$s.npz > logs/train_d1f_s$s.log 2>&1
done
C=(l0dino l0f l0f_s1 l0f_s2 d1 d1_s1 d1_s2 d1_s3 d2 d1f d1f_s1 d1f_s2)
for h in $C; do
  [ -f models/lite/$h.npz ] || continue
  for v in "s0 0 0" "a 0.02 1000" "b 0.02 2000"; do
    set -- ${=v}
    o=outputs/lite/sel/${h}_$1/val.jsonl
    [ -f $o ] && [ $(wc -l < $o) -ge 100 ] && continue
    .venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --head models/lite/$h.npz --sigma $2 --seed-offset $3 \
      --name "頭を選ぶ評価 $h($1)" --out $o > logs/sel_${h}_$1.log 2>&1
  done
  say "候補 $h $(cat outputs/lite/sel/${h}_*/val.jsonl | grep -c '"ok": true')/300"
done
.venv/bin/python kitchen/lite_select.py > logs/lite_select.log 2>&1; say "$(tail -3 logs/lite_select.log | tr '\n' ' ' | cut -c1-400)"
say "SELECT_DONE"
