#!/bin/zsh
# 頭のアンサンブル(行動を平均): 学習のたびに動作ごとの当たり外れが大きく揺れる(引き出し 0〜10)のを、頭を何個か平均してならす。
# 頭は 6MB・学習 15 秒、画像の特徴は 1 回で全部の頭に使い回すので、推論はほとんど遅くならない。比べる相手 = d1 単独・同じ撮り方(77/100)
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
until grep -q "FEAT_DONE" logs/queue_feat.out 2>/dev/null; do sleep 30; done
for s in 1 2 3; do
  .venv/bin/python kitchen/lite_train.py --data data/lite/r0_dino.npz data/lite/d1_dagger.npz --seed $s --out models/lite/d1_s$s.npz > logs/lite_train_d1_s$s.log 2>&1 && say "$(tail -1 logs/lite_train_d1_s$s.log)"
done
E() { .venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --head $1 --n-envs 8 --lazy-camera \
       --name "軽い手 頭のアンサンブル $2" --out outputs/lite/ens_$2/val.jsonl > logs/lite_eval_ens_$2.log 2>&1; say "ens $2 $(tail -1 logs/lite_eval_ens_$2.log)"; }
E models/lite/d1.npz,models/lite/d1_s1.npz,models/lite/d1_s2.npz,models/lite/d1_s3.npz d1x4
E models/lite/d1.npz,models/lite/d1_s1.npz,models/lite/d1_s2.npz,models/lite/d1_s3.npz,models/lite/d1_dino8.npz d1x4_8
say "ENS_DONE"
