#!/bin/zsh
# 軽い手の土台くらべ: 同じデータ(先生 SmolVLA の成功 227 本)・同じ評価(10 動作 × 初期状態 0〜9)で 3 つの土台
#   l0     : Qwen3.5-0.8B の途中の層(言語モデルを通す)
#   l0v    : 同じモデルの視覚部分だけ(言語モデルを通さない)
#   l0dino : DINOv2-small(言語なしの自己教師あり)
cd ~/dev/pai-final; source env.sh
# 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ、env.sh が読む)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
until grep -q "^保存:" logs/lite_data_r0.log 2>/dev/null; do sleep 20; grep -q Traceback logs/lite_data_r0.log && { say "r0 のデータ作りが落ちた"; exit 1; }; done
pkill -f "kitchen/lite_server.py"; sleep 3
nohup .venv-mlx/bin/python kitchen/lite_server.py --port 8766 > logs/lite_server.log 2>&1 &
until curl -s -o /dev/null http://127.0.0.1:8766/health; do sleep 2; done
say "サーバを入れ替えた(/vfeat あり)"
mkdir -p models/lite outputs/lite
.venv/bin/python kitchen/lite_data.py --demos data/rsi/r0/demos --every 4 --backend dino --out data/lite/r0_dino.npz > logs/lite_data_r0_dino.log 2>&1 &
P1=$!
.venv/bin/python kitchen/lite_data.py --demos data/rsi/r0/demos --every 4 --backend qwen08v --out data/lite/r0_qwen08v.npz > logs/lite_data_r0_qwen08v.log 2>&1 &
P2=$!
T() { .venv/bin/python kitchen/lite_train.py --data $1 --out models/lite/$2.npz > logs/lite_train_$2.log 2>&1 && say "$(tail -1 logs/lite_train_$2.log)" || say "$2 の学習が落ちた"; }
E() { .venv/bin/python kitchen/lite_eval.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --head models/lite/$1.npz --name "軽い手 $1 の評価($2)" --out outputs/lite/$1/val.jsonl > logs/lite_eval_$1.log 2>&1; say "$1 $(tail -1 logs/lite_eval_$1.log)"; }
T data/lite/r0.npz l0
E l0 "Qwen 0.8B の途中の層、先生の成功 227 本"
wait $P1 $P2
T data/lite/r0_dino.npz l0dino
T data/lite/r0_qwen08v.npz l0v
E l0dino "DINOv2-small、言語なし"
E l0v "Qwen 0.8B の視覚部分だけ"
say "LITE_CMP_DONE"
