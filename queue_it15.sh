#!/bin/zsh
# 15 周目: 学習の時間(周の 5 割)を減らす。点検(loop_health.py)で次の的になった。速さの試験(IT13_DONE)の後に回す
cd ~/dev/pai-final; source env.sh
# 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ、env.sh が読む)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
until grep -q "IT13_DONE" logs/it13.out 2>/dev/null; do sleep 120; done
D=outputs/rsi/ris1/train_list.txt
for m in fp32 bf16; do A=""; [ $m = bf16 ] && A="--bf16"
  .venv/bin/python kitchen/train_vla.py --data $D --batch 8 --k 4 $A --bench-only 12 --out models/bench.safetensors 2>&1 | grep '"mode"' | sed "s/^/$m /" | tee -a logs/it15.out
done
E() { .venv/bin/python kitchen/fast_rollout.py --skills 1 7 4 2 --inits 0-9 --n-envs 8 --n-action-steps 5 --sigma 0.1 --delta $1 \
      --snap data/snaps/val_rsi --tag $2 --name "15 周目: $3" --out outputs/rsi/abl/$2.jsonl > logs/it15_$2.log 2>&1; say "$3 $(tail -1 logs/it15_$2.log)" | tee -a logs/it15.out; }
mkdir -p outputs/rsi/abl
.venv/bin/python kitchen/train_vla.py --data $D --steps 600 --batch 8 --k 4 --out models/rsi/abl_s600.safetensors > logs/it15_train_s600.log 2>&1
E models/rsi/abl_s600.safetensors abl_s600 "ris1 のデータで 600 step(k=4)の評価 σ=0.1"
.venv/bin/python kitchen/train_vla.py --data $D --steps 750 --batch 8 --k 8 --out models/rsi/abl_k8.safetensors > logs/it15_train_k8.log 2>&1
E models/rsi/abl_k8.safetensors abl_k8 "ris1 のデータで k=8・750 step の評価 σ=0.1"
say "IT15_DONE" | tee -a logs/it15.out
