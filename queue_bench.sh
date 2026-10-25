#!/bin/zsh
# ベンチ: SmolVLA を論文と同じ作法(1 step ごとに推論し直す、n_action_steps=1)で、軽い手と同じ 100 試行(初期状態 0〜9 × 10 動作)
# 論文(SmolVLA 0.45B)の LIBERO-Goal は 92%(1 動作 10 回)。こちらの 5 step ずつの作法では 72%。
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
room() { local d=$(df -g ~ | awk 'NR==2{print $4}'); local s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6/1024)}'); [ $d -ge 4 ] && [ $s -le 9 ]; }
# 先生を強くできるかが 95% の分かれ目なので、頭の選択と並行ですぐ走らせる(10/23 19:10)
until room; do say "空きを待つ"; sleep 120; done
.venv/bin/python kitchen/fast_rollout.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --n-envs 8 --n-action-steps 1 \
  --name "ベンチ: SmolVLA を論文と同じ作法(毎 step 推論)" --out outputs/fast/val_all_n1.jsonl > logs/bench_smolvla_n1.log 2>&1
say "SmolVLA n1 $(grep '合計' logs/bench_smolvla_n1.log | tail -1)"
say "BENCH_DONE"
