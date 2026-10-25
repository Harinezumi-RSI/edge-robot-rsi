#!/bin/zsh
# DAgger 2 周目が終わって GPU が空いたら OFT の速さを測り(Mac の鍵を取り、ほかの重い処理なし)、止めていた待ち行列を再開する
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
until grep -q "LITE_LOOP2_DONE" logs/queue_lite_loop2.out 2>/dev/null; do sleep 30; done
until ! ps -ax -o command | grep -q "[f]ast_rollout\|[l]ite_batch\|[l]ite_dagger\|[t]rain_vla"; do sleep 30; done
say "OFT を測る(ほかの重い処理なし)"
.venv/bin/python - <<'PY' > logs/bench_oft.log 2>&1
import subprocess, sys
sys.path.insert(0, ".")
from kitchen.bench import mac_lock
with mac_lock():
    import os
    subprocess.run([os.environ.get("PAI_OFT_PY", ".venv-oft/bin/python"), "kitchen/bench_oft.py",
                    os.environ.get("PAI_OFT_CONFIG", "models/oft_goal")], check=False)  # 作りと大きさ(config)だけ読み、重みは乱数
PY
say "OFT $(tail -1 logs/bench_oft.log | cut -c1-300)"
nohup zsh queue_lite_loop3.sh > logs/queue_lite_loop3.out 2>&1 &
nohup zsh queue_feat.sh > logs/queue_feat.out 2>&1 &
nohup zsh queue_bench.sh > logs/queue_bench.out 2>&1 &
say "待ち行列を再開(lite_loop3・feat・bench)"
