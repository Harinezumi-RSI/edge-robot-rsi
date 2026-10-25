#!/bin/zsh
# SmolVLA 毎 step のベンチの残り: 動作 0 の初期状態 0〜7 は備え付けを固定する前に走った行だった(8 行)。同じ固定した条件で測り直し、きれいな 100 試行を別のファイルに組む(元のファイルは残す)
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
.venv/bin/python kitchen/fast_rollout.py --skills 0 --inits 0-7 --n-envs 8 --n-action-steps 1 \
  --name "ベンチの残り: SmolVLA 毎 step・動作 0 の初期状態 0〜7(固定した条件)" --out outputs/fast/val_s0_n1_fix.jsonl > logs/bench_smolvla_n1_s0.log 2>&1
.venv/bin/python - <<'PY'
import json
A = [l for l in open("outputs/fast/val_all_n1.jsonl") if "fixtures" in json.loads(l)]
B = [l for l in open("outputs/fast/val_s0_n1_fix.jsonl")]
assert len(A) == 92 and len(B) == 8, (len(A), len(B))
open("outputs/fast/val_all_n1_fix.jsonl", "w").writelines(A + B)
print("きれいな 100 試行", sum(json.loads(l)["ok"] for l in A + B))
PY
say "N1FIX_DONE"
