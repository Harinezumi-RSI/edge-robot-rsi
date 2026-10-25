#!/bin/zsh
# 1 周目の後: 規則で設定を決める → 練習 0 周目(全 10 動作 × 学習用 10〜39、改造 Jev の学習データ)→ 検証(全 10 動作 × 0〜9)
# → 改造 Jev の特徴・学習・検証(ゼロショットとの比較つき)。各段の時間は docs/rsi_runs.jsonl へ
cd ~/dev/pai-final; source env.sh
# 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ、env.sh が読む)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
until grep -q IT1_DONE logs/it1_n1.log 2>/dev/null; do sleep 30; done
NAS=$(.venv/bin/python - <<'PY'
import json
ok = lambda f: sum(json.loads(l)["ok"] for l in open(f))
n5, n1 = ok("outputs/fast/it1_n5.jsonl"), ok("outputs/fast/it1_n1.jsonl")
print(5 if n5 >= n1 - 3 else 1)   # 規則(先に固定): 5 step の成功が基準より 3 本以上少なくなければ 5 step を既定に
PY
)
say "1 周目の判定: n_action_steps=$NAS($(grep -c '"ok": true' outputs/fast/it1_n5.jsonl)/30 vs $(grep -c '"ok": true' outputs/fast/it1_n1.jsonl)/30)" | tee -a logs/it2.out
.venv/bin/python kitchen/fast_rollout.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 10-39 --n-envs 8 --n-action-steps $NAS \
  --snap data/snaps/train0 --record data/rsi/r0/demos --record-all --tag r0 --seed-offset 0 \
  --name "練習 0 周目(全 10 動作 × 学習用 10〜39、改造 Jev の学習データ)" --out outputs/rsi/r0/practice.jsonl > logs/r0_practice.log 2>&1
say "練習 0 周目: $(tail -1 logs/r0_practice.log)" | tee -a logs/it2.out
.venv/bin/python kitchen/fast_rollout.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --n-envs 8 --n-action-steps $NAS \
  --snap data/snaps/val_all --tag base --name "検証(全 10 動作 × 0〜9)" --out outputs/fast/val_all.jsonl > logs/val_all.log 2>&1
say "検証: $(tail -1 logs/val_all.log)" | tee -a logs/it2.out
S=$(date '+%Y-%m-%d %H:%M')
.venv-mlx/bin/python kitchen/jev.py from_runs data/judge_train0 outputs/rsi/r0/practice.jsonl data/snaps/train0 >> logs/jev.log 2>&1
.venv-mlx/bin/python kitchen/jev.py from_runs data/judge_val outputs/fast/val_all.jsonl data/snaps/val_all >> logs/jev.log 2>&1
.venv-mlx/bin/python kitchen/jev.py extract data/judge_train0 >> logs/jev.log 2>&1
.venv-mlx/bin/python kitchen/jev.py extract data/judge_val >> logs/jev.log 2>&1
.venv-mlx/bin/python kitchen/jev.py train data/judge_train0 >> logs/jev.log 2>&1
.venv/bin/python kitchen/runlog.py add judge "改造 Jev: 特徴の取り出し・学習(交差検証で層を選ぶ)" --start "$S" > /dev/null
S=$(date '+%Y-%m-%d %H:%M')
.venv-mlx/bin/python kitchen/jev.py evaltest data/judge_train0 data/judge_val --zero >> logs/jev.log 2>&1
.venv/bin/python kitchen/runlog.py add judge "改造 Jev の検証(ゼロショットとの比較)" --start "$S" > /dev/null
say "IT2_DONE" | tee -a logs/it2.out
