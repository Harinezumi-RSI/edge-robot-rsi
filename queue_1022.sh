#!/bin/zsh
# 10/22 の順番: ホームへ戻す試験の完了を待つ → 手の高速化(1 回の推論で何 step 実行するか)の成功率 → 改造 Jev の学習データ集め
cd ~/dev/pai-final; source env.sh
# 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ、env.sh が読む)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
until [ "$(grep -c 'init [0-9]:' logs/bench_home_test_b.log 2>/dev/null)" -ge 3 ] || grep -q Traceback logs/bench_home_test_a.log logs/bench_home_test_b.log 2>/dev/null; do sleep 30; done
say "ホームの試験が終わった"
for n in 5 10; do
  say "高速化: n_action_steps=$n"
  .venv/bin/python kitchen/bench.py --plans 1 7 3 --inits 0-2 --n-action-steps $n --out outputs/bench/speed_n$n.jsonl > logs/bench_speed_n$n.log 2>&1
  grep "init" logs/bench_speed_n$n.log | grep -v "task orders" | sed 's/^/   /'
done
say "改造 Jev の学習データ集め"
.venv/bin/python kitchen/collect_judge_data.py --skills 0-9 --inits 10-14 --every 10 --out data/judge > logs/collect_judge.log 2>&1
say "QUEUE_1022_DONE $(grep -c 'init' logs/collect_judge.log) 本"
