#!/bin/zsh
# 10/22 の順番(速さを極める): ホームの試験の完了待ち → 手の推論時間の内訳(静かな GPU で) → 改造ごとの成功率(種を固定した対応比較)
# → 改造 Jev の学習データ集め(速い設定で)
cd ~/dev/pai-final; source env.sh
# 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ、env.sh が読む)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
until [ "$(grep -c 'init [0-9]:' logs/bench_home_test_b.log 2>/dev/null)" -ge 3 ] || grep -q Traceback logs/bench_home_test_a.log logs/bench_home_test_b.log 2>/dev/null; do sleep 30; done
say "ホームの試験が終わった"
.venv/bin/python kitchen/profile_vla.py > logs/profile_vla.log 2>&1; grep "ms/推論\|失敗" logs/profile_vla.log | sed 's/^/   /'
run() {  # run <名前> <bench の引数...>
  local name=$1; shift
  say "改造の比較: $name"
  .venv/bin/python kitchen/bench.py --plans 1 7 3 --inits 0-2 --out outputs/bench/speed_$name.jsonl "$@" > logs/bench_speed_$name.log 2>&1
  grep "init" logs/bench_speed_$name.log | grep -v "task orders" | sed 's/^/   /'
}
run base                                  # 毎 step 推論・反復 10・読み込んだままの精度
run n5 --n-action-steps 5
run n10 --n-action-steps 10
run n5_k3 --n-action-steps 5 --num-steps 3
run n5_fp16 --n-action-steps 5 --dtype fp16
say "改造 Jev の学習データ集め"
.venv/bin/python kitchen/collect_judge_data.py --skills 0-9 --inits 10-14 --every 10 --out data/judge > logs/collect_judge.log 2>&1
say "QUEUE_1022B_DONE"
