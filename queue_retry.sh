#!/bin/zsh
# 本番で失敗に気づいたらやり直す(10/24 夜)。自己進化の列(queue_selfevo.sh)が終わってから:
#  1. 家事の通し(判断用の初期状態 0〜9、発話と頭の手順は c1o の 20/30 と同じ): c1o + やり直し 2 回(Jev が失敗と見た動作をホームから揺らぎ σ 0.06 で)
#  2. x1o ができていれば: x1o そのまま・x1o + やり直し 2 回
#  3. Long(libero_10)を先生 SmolVLA で(queue_suites.sh と同じ手順。夜中に)
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1
say() { echo "$(date '+%m/%d %H:%M') $*"; }
okc() { echo "$(grep -c '"ok": true' $1)/$(wc -l < $1 | tr -d ' ')"; }
room() { local d=$(df -g ~ | awk 'NR==2{print $4}'); local s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6/1024)}'); [ $d -ge 6 ] && [ $s -le 9 ]; }
while kill -0 41740 2>/dev/null; do sleep 60; done  # queue_selfevo.sh
e2e() {  # $1 = 記録の名前、$2 = 手、$3 以降 = 追加の引数
  local tag=$1 hand=$2; shift 2
  until room; do sleep 120; done
  pkill -f "kitchen/brain_server.py"; sleep 2
  nohup .venv-mlx/bin/python kitchen/brain_server.py > logs/brain_server_$tag.log 2>&1 &
  local bp=$!
  until curl -s -o /dev/null http://127.0.0.1:8765/health; do sleep 3; kill -0 $bp 2>/dev/null || { say "頭のサーバが落ちた($tag)"; return 1; }; done
  PAI_HISTORY=1 .venv/bin/python kitchen/e2e_session.py --hand $hand --inits 0-9 --tag $tag "$@" > logs/e2e_$tag.log 2>&1 || say "通し $tag が落ちた"
  pkill -f "kitchen/brain_server.py"; sleep 2
  [ -f outputs/e2e/$tag/summary.json ] && say "通し($tag)$(tr '\n' ' ' < outputs/e2e/$tag/summary.json | cut -c1-300)"
}
e2e dev4_c1o_r2 models/lite/c1o.npz --retries 2 --retry-explore 0.06
if [ -f models/lite/x1o.npz ]; then
  e2e dev4_x1o models/lite/x1o.npz
  e2e dev4_x1o_r2 models/lite/x1o.npz --retries 2 --retry-explore 0.06
fi
# 3. Long
SU=libero_10; O=outputs/suite/$SU; D=data/suite/$SU; M=models/suite/$SU
mkdir -p $O/smolvla_n5 $O/l0 $O/d1 $D $M
until room; do sleep 120; done
.venv/bin/python kitchen/suite_loop.py eval --suite $SU --inits 0-9 --teacher smolvla --teacher-steps 5 --out $O/smolvla_n5/val.jsonl > logs/suite_${SU}_smolvla.log 2>&1
say "$SU 先生 SmolVLA(5 step ずつ)$(okc $O/smolvla_n5/val.jsonl)"
[ -f $D/r0.npz ] || .venv/bin/python kitchen/suite_loop.py teach --suite $SU --inits 10-39 --out $D/r0.npz > logs/suite_${SU}_teach.log 2>&1 || { say "$SU 先生の練習が落ちた"; exit 1; }
say "$SU 先生の練習 $(tail -1 logs/suite_${SU}_teach.log | cut -c1-120)"
.venv/bin/python kitchen/lite_train.py --data $D/r0.npz --out $M/l0.npz > logs/suite_${SU}_train_l0.log 2>&1 || { say "$SU l0 の学習が落ちた"; exit 1; }
.venv/bin/python kitchen/suite_loop.py eval --suite $SU --inits 0-9 --head $M/l0.npz --out $O/l0/val.jsonl > logs/suite_${SU}_l0.log 2>&1
say "$SU l0(先生を写す)$(okc $O/l0/val.jsonl)"
[ -f $D/d1_dagger.npz ] || .venv/bin/python kitchen/suite_loop.py dagger --suite $SU --inits 10-39 --head $M/l0.npz --out $D/d1_dagger.npz > logs/suite_${SU}_dagger.log 2>&1 || { say "$SU DAgger が落ちた"; exit 1; }
.venv/bin/python kitchen/lite_train.py --data $D/r0.npz $D/d1_dagger.npz --out $M/d1.npz > logs/suite_${SU}_train_d1.log 2>&1 || { say "$SU d1 の学習が落ちた"; exit 1; }
.venv/bin/python kitchen/suite_loop.py eval --suite $SU --inits 0-9 --head $M/d1.npz --out $O/d1/val.jsonl > logs/suite_${SU}_d1.log 2>&1
say "$SU d1(DAgger)$(okc $O/d1/val.jsonl)"
say "RETRY_LONG_DONE"
