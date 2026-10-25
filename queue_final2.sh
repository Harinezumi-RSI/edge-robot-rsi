#!/bin/zsh
# 第 2 版の最終テスト(初期状態 40〜49、第 2 版の調整では一度も使っていない)。手を決めてから 1 回だけ回す。
#   zsh queue_final2.sh <手: models/lite/xx.npz か models/lite/route_xx.json>
# 1. 軽い手だけ(10 動作 × 初期状態 40〜49、ずれなし)  2. 続けて動く(狙いの動作の前に別の動作)  3. 家事ロボットの通し(発話は判断用と同じ、初期状態だけ 40〜49)
# 4. 比べるための頭(先生を替えた効き目・DAgger の効き目を学習外の初期状態で。手を選ぶのには使わない)
# 先生(OFT・SmolVLA)の 40〜49 はベンチ用に別の列で(手の結果を先に出す)。
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
room() { local d=$(df -g ~ | awk 'NR==2{print $4}'); local s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6/1024)}'); [ $d -ge 6 ] && [ $s -le 9 ]; }
HAND=$1
[ -f "$HAND" ] || { say "手が無い: $HAND"; exit 1; }
O=outputs/final2
mkdir -p $O
[ -f $O/hand.txt ] && { say "最終テストは 1 回だけ($O/hand.txt がある)"; exit 1; }
{ echo "$HAND"; shasum -a 256 $HAND; [[ $HAND == *.json ]] && for p in $(.venv/bin/python -c "import json,sys; print(' '.join(sorted(set(json.load(open(sys.argv[1])).values()))))" $HAND); do shasum -a 256 $p; done; } > $O/hand.txt
say "最終テストの手 $(head -1 $O/hand.txt)"
until room; do sleep 120; done
.venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 40-49 --head $HAND --name "最終テスト: 軽い手" --out $O/lite.jsonl > logs/final2_lite.log 2>&1 || say "軽い手が落ちた"
say "軽い手 $(grep -c '"ok": true' $O/lite.jsonl)/$(wc -l < $O/lite.jsonl)"
until room; do sleep 120; done
.venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 40-49 --chain --head $HAND --name "最終テスト: 軽い手(続けて動く)" --out $O/lite_chain.jsonl > logs/final2_chain.log 2>&1 || say "続けて動くが落ちた"
say "続けて動く $(grep -c '"ok": true' $O/lite_chain.jsonl)/$(wc -l < $O/lite_chain.jsonl)"
until room; do sleep 120; done
pkill -f "kitchen/brain_server.py"; sleep 2
nohup .venv-mlx/bin/python kitchen/brain_server.py > logs/brain_server_final2.log 2>&1 &
BP=$!
until curl -s -o /dev/null http://127.0.0.1:8765/health; do sleep 3; kill -0 $BP 2>/dev/null || { say "頭のサーバが落ちた"; exit 1; }; done
PAI_HISTORY=1 .venv/bin/python kitchen/e2e_session.py --hand $HAND --inits 40-49 --tag final2 > logs/e2e_final2.log 2>&1 || say "通しが落ちた"
pkill -f "kitchen/brain_server.py"; sleep 2
[ -f outputs/e2e/final2/summary.json ] && say "通し(final2)$(tr '\n' ' ' < outputs/e2e/final2/summary.json | cut -c1-400)"
# 4. 比べるための頭(選ぶのには使わない、報告書の主張を学習外の初期状態で確かめる): SmolVLA を写した l0f・その DAgger d1f / OFT を写した l0o・普段の場面の DAgger d1o
for H in l0f d1f l0o d1o; do
  [[ $HAND == models/lite/$H.npz ]] && continue
  until room; do sleep 120; done
  .venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 40-49 --head models/lite/$H.npz --name "最終テスト: 比べる頭 $H" --out $O/cmp_$H.jsonl > logs/final2_cmp_$H.log 2>&1 || say "比べる頭 $H が落ちた"
  say "比べる頭 $H $(grep -c '"ok": true' $O/cmp_$H.jsonl)/$(wc -l < $O/cmp_$H.jsonl)"
done
# 5. 続けて動く(変わった場面)の比べ: 写すだけの l0o と、普段の場面からの DAgger の d1o(判断用で 52・63、手の c1o は 63)
for H in l0o d1o; do
  [[ $HAND == models/lite/$H.npz ]] && continue
  until room; do sleep 120; done
  .venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 40-49 --chain --head models/lite/$H.npz --name "最終テスト: 比べる頭 $H(続けて動く)" --out $O/cmp_${H}_chain.jsonl > logs/final2_cmp_${H}_chain.log 2>&1 || say "比べる頭 $H(続けて動く)が落ちた"
  say "比べる頭 $H(続けて動く)$(grep -c '"ok": true' $O/cmp_${H}_chain.jsonl)/$(wc -l < $O/cmp_${H}_chain.jsonl)"
done
say "FINAL2_DONE"
