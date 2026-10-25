#!/bin/zsh
# l0o(OFT の練習を写すだけ、判断用 90/100)を動作ごとに頭を選ぶ候補に足す(10/24 朝)。ずれなしの条件(s0)は判断用の評価と同じ条件なので写した
# (今のコードで d1o・d1f は val と s0 が 100/100 一致)。残りの小さなずれ 2 通りだけ測る。lite_select.py の候補の並びに l0o を足してある。
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
room() { local d=$(df -g ~ | awk 'NR==2{print $4}'); local s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6/1024)}'); [ $d -ge 6 ] && [ $s -le 9 ]; }
ALL=(0 1 2 3 4 5 6 7 8 9)
H=l0o
for v in "a 0.02 1000" "b 0.02 2000"; do
  set -- ${=v}
  o=outputs/lite/sel/${H}_$1/val.jsonl
  [ -f $o ] && [ $(wc -l < $o) -ge 100 ] && continue
  until room; do sleep 120; done
  .venv/bin/python kitchen/lite_batch.py --skills $ALL --inits 0-9 --head models/lite/$H.npz --sigma $2 --seed-offset $3 \
    --name "頭を選ぶ評価 $H($1)" --out $o > logs/sel_${H}_$1.log 2>&1
done
say "候補 $H $(cat outputs/lite/sel/${H}_*/val.jsonl | grep -c '"ok": true')/300"
say "L0O_SEL_DONE"
