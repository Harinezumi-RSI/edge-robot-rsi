#!/bin/zsh
# 続けて動く評価の足りない分(10/24 朝)。c1o(変わった場面の DAgger)は続けて動く評価で 63/100 だが、l0o・d1o は測っていないので、
# 「変わった場面の練習」の効き目と「先生を OFT に替えた」効き目を分けられない。queue_demo2.sh が終わってから、同じ条件(判断用の初期状態 0〜9・--chain)で測る。
# 対応表(家事の通しで使う手)も同じ条件で測る。
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1  # 鍵 PAI_MAC_LOCK は env.local.sh(この Mac だけ)
say() { echo "$(date '+%m/%d %H:%M') $*"; }
room() { local d=$(df -g ~ | awk 'NR==2{print $4}'); local s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6/1024)}'); [ $d -ge 6 ] && [ $s -le 9 ]; }
while kill -0 86937 2>/dev/null; do sleep 60; done  # queue_demo2.sh の zsh
HEADS=(l0o d1o route_oft2)
if [ -f models/lite/route_oft3.json ] && ! cmp -s models/lite/route_oft2.json models/lite/route_oft3.json; then HEADS+=(route_oft3); fi  # 「&& ! cmp && 配列+=」は zsh で構文エラー
for H in $HEADS; do
  if [ -f models/lite/$H.npz ]; then P=models/lite/$H.npz; else P=models/lite/$H.json; fi
  o=outputs/lite/${H}_chain/val.jsonl
  [ -f $o ] && [ $(wc -l < $o) -ge 100 ] && continue
  until room; do sleep 120; done
  .venv/bin/python kitchen/lite_batch.py --skills 0 1 2 3 4 5 6 7 8 9 --inits 0-9 --chain --head $P \
    --name "軽い手の評価 $H(続けて動く)" --out $o > logs/chainval_$H.log 2>&1 || say "続けて動く評価 $H が落ちた"
  say "続けて動く $H $(grep -c '"ok": true' $o)/$(wc -l < $o)"
done
say "CHAINVAL_DONE"
