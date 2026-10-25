#!/bin/zsh
# 耳と口の改造(余裕枠): 往復テスト。口の設定ごとに声を作り、耳の設定ごとに聞き取って誤り率
cd ~/dev/pai-final; source env.sh
say() { echo "$(date '+%m/%d %H:%M') $*"; }
room() { local d=$(df -g ~ | awk 'NR==2{print $4}'); local s=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6/1024)}'); [ $d -ge 4 ] && [ $s -le 9 ]; }
P=.venv-mlx/bin/python
for cfg in "base --steps 4 --bits 8" "s2 --steps 2 --bits 8" "s1 --steps 1 --bits 8" "b4 --steps 4 --bits 4"; do
  set -- ${=cfg}; name=$1; shift
  until room; do say "空きを待つ(ディスク 4GB 以上・スワップ 9GB 以下)"; sleep 120; done
  $P kitchen/voice_bench.py tts --name $name $@ > logs/voice_tts_$name.log 2>&1 && say "口 $name $(tail -1 logs/voice_tts_$name.log)" || say "口 $name が落ちた"
  $P kitchen/voice_bench.py stt --name $name --tag hot > logs/voice_stt_${name}_hot.log 2>&1 && say "耳 $name $(tail -1 logs/voice_stt_${name}_hot.log)"
done
# 耳の改造は元の声で: ヒントなし・音を読む部分を 12・9 層で打ち切る
for cfg in "nohot --no-hot" "e12 --enc-layers 12" "e9 --enc-layers 9" "e15 --enc-layers 15"; do
  set -- ${=cfg}; tag=$1; shift
  $P kitchen/voice_bench.py stt --name base --tag $tag $@ > logs/voice_stt_base_$tag.log 2>&1 && say "耳 base/$tag $(tail -1 logs/voice_stt_base_$tag.log)" || say "耳 $tag が落ちた"
done
say "VOICE_DONE"
