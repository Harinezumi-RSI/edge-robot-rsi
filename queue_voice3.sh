#!/bin/zsh
# 耳の改造の続き(書き起こしの上限 96 トークン): 音を読む部分の打ち切り 15・12・9 層、漏れたら聞き直すを別の音声でも
cd ~/dev/pai-final; source env.sh
say() { echo "$(date '+%m/%d %H:%M') $*"; }
P=.venv-mlx/bin/python
for cfg in "e15 --enc-layers 15" "e12 --enc-layers 12" "e9 --enc-layers 9"; do
  set -- ${=cfg}; tag=$1; shift
  $P kitchen/voice_bench.py stt --name base --tag $tag $@ > logs/voice_stt_base_$tag.log 2>&1 && say "耳 base/$tag $(tail -1 logs/voice_stt_base_$tag.log)" || say "耳 $tag が落ちた"
done
for name in b4 s2 base; do
  for cfg in "nohot --no-hot" "fix --relisten"; do
    set -- ${=cfg}; tag=$1; shift
    [ $name = base ] && [ $tag = nohot ] && continue
    $P kitchen/voice_bench.py stt --name $name --tag $tag $@ > logs/voice_stt_${name}_$tag.log 2>&1 && say "耳 $name/$tag $(tail -1 logs/voice_stt_${name}_$tag.log)"
  done
done
say "VOICE3_DONE"
