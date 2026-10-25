#!/bin/zsh
# 「漏れたら聞き直す」を、規則を思いついたのとは別の音声(口の設定違い)で確かめる
cd ~/dev/pai-final; source env.sh
say() { echo "$(date '+%m/%d %H:%M') $*"; }
until grep -q "VOICE_DONE" logs/queue_voice.out 2>/dev/null; do sleep 30; done
P=.venv-mlx/bin/python
for name in b4 s2 base; do
  for cfg in "nohot --no-hot" "fix --relisten"; do
    set -- ${=cfg}; tag=$1; shift
    $P kitchen/voice_bench.py stt --name $name --tag $tag $@ > logs/voice_stt_${name}_$tag.log 2>&1 && say "耳 $name/$tag $(tail -1 logs/voice_stt_${name}_$tag.log)"
  done
done
say "VOICE2_DONE"
