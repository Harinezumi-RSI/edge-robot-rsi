#!/bin/zsh
# 聞き返す回を撮る(10/24)。下見(outputs/demo/v3_rehearse.json)で、あいまいな頼み 3 つは聞き返さずに「器を棚の上」に決めた(確認の確率 0.38〜0.59)。
# 「器を一番上の引き出しにしまって」は頭が引き出しを棚と読み違え(手順 [4])、確認の関門が聞き返した(0.068)。言い直し「一番上の引き出しにしまってね」は [3]。
# 聞き返す仕組みが本当に働く場面なので、この 2 つで 1 回撮る(下見の表の枠の外から選んだことは記録と動画の説明に書く)。
cd ~/dev/pai-final; source env.sh
say() { echo "$(date '+%m/%d %H:%M') $*"; }
LOCK=$PAI_MAC_LOCK  # 鍵(env.local.sh)。空なら取らない
ME="paifinal-demo $$"
if [ -n "$LOCK" ]; then
  until mkdir $LOCK 2>/dev/null; do sleep 30; done
  echo "$ME" > $LOCK/owner
  release() { [ "$(cat $LOCK/owner 2>/dev/null)" = "$ME" ] && { rm -f $LOCK/owner; rmdir $LOCK; }; }
  trap release EXIT; trap 'release; exit 1' INT TERM
fi
zsh offline_demo.sh v3_askback models/lite/c1o.npz "器を一番上の引き出しにしまって" "一番上の引き出しにしまってね" > logs/demo_v3_askback.out 2>&1
say "聞き返す回 $(tail -4 logs/demo_v3_askback.out | tr '\n' ' ' | cut -c1-400)"
say "ASKBACK_DONE"
