#!/bin/zsh
# kitchen/oft_session.py の中で走る(鍵とサーバは用意済み、PAI_MAC_LOCK は空)。先に初期状態 0 の 10 試行で動作確認してから残りを測る
cd ~/dev/pai-final; source env.sh; export HF_HUB_OFFLINE=1
ALL=(0 1 2 3 4 5 6 7 8 9)
.venv/bin/python kitchen/fast_rollout.py --teacher oft --skills $ALL --inits 0 --n-envs 4 --name "OFT の動作確認(初期状態 0)" --out outputs/fast/val_all_oft.jsonl
n=$(grep -c '"ok": true' outputs/fast/val_all_oft.jsonl)
[ $n -ge 6 ] || { echo "★ 動作確認 $n/10 で止める"; exit 3; }
.venv/bin/python kitchen/fast_rollout.py --teacher oft --skills $ALL --inits 0-9 --n-envs 4 --name "先生の候補: OpenVLA-OFT(公開、LIBERO-Goal)" --out outputs/fast/val_all_oft.jsonl
