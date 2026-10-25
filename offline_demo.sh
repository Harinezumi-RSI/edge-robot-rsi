#!/bin/zsh
# Wi-Fi なしで動く確認: 外向きの通信を禁止した囲い(kitchen/offline.sb、127.0.0.1 だけ許す)の中で、頭のサーバ(頭・目・耳・口)と声のデモ(手)を動かす。
#   zsh offline_demo.sh <出力名> <手: smolvla か models/lite/xx.npz> <コーチの発話 1> [<発話 2> ...]
# 記録: logs/offline_<名>_check.txt(囲いの中から外へ出られないこと)・logs/offline_<名>_conns.txt(10 秒おきの通信先)
cd ~/dev/pai-final; source env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
NAME=$1; HAND=$2; shift 2
SB=kitchen/offline.sb
pkill -f "kitchen/brain_server.py"; sleep 2
sandbox-exec -f $SB .venv-mlx/bin/python kitchen/brain_server.py > logs/offline_${NAME}_brain.log 2>&1 &
BP=$!
until curl -s -o /dev/null http://127.0.0.1:8765/health; do sleep 2; kill -0 $BP 2>/dev/null || { echo "頭のサーバが落ちた"; exit 1; }; done
{ echo "囲いの中から huggingface.co: $(sandbox-exec -f $SB curl -s -m 5 -o /dev/null -w '%{http_code}' https://huggingface.co || echo 失敗)";
  echo "囲いの中から 127.0.0.1:8765: $(sandbox-exec -f $SB curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8765/health)"; } > logs/offline_${NAME}_check.txt
sandbox-exec -f $SB .venv/bin/python kitchen/demo.py --voice --seen --hand $HAND --utterance "$@" --out outputs/demo/$NAME > logs/offline_${NAME}_demo.log 2>&1 &
DP=$!
: > logs/offline_${NAME}_conns.txt
while kill -0 $DP 2>/dev/null; do lsof -nP -i -a -p $BP,$DP 2>/dev/null | awk 'NR>1{print $1, $9}' >> logs/offline_${NAME}_conns.txt; sleep 10; done
kill $BP
echo "== 囲い"; cat logs/offline_${NAME}_check.txt
echo "== 通信先(10 秒おきに数えた回数)"; awk '{print $2}' logs/offline_${NAME}_conns.txt | sed 's/->/ → /' | sort | uniq -c
echo "== デモ"; grep -v "robosuite\|objc\|Class AV\|info\]\|assets\|WARNING\|Loading\|torch_dtype" logs/offline_${NAME}_demo.log | tail -6
