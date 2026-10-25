# ベンチ(make_bench.py が作る)

| 方策 | 測定環境 | 1 回の推論 | 実行する step | 1 step あたり | 1 秒に出せる行動 | LIBERO-Goal | 試行 | 大きさ | メモ |
|---|---|---|---|---|---|---|---|---|---|
| 軽量方策(本研究) | Mac mini GPU | 15.5ms | 4 | 3.9ms | 258 | 91% | 100 | DINOv2-small 22M + ヘッド 6MB | 学習 1 分以内・人手デモなし・教師 OFT |
| 軽量方策(GPU なし) | Mac mini CPU だけ | 25.9ms | 4 | 6.5ms | 154 | 91% | 100 | 同上 | 同一ヘッド |
| SmolVLA(5 step ごとに推論) | Mac mini GPU | 537.0ms | 5 | 107.4ms | 9 | 74% | 100 | 0.45B | 速度優先の設定 |
| SmolVLA(毎 step 推論、原論文の設定) | Mac mini GPU | 537.0ms | 1 | 537.0ms | 2 | 86% | 100 | 0.45B | 原論文と同じ設定 |
| OpenVLA-OFT(同じ Mac mini) | Mac mini GPU | 1805.8ms | 8 | 225.7ms | 4 | 97% | 100 | 7.7B | 公開の重み・16GB |
| OpenVLA-OFT | 論文(A100) | 112.0ms | 8 | 14.0ms | 71 | 98% | 500 | 7B | 表 I・II(手首カメラ・手先の状態あり) |
| OpenVLA | 論文(A100) | 239.6ms | 1 | 239.6ms | 4 | 79% | 500 | 7B | 表 I・II |
