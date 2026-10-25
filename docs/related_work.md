# 関連研究(改造 Jev の位置づけ)

改造 Jev = 手元で動く小さな視覚言語モデル(Qwen3.5-4B、4bit)の途中の層の状態に、目標ごとの判定の頭(線形)を付けた速い判定器。
正解はシミュレータの目標判定(BDDL の述語)から自動で付ける。画像 1 回の読み込みで 10 個の目標の達成確率を出す。

## 1. Jev(TypeSafe の System One モデル)
- 2026-09-15 公開(早期アクセス)。文章を生成せず、状態(文字列/JSON)と型付きの問いを受け取り、確率つきの決定だけを返す。
  問いの型は 3 つ: Choice(選択肢から 1 つ)・Score(段階評価)・Noul(はい/いいえの確率)。出力は型で縛られるので形式の誤りが無いと主張。
- 速さ・値段(ベンダー発表): 1 件 0.07〜0.5 秒、入力 $0.042/100 万トークン。TypeSafe 自身の 4 業務の評価で正解率 67.8%(GPT-5.6 Terra 67.9%、Claude Opus 5 73.1%)を 0.4 秒で。
  中身は「新しい構造」「全出力を 1 回で出す並列サンプラ」「RLCD(較正された決定のための強化学習)」とだけ書かれ、微調整した LLM か判定の頭かは非公開。数字はベンダー自身の評価で、独立の再現は無い。
- 本作品との関係: Choice/Noul の形(確率つきの型付き判定)を借り、中身は公開モデル + 学習した頭で作る。
  出典: [DataCamp の解説](https://www.datacamp.com/blog/system-one-models-jev)、[eesel の解説](https://www.eesel.ai/blog/typesafe-jev)、[Towards AI](https://pub.towardsai.net/meet-jev-the-ai-model-that-doesnt-write-and-why-data-engineers-should-care-f3ebfa2aab55)

## 1b. 公開モデルで Jev を作る試み(2026-09〜)
- **DiffusionGemma を Jev にする vLLM の PR #57250**(Matt Mastracci、2026-09-19 ごろ、「We have Jev at home」): Google の拡散型 Gemma 4(diffusiongemma-26B-A4B、
  総 26B・1 トークンあたり 4B)で、問いごとの 1 トークンの答え欄をキャンバスに並べ、**1 回のノイズ除去で全部の答え欄を同時に埋め**、各欄の対数確率から選択と確信度を読む。
  1 回に最大 85 問。DGX Spark 1 台での非公式な比較で Jev とほぼ同点。作業中(マージ前)。出典: [解説](https://explainx.ai/blog/diffusiongemma-jev-vllm-open-source-2026)、
  [PR](https://github.com/vllm-project/vllm/pull/57250)
  → 「多くの問いを 1 回で」は本作品の改造 Jev(10 目標を 1 回の読み込みで)と同じ狙い。違いは、こちらは答え欄の生成でなく学習した頭で読むこと。
  なお MLX 版(mlx-community/diffusiongemma-26B-A4B-it-4bit)は 16.6GB で、本作品の「低ストレージ」の条件には収まらない。
- **OpenJev**(MIT、Python): 固定した Qwen3.5-4B に 1 回順伝播し、選択肢ごとのロジットを読む(学習なし)。RTX 3090 で確率 21 組が 1.0 秒(JSON を生成させると 5.3 秒)、
  TypeSafe の 102 件で Jev の最頻の答えとの一致 0.845(Jev 自身は 0.883、作者の報告)。→ **本作品のゼロショットの判定(選択肢の記号の確率)と同じ方式・同じ基盤モデル**。
- その他: mini-jev(固定した Qwen3-4B-Instruct の選択肢の記号のロジット + HTTP サーバ)、jevlike(自分のラベルで一から学習する小さな採点器)、
  MLX の並列の制約つきデコード(Apple Silicon で Qwen2.5-1.5B 4bit のスキーマの欄を並列に評価)。
  出典: [apidog のまとめ](https://apidog.com/blog/openjev-open-source-jev-alternatives/)、[regolo.ai](https://regolo.ai/jev-and-system-one-models-benchmarks-open-source-alternatives-and-when-to-use-them.md)、
  [TDS「自分の Jev を作る」(本文未読)](https://towardsdatascience.com/how-to-make-your-own-jev-model-from-an-open-llm/)

## 2. 言語モデルの「選択肢の確率」を判定に使う
- Kadavath ら 2022「Language Models (Mostly) Know What They Know」: 大きいモデルは、問いの形を整えれば多肢選択・真偽の答えのトークン確率がおおむね較正される。
  自分の答えが正しい確率 P(True) を出させる方法も提案。→ 本作品の「この手順は望みに合っているか」の確認(ゼロショット)はこの系統。
  出典: [arXiv 2207.05221](https://arxiv.org/abs/2207.05221v1)
- SayCan(Ahn ら 2022): 言語モデルが各スキルの確率を出し、価値関数(そのスキルが今の状態で成功する見込み)と掛けて選ぶ。終了を選ぶまで繰り返す。
  → 本作品の「次の一手を選ぶ判定」は SayCan の言語側に相当。4B では「終わり」を選べず余計な手を足した(SayCan が価値関数で絞る理由と同じ方向)。
  出典: [arXiv 2204.01691](https://arxiv.org/pdf/2204.01691)

## 3. 出来の確認(成功判定)をロボットの計画に戻す
- Inner Monologue(Huang ら 2022): 成功判定器・物体検出・人の言葉を計画の文脈に戻し、失敗ならやり直す。成功判定の偽陰性は余計なやり直しに、偽陽性は見落としになる。
  → 本作品の「確かめてやり直す」の土台。出典: [arXiv 2207.05608](https://arxiv.org/pdf/2207.05608)
- Du ら 2023「Vision-Language Models as Success Detectors」(SuccessVQA、CoLLAs 2023): 成功判定を VQA として Flamingo を微調整。言葉や見た目が変わる分布外で専用の報酬モデルを上回る。
  出典: [arXiv 2303.07280](https://arxiv.org/pdf/2303.07280)
- Luo(AAAI-24 学生枠): MiniGPT-4 を Bridge/UR5 の軌跡で微調整し、環境をまたぐ転移は分布のずれで難しいと報告。出典: [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/30552/32714)
- AHA(Duan ら、ICLR 2025): 失敗の検出と言葉での説明をする VLM。シミュレーションで成功デモを手続き的に壊して失敗データを作り(FailGen)、実ロボットにも汎化。
  下流の 3 つの枠組みで成功率を平均 21.4% 上げた(著者の報告)。出典: [arXiv 2410.00371](https://arxiv.org/pdf/2410.00371)
- **I-FailSense(Grislain ら 2025)**: VLM の**途中の層に軽い分類の頭(FS ブロック)を付け、アンサンブル**して失敗(指示とのずれ)を検出。同規模・より大きい VLM を上回り、
  他のシミュレーションや実機へ少ない追加学習で移るとする。→ **本作品に最も近い**(VLM の途中の層 + 軽い頭)。
  違いは、本作品が (a) Mac で速く動かすこと(4bit・1 回の読み込みで 10 目標・浅い層で打ち切り)、(b) 正解をシミュレータの述語から自動で付けること。
  出典: [arXiv 2509.16072](https://arxiv.org/pdf/2509.16072)
- SAFE(Gu ら、NeurIPS 2025): VLA(OpenVLA・π0・π0-FAST)の内部特徴で、失敗した試行が課題をまたいで 1 か所に集まる(failure zone)ことを見つけ、
  内部特徴から失敗の見込みを回帰。未知の課題でも働く。→ 判定器を VLA の内部に置く選択肢(本作品は判定を VLA と別のモデルに分けた)。
  出典: [arXiv 2506.09937](https://arxiv.org/pdf/2506.09937)
- VLM を報酬・価値に: VLM-RMs(Rocamonde ら、ICLR 2024、CLIP をゼロショットの報酬に)・GVL(Ma ら、ICLR 2025、並べ替えたフレームの順序づけで進み具合を推定)。
  出典: [arXiv 2310.12921](https://proceedings.iclr.cc/paper_files/paper/2024/hash/7a7f6cc5dc2a84fb4edf0feb8e5cfd50-Abstract-Conference.html)、[arXiv 2411.04549](https://arxiv.org/html/2411.04549v1)

## 4. 途中の層の状態に線形の判定器(プローブ)を付ける
- Azaria & Mitchell 2023「The Internal State of an LLM Knows When It's Lying」: 隠れ層の状態から発言の真偽を分類器で高精度に当てる。
- Burns ら 2023(CCS): 教師なしで「真偽の方向」を線形に見つける。
- Marks & Tegmark「The Geometry of Truth」(COLM 2024): 真偽は線形に表れるが、プローブが学習データで真偽と相関する別の特徴を拾う恐れ → **学習に使っていない分布で確かめる必要**。
  出典: [arXiv 2310.06824](https://arxiv.org/html/2310.06824v3)
- 2026 年の比較(arXiv 2606.02628): 量子化した 7〜8B モデルで、**真ん中あたりの 1 層**の線形プローブが幻覚を AUROC 0.904〜1.000 で検出(サンプリング型は 0.541 以下)。
  ピークは 32 ブロック中 13〜18。→ 4bit のモデルでも途中の層に判定に使える情報が残る裏づけ。出典: [arXiv 2606.02628](https://arxiv.org/pdf/2606.02628)
- 欺きの検出(arXiv 2508.19505): 層ごとのプローブ精度は「浅い層で偶然並み → 真ん中で最大 → 深い層で少し下がる」。出典: [arXiv 2508.19505](https://arxiv.org/html/2508.19505v2)
- 報酬モデル = 言語モデル + スカラーの頭(InstructGPT、Ouyang ら 2022、arXiv 2203.02155)も同じ「本体 + 小さな頭」の形。

## 5. どの層を使うか・どこで打ち切るか(速さ)
- Skean ら「Layer by Layer」(ICML 2025)・「Does Representation Matter?」(NeurIPS 2024): 下流の課題では途中の層の表現が最終層より良いことが多い。
  出典: [arXiv 2502.02013](https://arxiv.org/html/2502.02013v2)、[arXiv 2412.09563](https://arxiv.org/html/2412.09563v1)
- Gromov ら「The Unreasonable Ineffectiveness of the Deeper Layers」(ICLR 2025): 深い層は隣の層と似ていて、Llama-2-70B なら約半分の層を抜いても質問応答はほぼ保てる。
  → 判定に使う層より深い層は計算しない(打ち切り)ことで速くできる根拠。出典: [arXiv 2403.17887](https://arxiv.org/pdf/2403.17887)

## 6. 固定した VLM の途中の層から行動を出す(軽い手の位置づけ、2026-10-23 追加)
- **R3M(Nair ら 2022)の評価の型**: 固定した視覚表現 + 手先の状態 → 2 層の MLP で模倣学習。軽い手の頭も同じ形で、入力を VLM の途中の層にした。
  出典: [arXiv 2203.12601](https://arxiv.org/pdf/2203.12601)
- **Hansen ら(2022)「On Pre-Training for Visuo-Motor Control: Revisiting a Learning-from-Scratch Baseline」**: よく調整したゼロからの学習が、固定した事前学習の表現と互角以上のことが多い。
  → 固定表現は万能ではない、という注意。出典: [arXiv 2212.05749](https://arxiv.org/pdf/2212.05749)
- **VLM4VLA(2026-01)**: VLM が高い水準の理解のために学ぶ表現と、細かい制御に要る表現の間に「意味のずれ」がある。VLA の微調整で視覚部分を固定すると大きく落ちる。
  → 軽い手は画像の平均だけでなく 4×4 のマス(位置)も読む。出典: [arXiv 2601.03309](https://arxiv.org/pdf/2601.03309)
- **M²-VLA(2026-04)**: VLM の複数の層の状態を行動の頭が読む(Mixture of Layers)。途中の層を読む発想は同じで、頭は多層の Transformer。
  出典: [arXiv 2604.24182](https://arxiv.org/pdf/2604.24182)
- **小さな VLA(どれも土台ごと学習する)**:
  - CoTinyVLA(2026-07): 同じ **Qwen3.5-0.8B** を土台に約 0.9B。LIBERO-Plus で 2 エポック微調整し、LIBERO 平均 97.5%・LIBERO-Plus Goal 86.6%、推論のメモリ 2.25 GiB。
    出典: [arXiv 2607.25487](https://huggingface.co/papers/2607.25487)
  - VLA-Adapter: Qwen2.5-0.5B + 行動の方策で LIBERO 97.3%(公開の説明による)。学習に GPU 10GB 以上。出典: [Hugging Face](https://huggingface.co/VLA-Adapter/LIBERO-Spatial)
  - MiniVLA(Stanford、約 1B)・TurboVLA(言語部分を BERT にして約 0.2B)。出典: [MiniVLA](https://ai.stanford.edu/blog/minivla)、[TurboVLA の紹介](https://neurohive.io/en/state-of-the-art/turbovla-robot-gets-commands-several-times-faster-at-a-97-success-rate-thanks-to-swapping-the-llm-for-bert/)
- **OpenVLA-OFT(Kim ら 2025)**: OpenVLA(7B)を並列デコード・行動の塊・連続の行動で微調整し、LIBERO で行動の生成を 26 倍速く。
  表 II(LIBERO、NVIDIA A100、塊 8、100 回の平均): OpenVLA 4.2Hz・0.240 秒 → OFT 109.7Hz・0.073 秒(正面カメラのみの構成。手首カメラ・手先の状態ありでは 71.4Hz・0.112 秒)。
  軽い手は家庭用 Mac mini で 1 回 15.5ms・516Hz(GPU なしで 308Hz)。成功率は比べない(OFT は LIBERO 97%)。
  出典: [arXiv 2502.19645](https://arxiv.org/html/2502.19645v2)
- **DAgger(Ross ら 2011)**: 学ぶ側が自分で動いて行った場面に、先生が正しい行動をつけて足す。写すだけだと外れた場面を知らないまま、という問題への定番の答え。
  軽い手では先生が SmolVLA(方策から方策への蒸留)。

| 行動を出す手 | 土台 | 土台の学習 | 行動の頭 | 学習の機材 |
|---|---|---|---|---|
| CoTinyVLA | Qwen3.5-0.8B | する(2 エポック) | 専用の頭 | GPU |
| VLA-Adapter | Qwen2.5-0.5B | する | 橋渡しの注意 | GPU 10GB 以上 |
| M²-VLA | VLM | (要確認) | Transformer(層の混合) | GPU |
| **軽い手(本作品)** | Qwen3.5-0.8B 4bit | **しない(固定)** | **2 層の MLP(数 MB)** | **Mac の CPU で数分** |

狙いは成功率で土台ごと学習する手に勝つことではない。**頭だけなら数分で学び直せる**ので、自分で練習 → 自分で採点 → 学び直す周が速く回る、という点。

## 本作品の立ち位置(まとめ)
| | 判定の作り方 | 正解ラベル | 速さの工夫 |
|---|---|---|---|
| Jev | 非公開(独自構造・RLCD) | — | 1 回で全出力 |
| DiffusionGemma の Jev(vLLM PR) | 拡散 LM の答え欄を 1 回のノイズ除去で埋める | なし(ゼロショット) | 最大 85 問を 1 回で(26B-A4B・GPU) |
| OpenJev / mini-jev | 固定した 4B の選択肢のロジット | なし(ゼロショット) | 生成せず 1 回の順伝播 |
| SuccessVQA / AHA | VLM を微調整して文章で答える | 人手 / 手続きで壊した失敗 | — |
| I-FailSense | VLM の途中の層 + 軽い頭のアンサンブル | 既存データから作成 | — |
| SAFE | VLA の内部特徴 + 回帰 | 成功/失敗の試行 | 早期検出 |
| **改造 Jev(本作品)** | 4bit の小さな VLM の途中の層 + 目標ごとの線形の頭 | **シミュレータの述語で自動** | **1 回で 10 目標・浅い層で打ち切り・Mac** |

評価の注意(4. の Marks & Tegmark): 同じ試行の似た画像が学習と評価にまたがらないよう、初期状態ごとに分けて交差検証する。
