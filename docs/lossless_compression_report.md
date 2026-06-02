# データの可逆圧縮 調査資料

調査日: 2026-05-21
対象: 可逆 (Lossless) データ圧縮 ― 理論的限界、現実的手法、「不可能」と言われた逸話、最新動向

---

## 目次

1. 可逆圧縮とは何か
2. 理論的限界 ― 「数学的に不可能」の正体
3. 古典的アルゴリズムの系譜
4. 現代の主要アルゴリズム (Zstd / Brotli / ANS など)
5. 「不可能」を称した有名な事件・主張
6. 機械学習 × 圧縮 ― 2023〜2026 の革命
7. 圧縮ベンチマークと記録 (Hutter Prize ほか)
8. 専門領域の可逆圧縮 (ゲノム・LLM 重み・量子)
9. 「圧縮=知能」仮説
10. まとめと展望
11. 参考文献

---

## 1. 可逆圧縮とは何か

可逆 (lossless) 圧縮は、圧縮された出力から元のビット列を **完全に復元** できるアルゴリズムの総称。ZIP, gzip, PNG, FLAC などが代表例で、テキスト、ソースコード、医療画像、ゲノムデータなど「1 ビットの誤りも許されない」分野で必須。

非可逆 (lossy) 圧縮 (JPEG, MP3, H.264 等) は人間の知覚限界を利用して情報を「捨てる」ため、より高い圧縮率を実現できるが、可逆圧縮は出力サイズに数学的な下限が存在する。

---

## 2. 理論的限界 ― 「数学的に不可能」の正体

### 2.1 シャノンの情報源符号化定理 (1948)

クロード・シャノンは、確率的情報源から出力される記号列の平均符号長 $L$ について、

$$L \geq H(X)$$

が成り立つことを示した ($H(X)$ はソースのシャノンエントロピー)。すなわち **どんな可逆圧縮も、エントロピー未満には平均的に圧縮できない**。逆に算術符号化やレンジ符号化を使えば任意の精度でエントロピーに漸近できる。

### 2.2 コルモゴロフ複雑性 (1965)

A. N. コルモゴロフは、個別の文字列 $x$ に対して「$x$ を出力する最短プログラムの長さ」を $K(x)$ と定義。これは **個別系列に対する圧縮の究極の下限** だが、**計算不可能 (uncomputable)** であることが証明されている。

| 指標 | 対象 | 計算可能か |
|---|---|---|
| シャノンエントロピー | 確率分布 | 可 |
| コルモゴロフ複雑性 | 個別の系列 | 不可 |

### 2.3 鳩の巣原理による「万能圧縮」の不可能性

$n$ ビットの可能な入力は $2^n$ 通り存在するが、$n$ ビット未満の出力は $2^{n-1}$ 通り以下。よって鳩の巣原理により **「すべての入力を必ず縮める可逆アルゴリズム」は存在しない**。圧縮できる入力があれば、必ず「膨張する入力」も存在する。

これは「再帰的に圧縮し続ければ任意のサイズに圧縮できる」という素朴な誤解を完全に否定する根拠でもある。

---

## 3. 古典的アルゴリズムの系譜

| 年 | アルゴリズム | 発案者 | 特徴 |
|---|---|---|---|
| 1952 | ハフマン符号 | D. Huffman | 記号ごとに最適な可変長プレフィックス符号 |
| 1976 | LZ77 | Ziv & Lempel | スライディングウィンドウによる辞書圧縮 |
| 1978 | LZ78 | Ziv & Lempel | 動的辞書構築。LZW (1984) の基盤 |
| 1979 | 算術符号化 | Rissanen 他 | 1 記号 1 ビット未満も表現可能。エントロピーに極めて近い |
| 1994 | BWT (Burrows-Wheeler) | Burrows & Wheeler | ブロック並べ替え。bzip2 の核 |
| 2002〜 | PAQ ファミリ | Matt Mahoney | 文脈混合 (context mixing) で最高峰の圧縮率 |
| 2009 | ANS (非対称数体系) | Jarek Duda | 算術符号の圧縮率と Huffman の速度を両立 |

---

## 4. 現代の主要アルゴリズム

### 4.1 Zstandard (Zstd) ― Facebook, 2016

LZ77 + Huffman/FSE (ANS の派生) + 大きな辞書。圧縮率と速度のバランスが極めて良く、2024 年に Cloudflare がサポートを追加してから急速に普及。

- Brotli より最大 42 % 高速で、ほぼ同等の圧縮率
- レベル 1〜22 まで調整可能

### 4.2 Brotli ― Google, 2015

WOFF2 と HTTP コンテンツ圧縮向け。静的辞書 (英語語彙) を内蔵し、Web テキストでは Zstd を上回る圧縮率を達成。

### 4.3 非対称数体系 (ANS) ― Jarek Duda, 2009

Jagiellonian 大学の Jarek Duda が発表。**「算術符号の圧縮率と Huffman の速度を同時に実現する」** という長年の懸案を解いた。
- 256 シンボルのアルファベットで Huffman の約 1.5 倍速い復号
- Apple, Google, Facebook, Dropbox, Microsoft, Pixar が採用
- Zstd, LZFSE, JPEG XL の中核

### 4.4 PAQ / ZPAQ ― Matt Mahoney, 2002 / 2009

500 以上の予測モデルを混合して算術符号化。ベンチマーク最強だが速度は遅い。Hutter Prize 受賞作の多くは PAQ 系の派生。

---

## 5. 「不可能」を称した有名な事件・主張

### 5.1 Sloot Digital Coding System (1995, オランダ)

電子技師 Romke Jan Sloot が **「映画 1 本を 1 KB に圧縮できる」** と主張。Philips の元 CEO Roel Pieper や Tom Perkins ら投資家を説得し、113,000 ユーロ相当の契約直前に **心臓発作で急死**、ソースコードは永遠に失われた。

- シャノンの定理に明確に違反するため数学的に不可能
- 後年の分析: 実機にはハードディスクが内蔵されており、実態は **共有辞書圧縮の亜種** だった可能性
- 圧縮詐欺の代表例として今も語り継がれる

### 5.2 Mark Nelson の「100 万乱数チャレンジ」

データ圧縮の権威 Mark Nelson が 100 ドルの賞金をかけて挑戦状を提示: 「RAND Corp. の真性乱数ファイルを、復号器込みで 1 バイトでも小さく圧縮せよ」。
- これまで誰も達成していない
- Patrick Craig がファイルを「区切り文字で分割して保存しない」というメタトリックで小さくしたが、ルール違反と判定された

### 5.3 普及している誤解

- 「圧縮ファイルをもう一度圧縮すれば永遠に小さくできる」 → 鳩の巣原理で不可能
- 「ランダムデータも圧縮できる新方式」 → 真のランダムは定義上 $K(x) \approx |x|$ なので不可能
- 「映画を SHA-256 ハッシュで保存して復元」 → ハッシュは衝突するため復元不可能

### 5.4 思考実験: 「一本の棒に印をつけて全データを表現する」

情報理論の古典的な思考実験で、近年 TikTok や Wolfram Community でも繰り返し話題になる:

> 「a=001, b=002, c=003 ... と各文字に番号を割り当て、文章を巨大な整数 $N$ にする。その先頭に小数点を打って $0.N$ という有理数を作り、長さ 1 m の金属棒の **その位置に印を 1 本だけ** つければ、任意の本・映画・図書館全体を『棒に 1 つの印』として保存できる」

一見すると **情報量を無限に圧縮した** ように見えるが、これは可逆圧縮の限界を破ったわけではない。重要なのは:

**(1) 情報は「印」ではなく「印の精度」に格納されている**
$L$ ビットのデータを符号化するには、棒の長さを $2^L$ 等分できる解像度が必要。すなわち印の位置を **$L$ ビットの精度で測定・刻印** しなければならず、必要な情報量は元データと完全に等しい。「圧縮」ではなく、ビット列を「位置の精度」という別のドメインに移しただけ。

**(2) これは算術符号化の物理版**
区間 $[0, 1)$ をデータの確率に応じて細かく分割し、最終区間内の任意の 1 点を選んで符号化する **算術符号化** (1979) は、まさにこの思考実験を数学的に厳密化したもの。出力は「1 つの実数」だが、その実数を表現するのに必要なビット数は ≈ シャノンエントロピーとなり、限界を破らない。

**(3) 物理的にも不可能**
- どんなに精密な測定器でも、**プランク長 (約 $1.6 \times 10^{-35}$ m)** より細かい位置は量子力学的に意味を持たない
- 1 m の棒で表現できる情報量は最大でも $\log_2(10^{35}) \approx 116$ ビット程度
- ハイゼンベルクの不確定性原理、熱ゆらぎ、原子の離散性によって実際の上限はさらに低い
- **ベケンシュタイン境界** によれば、有限の質量・サイズを持つ物理系に格納できる情報量には絶対的な上限がある

**(4) 教訓**
「ただ 1 つの印・1 つの実数」というレトリックは、情報が「どこか別の場所 (精度・桁数・物理測定器の解像度)」に隠れているだけだという典型例。Sloot 事件 (5.1) の **「1 KB のスマートカードに映画 1 本」** 主張も、突き詰めればこれと同じ「ビット数の見かけの隠蔽」に過ぎない。

この思考実験は、シャノン限界とコルモゴロフ複雑性が **どんなに巧妙な物理的トリックでも回避できない** ことを直感的に示す、最も美しい例の 1 つとされる。

---

## 6. 機械学習 × 圧縮 ― 2023〜2026 の革命

### 6.1 「Language Modeling Is Compression」 (DeepMind, 2023, ICLR 2024)

DeepMind の論文。LLM の次トークン予測確率を算術符号と組み合わせると、**Chinchilla 70B はテキスト以外のドメインでも汎用圧縮器を凌駕** することを示した。

| データ | Chinchilla 70B | 専用ベスト |
|---|---|---|
| ImageNet パッチ | **43.4 %** | PNG: 58.5 % |
| LibriSpeech 音声 | **16.4 %** | FLAC: 30.3 % |

注: ただしモデル本体 (数十 GB) を計上していない「予測器コスト」前提の評価。

### 6.2 LLM ベース圧縮の汎用化 (2025)

2025 年 5 月の TechXplore 報道では、**LLM を用いた汎用圧縮アルゴリズムが古典手法の少なくとも 2 倍の可逆圧縮率を達成** したと報告。テキスト・画像・音声・動画すべてで有効。

### 6.3 AlphaZip (2024)

Transformer ブロックで次トークンを予測 → 予測順位列を Adaptive Huffman / LZ77 / gzip で圧縮する 2 段方式。

### 6.4 ニューラルネット重みの可逆圧縮

LLM 自体が巨大化するなか、**モデルウェイトを可逆に縮小** する研究が活況:

| 手法 | 圧縮率 | 開発元 |
|---|---|---|
| **ZipNN** | 約 33〜50 % 削減、1.5 倍高速 | IBM Research |
| **DFloat11** (Dynamic-Length Float) | 30 % 削減、ビット完全一致 | 大学コンソーシアム |
| **Unweight** | MLP 重みの可逆圧縮 | Cloudflare Research (2026) |

---

## 7. 圧縮ベンチマークと記録

### 7.1 Hutter Prize (Marcus Hutter, 2006〜)

英語 Wikipedia の先頭 1 GB (enwik9) を **どれだけ小さく圧縮できるか** を競う賞金コンテスト。

- 賞金: 1 % 削減ごとに 5,000 ユーロ (最大 500,000 ユーロ)
- 制約: 単一 CPU コア、50 時間、10 GB RAM、100 GB HDD
- **現在の記録: 114,156,155 バイト (約 11.4 %, Saurabh Kumar)**
- 直近では Kaido Orav の "fx-cmix" が 1.38 % 改善

### 7.2 Large Text Compression Benchmark (Matt Mahoney)

ENWIK8/9 を中心とした圧縮率と CPU 時間のオープンランキング。PAQ 系・CMIX 系が常に上位。

---

## 8. 専門領域の可逆圧縮

### 8.1 ゲノム DNA 配列

- アルファベットが 4 (A/C/G/T) と小さいが分布は一様に近く、汎用圧縮 (gzip, Zstd) は効きにくい
- **CRAM** (BAM の後継): 参照ゲノム差分 + コンテキストモデル
- 参照ゲノムなしの場合は重み付きコンテキストモデル + ストキャスティック繰返しモデルを競合させる手法が研究されている
- Illumina/Enancio が商用化

### 8.2 量子データ圧縮 (2025〜)

- 量子ゲートは **本質的に可逆** ― 可逆圧縮との親和性が極めて高い
- フォン・ノイマンエントロピーが下限
- ICASSP 2025 で **Quantum Run-Length Encoding (QRLE)** が発表
- 重ね合わせ・もつれを利用して古典限界を超える圧縮を目指す研究が進行中

### 8.3 顕微鏡画像・科学データ

- ライフサイエンスでは 1 セッションで TB 級画像を生成
- bioRxiv 等で Zstd, Brotli, JPEG-LS, JPEG XL の科学画像での比較研究が進む

---

## 9. 「圧縮=知能」仮説

Marcus Hutter, Jürgen Schmidhuber らが提唱: **「より良く圧縮できることは、より良く未来を予測することと等価」**。

### 9.1 ソロモノフ帰納

任意の計算可能データ列の確率を、その列を生成する **最短プログラムの長さ** で重み付けする普遍事前分布。データが圧縮できる ⇔ 短いプログラムで生成できる ⇔ パターンを発見できている。

### 9.2 言語モデル ≒ 圧縮

LLM が次トークン分布 $P(x_t \mid x_{<t})$ を高精度で出すなら、算術符号でほぼ最適な可逆圧縮になる。すなわち **「LLM の損失 = 圧縮率の対数」** であり、Hutter Prize の存在意義 (圧縮 = AI 研究) と整合する。

### 9.3 「Compression Represents Intelligence Linearly」 (2024)

最新の研究では、複数の LLM について **圧縮率とベンチマーク性能がほぼ線形に相関** することが実証されており、「圧縮 = 知能」仮説の経験的支持となっている。

---

## 10. まとめと展望

| 観点 | 現状 (2026) |
|---|---|
| 理論的限界 | シャノン (確率源) / コルモゴロフ (個別系列) で確立済み。「全入力を縮める」万能圧縮は永遠に不可能 |
| 汎用古典圧縮 | Zstd / Brotli が業界標準。ANS が次世代エントロピー符号として定着 |
| 最強圧縮 | PAQ / CMIX 系がベンチマーク最強だが遅い |
| 新潮流 | **LLM を確率予測器として算術符号と組み合わせる** 方式が古典手法を 2 倍超える圧縮率を達成 |
| 専門領域 | ゲノム・モデル重み・量子データそれぞれで専用手法が急進展 |
| 哲学 | 「圧縮 = 予測 = 知能」が AI 時代に再評価され、Hutter Prize の意義が高まっている |

「夢の圧縮」を求めるアプローチは **シャノン限界の中で予測器を強化する** 方向に収束しており、近年の LLM 圧縮はその到達点とも言える。Sloot のような「物理を超える圧縮」は依然として詐欺か誤解だが、**ドメイン知識を組み込んだ予測モデル** で実用上の限界はまだ大きく更新の余地がある。

---

## 11. 参考文献

### 理論

- [Shannon's source coding theorem - Wikipedia](https://en.wikipedia.org/wiki/Shannon's_source_coding_theorem)
- [シャノンの情報源符号化定理 - Wikipedia](https://ja.wikipedia.org/wiki/%E3%82%B7%E3%83%A3%E3%83%8E%E3%83%B3%E3%81%AE%E6%83%85%E5%A0%B1%E6%BA%90%E7%AC%A6%E5%8F%B7%E5%8C%96%E5%AE%9A%E7%90%86)
- [Shannon Information and Kolmogorov Complexity (arXiv)](https://arxiv.org/pdf/cs/0410002)
- [Empirical Lossless Compression Bound of a Data Sequence (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12385675/)
- [Why guaranteed file compression is impossible - Matt Might](https://matt.might.net/articles/why-infinite-or-guaranteed-file-compression-is-impossible/)

### 古典・現代アルゴリズム

- [Asymmetric numeral systems - Wikipedia](https://en.wikipedia.org/wiki/Asymmetric_numeral_systems)
- [ANS: entropy coding combining speed of Huffman with compression rate of arithmetic coding (arXiv)](https://arxiv.org/abs/1311.2540)
- [The Compression Optimality of Asymmetric Numeral Systems (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10137965/)
- [PAQ - Wikipedia](https://en.wikipedia.org/wiki/PAQ)
- [The ZPAQ Compression Algorithm - Matt Mahoney](https://mattmahoney.net/dc/zpaq_compression.pdf)
- [ZSTD vs Brotli vs GZip Comparison - SpeedVitals](https://speedvitals.com/blog/zstd-vs-brotli-vs-gzip/)
- [Choosing Between gzip, Brotli and zStandard Compression - Paul Calvano](https://paulcalvano.com/2024-03-19-choosing-between-gzip-brotli-and-zstandard-compression/)

### 「不可能」事件

- [Sloot Digital Coding System - Wikipedia](https://en.wikipedia.org/wiki/Sloot_Digital_Coding_System)
- [The Man Who Was Paid €113,000 For His Code Which Compressed Entire Movies in 8KB - LowEndBox](https://lowendbox.com/blog/the-man-who-was-paid-e113000-for-his-code-which-compressed-entire-movies-in-8kb-of-disk-and-then-he-died/)
- [Malicious Life Podcast: Jan Sloot's Incredible Data Compression System - Cybereason](https://www.cybereason.com/blog/malicious-life-podcast-jan-sloots-incredible-data-compression-system)
- [The Million Random Digit Challenge Revisited - Mark Nelson](https://marknelson.us/posts/2006/06/20/million-digit-challenge.html)
- [The Random Compression Challenge Turns Ten - Mark Nelson](https://marknelson.us/posts/2012/10/09/the-random-compression-challenge-turns-ten.html)
- [Just another notch — David Bradley (棒に印で全文学を符号化する思考実験)](https://www.sciencebase.com/science-blog/just-another-notch.html)
- [A Book in a Notch: encoding a book into a single notch on a metal rod — Wolfram Community](https://community.wolfram.com/groups/-/m/t/2827787)
- [The light-second rod — mitxela.com (棒の物理的限界に関する思考実験)](https://mitxela.com/projects/the_light_second_rod)

### ML × 圧縮

- [Language Modeling Is Compression (arXiv)](https://arxiv.org/pdf/2309.10668)
- [google-deepmind/language_modeling_is_compression - GitHub](https://github.com/google-deepmind/language_modeling_is_compression)
- [DeepMind's Chinchilla AI toasts FLAC and PNG at lossless data compression - PC Gamer](https://www.pcgamer.com/deepminds-chinchilla-ai-toasts-flac-and-png-at-lossless-data-compression-despite-essentially-being-just-a-large-language-model/)
- [Algorithm based on LLMs doubles lossless data compression rates - TechXplore (2025-05)](https://techxplore.com/news/2025-05-algorithm-based-llms-lossless-compression.html)
- [AlphaZip: Neural Network-Enhanced Lossless Text Compression (arXiv)](https://arxiv.org/pdf/2409.15046)
- [Lossless compression tailored for AI (ZipNN) - IBM Research](https://research.ibm.com/blog/Zip-NN-AI-compression)
- [DFloat11: Lossless LLM Compression for Efficient GPU Inference - OpenReview](https://openreview.net/forum?id=xdNAVP7TGy)
- [Unweight: Lossless MLP Weight Compression for LLM Inference - Cloudflare Research](https://research.cloudflare.com/papers/unweight-2026.pdf)
- [Compression Represents Intelligence Linearly (arXiv)](https://arxiv.org/pdf/2404.09937)

### ベンチマーク

- [Hutter Prize - Wikipedia](https://en.wikipedia.org/wiki/Hutter_Prize)
- [Human Knowledge Compression Contest - Hutter Prize 公式](http://prize.hutter1.net/)
- [Large Text Compression Benchmark - Matt Mahoney](https://www.mattmahoney.net/dc/text.html)

### 専門領域

- [Lossless Genomic Data Compression - Illumina](https://www.illumina.com/science/technology/development/genomic-data-compression.html)
- [A Reference-Free Lossless Compression Algorithm for DNA Sequences (MDPI)](https://www.mdpi.com/1099-4300/21/11/1074)
- [Data Compression with Quantum Computers - WIN-SE / JKU](https://se.jku.at/data-compression-with-quantum-computers/)
- [Lossless Quantum Compression (arXiv)](https://arxiv.org/pdf/quant-ph/0508170)
- [Design and implementation of run-length encoding on quantum computers - ScienceDirect (ICASSP 2025)](https://www.sciencedirect.com/science/article/abs/pii/S1568494625015868)
