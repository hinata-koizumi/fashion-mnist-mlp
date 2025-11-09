# 東大 深層学習講座 コンペティション

[English](README.md) | 日本語

## コンペティション結果
- **最終順位**: **11位**/1466人中
- **LBスコア**: **0.9204**

## 概要
MNISTのファッション版 (Fashion MNIST，クラス数10) を多層パーセプトロンによって分類．

Fashion MNISTの詳細については以下のリンクを参考にしてください．
Fashion MNIST: https://github.com/zalandoresearch/fashion-mnist

## ルール
- 訓練データはx_train， t_train，テストデータはx_testで与えられます．
- 予測ラベルは one_hot表現ではなく0~9のクラスラベル で表してください．
- 下のセルで指定されているx_train，t_train以外の学習データは使わないでください．
- 多層パーセプトロンのアルゴリズム部分はNumPyのみで実装してください． (sklearnやtensorflowなどは使用しないでください)．
- データの前処理部分でsklearnの関数を使う (例えば sklearn.model_selection.train_test_split) のは問題ありません．

## アプローチ

- データ前処理/分割
  - `x_train.npy`, `y_train.npy`, `x_test.npy` を読み込み、28×28の `float32` にフラット化（784次元）
  - 訓練データ末尾10%を検証に分割（固定シード、既定 `seed=42`）
  - 訓練分割の平均/標準偏差で全データを標準化（特徴ごと、標準偏差に1e-7を加算してゼロ除算回避）
  - ミニバッチは学習時のみシャッフル、検証/テストは順序固定

- 画像拡張（スケジュール付き）
  - 平行移動: 最大±2px（上下左右、`np.roll`で実装、境界は0埋め）
  - Cutout: 正方形（2〜4px）をランダム位置でマスク
  - ガウスノイズ: σ=0.02（`--enable_noise_aug` フラグ時のみ、確率0.2→0.1に減衰）
  - 適用確率スケジュール: 学習進捗<80%で平行移動p=0.7、Cutout p=0.20。80%以降は平行移動pを0.7→0.4、Cutout pを0.20→0.08に線形減少。90%以降はCutout停止
  - 拡張は標準化前の生画素に適用し、その後再標準化

- 特徴量
  - 生画素のみ: 28×28の生画素を784次元へフラット化（HOG等の外部特徴は未使用）
  - 標準化: 訓練分割で推定した特徴ごとの平均/標準偏差で標準化

- モデル（NumPy実装MLP）
  - 層構成: 入力784 → 隠れ層（既定 `--hidden_sizes` 512→256→128）→ 出力10
  - 活性化: 中間層はReLU、最終層はロジット（softmaxなし）
  - Dropout: 逆ドロップアウト（inverted dropout、訓練時のみ）。80%以降は `--final_dropout_target`（既定0.05）へ線形に低減
  - 初期化: He初期化（ReLU向け、`sqrt(2.0 / in_dim)`）
  - 最適化: AdamW（既定、`--adamw`）またはAdam（`--no-adamw`）。AdamW時はdecoupled weight decay、Adam時はL2正則化を勾配に加算
  - 勾配クリップ: グローバルノルム `--grad_clip`（既定5.0）
  - EMA: `--ema_decay`（既定0.9995）で移動平均。学習90%以降は一時的に0.9997に強化

- 学習と正則化
  - Optimizer: AdamW（`weight_decay` 既定1e-4、`beta1=0.9`, `beta2=0.999`, `epsilon=1e-8`）
  - 学習率: 総ステップの5%をウォームアップ後、Cosine decay（基準LR `--lr` 既定1e-3、最小学習率は基準LRの1e-2）
  - 損失: ソフトマックス交差エントロピー。ラベルスムージング（`--label_smoothing_eps` 既定0.03）は学習80%以降に0へ線形減衰
  - 正則化: AdamWのdecoupled weight decay（既定1e-4）。Adam選択時はL2項（`--l2` 既定1e-4）を勾配に追加
  - 勾配クリップ: グローバルノルム5.0
  - 早期終了: 検証精度が `--patience` 回（既定15）更新されなければ停止。最良エポックのパラメータを保持
  - SWA（任意）: 学習後半（`--swa_start_frac` 既定0.8以降、`--swa_interval` ごと）でEMA重みをスナップショットし、終了時に平均化してEMAに設定
  - ファインチューニング（任意）: 学習後に全訓練で `--ft_epochs`（既定2）エポック微調整（LRを0.2×、Dropoutを1/3、ラベルスムージング0）

- 検証・選択（Inference Selector）
  - 検証の推論: MC Dropout（n=30、温度 `--mc_temp` 既定1.0）で確率を安定化し精度を評価。5エポックごとに混同行列、クラス別Precision/Recall/F1、校正指標（ECE・エントロピー・マージン）を記録
  - 候補戦略: `det`（EMA重みでの決定論推論）、`mc@T`（温度T∈{0.75,0.80,0.85,0.90,0.95,1.00}とdropout override∈{None,0.15,0.12,0.10,0.075}を探索）、`tta5mc@T`（5方向でロジット平均しつつMC）、`tta9mc@T`（9方向でロジット平均しつつMC）、`tta5_center`（5方向の確率平均、中心に重み2.0、MCなし）
  - 選択方法: 検証精度最大の候補を採用。差が僅少（3e-4以内）なときは上位2/3候補のブレンド（α∈{0.2..0.8}）を探索。`--enable_adaptive_blend` 時は適応的にαを最適化
  - マルチシード: `--multi_seed` で複数seedを順次実行し、各seedの検証精度に基づく重み付きアンサンブル（精度^`--ensemble_weighted_power`）も対応

- TTA と温度スケーリング
  - TTA: シフトのみ（回転/反転なし）。5方向（中心+上下左右）または9方向（中心+8近傍）。EMAを1回だけswapして全方向をまとめて推論し高速化。ロジット平均（9方向）または確率平均（5方向・中心重み）
  - 温度スケーリング（任意）: `--enable_temp_scale` 時、選ばれた候補の検証確率に対しT∈{0.85,0.9,0.95,1.0,1.05,1.1}をグリッド探索し、テスト確率に適用

- MC Dropout（自動）
  - 推論時もdropoutを有効化し、複数サンプル（既定30）の平均確率を使用。検証で温度とdropout overrideを選択。TTA×MCと組み合わせ可能

- 推論/保存
  - 選択された戦略の確率からクラスを決定し、`data/output/y_pred.csv` に `label` ヘッダで保存
  - モデル保存: `data/train/mlp_model.npz` として重み/バイアス、EMA、オプティマイザ状態、学習設定、標準化のmean/std、メタ情報（最良検証精度や戦略名）を同梱
  - オプション: `--save_test_proba` でテスト確率を `*.npy` 保存し、サイドカーの `*.meta.json` に選択戦略・検証精度を出力。`--ensemble_from` で複数確率の（重み付き）平均とCSV生成に対応

## 使用技術

- Python 3
- NumPy（`numpy`）
- Python標準ライブラリ（`argparse`, `json`, `os`, `sys`, `subprocess`, `typing`）

