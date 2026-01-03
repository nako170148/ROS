# Lip-Reading Vowel Classifier (Japanese 5 vowels)

PCカメラ映像からMediaPipe FaceLandmarkerで顔ランドマークを推定し、唇の形状から日本語5母音（`a/i/u/e/o`）をリアルタイムに推定します。

このコードの特徴は、固定閾値ではなく「キャリブレーションで得た個人専用プロファイル」に対する距離（ユークリッド距離）で母音を決める点です。距離・個人差の影響は 目尻間距離で正規化して抑えます。

---

## ファイル構成

- `mediapipe_lip_tracker.py`
  - カメラ入力
  - MediaPipe FaceLandmarker 推論
  - キャリブレーション（起動時）
  - 母音のリアルタイム表示
  - 発話バッファ → 多数決で1文字確定 → 履歴表示
- `vowel_classifier.py`
  - ランドマークから唇特徴量（`MouthFeatures`）を抽出
  - キャリブレーション済プロファイル管理
  - 距離に基づく母音分類
- `face_landmarker.task`
  - MediaPipe の FaceLandmarker モデル（同階層に配置が必要）

---

## 動作環境

- Windows
- Python 3.10+（手元では 3.13 で動作）
- 主要ライブラリ
  - `mediapipe`（Tasks API を使用）
  - `opencv-python`
  - `numpy`

---

## セットアップ

### 1) 依存ライブラリのインストール

例（pip）：

```bash
pip install mediapipe opencv-python numpy
```

※ バージョンの相性で動かない場合は `mediapipe==0.10.31` を試してください。

### 2) モデルファイルの配置

`mediapipe_lip_tracker.py` と同じフォルダに `face_landmarker.task` を置きます。

- ダウンロード元（Google 配布）:
  - https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

---

## 実行方法

```bash
python mediapipe_lip_tracker.py
```

起動すると自動的にキャリブレーションが始まります。

---

## キャリブレーション

起動直後に **あ→い→う→え→お** の順で、各 3 秒ずつ発声してもらい、その間の唇特徴量を平均して **個人専用の参照プロファイル**を作ります。

- キャリブレーション中の操作
  - `s`: その母音のキャリブレーションをスキップ（前のプロファイルを維持）
  - `q`: キャリブレーション中断

キャリブレーションが十分に取れない（顔が検出できない等）場合は、その母音だけ「前のプロファイル」を維持します。

---

## リアルタイム推定と1文字確定ロジック

画面上部に以下を表示します。

- `Buffer`: 発話中（口が開いている間）に推定された母音列
- `History`: 発話終了（口が閉じたと判定）で確定した母音の履歴

### 口を閉じている判定

`features.vertical`（口の開き：正規化済み）が閾値以下なら沈黙とみなします。

- `SILENCE_VERTICAL_THRESHOLD`（デフォルト `0.05`）

### チャタリング対策（1フレーム閉口で確定しない）

「口が閉じた」状態が **連続 5 フレーム**続いたら発話終了とみなします。

- `SILENCE_FRAMES_REQUIRED`（デフォルト `5`）

### 発話中バッファと多数決

- 口が開いている間：スムージング後の母音を `vowel_buffer` に追加
- 発話終了（連続閉口 5 フレーム）になった瞬間：
  - `vowel_buffer` の最頻出母音を 1 文字として確定し `confirmed_text` に追加
  - `vowel_buffer` をクリア

### 反応速度

母音推定の揺れを抑えるため、直近フレームの多数決でスムージングしています。

- `VOWEL_SMOOTHING_WINDOW`（デフォルト `4`）

値を小さくすると反応が速くなりますが、揺れが増えます。

---

## キー操作（実行中）

- `q`: 終了
- `c`: 再キャリブレーション
- `r`: `History`（確定済み文字列）と `Buffer` をクリア

---

## アルゴリズム概要

### 1) ランドマーク（Landmarks）

MediaPipe FaceLandmarker が顔の 468 点ランドマークを推定します（正規化座標）。

### 2) 特徴量（MouthFeatures）

`vowel_classifier.py` で以下を計算します。

- `vertical`: 上唇中央 (13) と下唇中央 (14) の距離
- `horizontal`: 口角 (61, 291) の距離
- `ratio`: `vertical/horizontal`
- `area`: 外唇輪郭の多角形面積（シューレース公式）

### 3) 正規化（距離・スケール補正）

目尻 (33, 263) の距離を `eye_distance` とし、

- `vertical_n = vertical/eye_distance`
- `horizontal_n = horizontal/eye_distance`
- `area_n = area/eye_distance^2`

としてスケール変化を抑えます。

### 4) キャリブレーションと距離分類

各母音のキャリブレーション中に得た特徴量の平均を「参照プロファイル」とし、推定時は

\[ d = \sqrt{\Delta v^2 + \Delta h^2 + \Delta r^2 + \Delta a^2} \]

の距離が最小の母音を出力します。

---

## トラブルシューティング

- **起動時に Warning が出る**
  - `inference_feedback_manager requires a model` 等の警告が出る場合がありますが、動作自体に致命的でないことがあります。
- **キャリブレーションで samples が少ない**
  - 顔がフレーム内に入っているか、明るさ、カメラ解像度を確認してください。
- **モデルが見つからない**
  - `face_landmarker.task` が `mediapipe_lip_tracker.py` と同じフォルダにあるか確認してください。

---

## 調整パラメータ

`mediapipe_lip_tracker.py` 冒頭の定数を変更します。

- `SILENCE_VERTICAL_THRESHOLD`
- `SILENCE_FRAMES_REQUIRED`
- `VOWEL_SMOOTHING_WINDOW`

---
