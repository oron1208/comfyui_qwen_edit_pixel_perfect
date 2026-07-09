# ComfyUI Qwen-Image-Edit Pixel-Perfect

Qwen-Image-Edit-2511 で「ピクセル位置パーフェクト」な編集を実現するための ComfyUI カスタムノード集です。

Qwen-Image-Edit は内部で画像を `1024×1024` 面積＋8の倍数アラインメントにリサイズするため、生成画像が元画像に対して数ピクセルの **スケール＋シフトのズレ** を生じます。また編集領域をピクセル単位で指定できません。本パッケージはこれらの問題を **2つのアプローチ** で解決します。

English summary below (日本語解説の続き).

---

## 🎯 解決する問題

| 問題 | 原因 |
|---|---|
| 生成線画が元画像からズレる | Qwen が内部で `1024×1024` 面積＋`round(.../8)*8` リサイズを行うため |
| マスク外も再生成されてしまう | コアノード `TextEncodeQwenImageEditPlus` に MASK 入力がないため |

## ✨ 含まれるノード（5つ）

### 🔧 アプローチA：事後補正（生成後にズレを直す）

| ノード | 役割 |
|---|---|
| **`Align to Reference (AKAZE) 🎯`** | AKAZE 特徴点マッチング＋RANSAC で、Qwen 出力を元画像に自動位置合わせする |

### 🛡️ アプローチB：事前予防（ズレが発生しないようにする）

| ノード | 役割 |
|---|---|
| **`Pre-Crop to Qwen (Drift-Free) ✂️`** | 元画像を Qwen の内部リサイズが「恒等変換（何もしない）」になるサイズにセンタークロップする |
| **`Upscale to Original Size 📐`** | Qwen 出力を元画像サイズに復元する |

### 🎨 マスクベース部分編集（補助）

| ノード | 役割 |
|---|---|
| **`Qwen Image Edit+ (Mask) 🔒`** | `TextEncodeQwenImageEditPlus` に MASK 入力を追加。マスクをVL画像に赤オーバーレイ＋プロンプトで位置を認識させる |
| **`Composite Masked (Hash-Perfect) 🔒`** | マスク外を元画像のピクセルで完全上書き。`torch.equal` レベルで完全一致を保証 |

## 📊 2つのアプローチ比較

| | 🔧 事後補正 (AKAZE) | 🛡️ 事前予防 (PreCrop) |
|---|---|---|
| 原理 | ズレてから特徴点で直す | ズレないサイズに最初からクロップ |
| 精度 | 特徴点数次第（高い） | **数学的にゼロ保証** |
| 画角 | **全画面維持** | クロップで端が欠ける |
| 計算コスト | 高い（AKAZE＋RANSAC） | 低い（クロップのみ） |
| 依存ライブラリ | OpenCV (`opencv-python`) | なし |

**選び方:**
- **画角を一切変えたくない** → 事後補正 (AKAZE)
- **ズレは絶対許容しない、端が欠けてもOK** → 事前予防 (PreCrop)
- **両方試したい** → 同梱のワークフロー例を両方読み込んで比較

## 📦 インストール

### ComfyUI-Manager 経由（推奨）
ComfyUI-Manager からこのリポジトリを検索してインストール。

### 手動インストール
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-account>/comfyui_qwen_edit_pixel_perfect.git
```
ComfyUI を再起動してください。

### 要件
- ComfyUI（`comfy_api.latest` の `io.Schema` 新形式ノード定義に対応したバージョン）
- **事後補正 (AKAZE) を使う場合のみ:** `opencv-python`
  ```bash
  pip install opencv-python
  ```
  （事前予防・マスク合成ノードは OpenCV 不要で動作します）

## 🚀 使い方

[`workflows/`](./workflows) フォルダに3つのサンプルワークフローを同梱しています。ComfyUI にドラッグ＆ドロップするだけで読み込めます。

### 1. 事後補正でズレを直す
- **`Q image Edit 2511 - Auto Align.json`**
- 元の Qwen Edit ワークフローに AKAZE 補正ノードを組み込んだ版。生成→自動補正→スライダー比較まで1回の実行で完結。

### 2. 事前予防でズレを防ぐ
- **`Q image Edit 2511 - Preventive (Drift-Free).json`**
- PreCrop でズレないサイズにクロップしてから生成、出力を元サイズに復元。ズレが数学的にゼロになります。

### 3. 線画ズレ補正（スタンドアロン）
- **`Qwen Edit 2511 - LineArt Align.json`**
- 2枚の LoadImage（元画像／Qwen線画）を読み込んで AKAZE 補正だけを行う独立ワークフロー。

### 4. マスクベース部分編集
- **`Q image Edit 2511 - PixelPerfect.json`**
- `Qwen Image Edit+ (Mask)` と `Composite Masked (Hash-Perfect)` を組み合わせ、マスク外を完全に維持した部分編集を行う版。

## 🔬 技術解説

### なぜズレるのか
Qwen-Image-Edit の `TextEncodeQwenImageEditPlus`（`comfy_extras/nodes_qwen.py:92-98`）は内部で以下の処理を行います：

```python
total = 1024 * 1024
scale_by = sqrt(total / (W * H))
width  = round(W * scale_by / 8.0) * 8   # 8の倍数に丸め
height = round(H * scale_by / 8.0) * 8
common_upscale(samples, width, height, "area", "disabled")
```

この `round() × 8` の丸めと面積ベースのリサイズが、元画像に対して微小なスケール歪みを生じさせます。画像全体に等方的にかかるため、アフィン/Similarity 変換としてモデル化・補正できます。

### 事前予防の数学
Qwen の内部リサイズ関数 `f(W,H) = (round(W·s/8)·8, round(H·s/8)·8)` の **不動点**（`f(W,H)=(W,H)` となる `(W,H)`）を探索し、そのサイズにクロップします。クロップ後の画像は Qwen に渡してもリサイズが恒等変換になり、ズレが発生しません。

### 事後補正の数学
AKAZE 特徴点検出 → Lowe の比率テストでマッチング → RANSAC で外れ値に強い Similarity 変換行列を推定 → ワープ適用。コンピュータビジョンの標準的な画像レジストレーション手法です。

## 📁 リポジトリ構成

```
comfyui_qwen_edit_pixel_perfect/
├── __init__.py          # ノード登録
├── nodes.py             # 5ノードの実装
├── workflows/           # サンプルワークフロー
│   ├── Q image Edit 2511 - Auto Align.json
│   ├── Q image Edit 2511 - Preventive (Drift-Free).json
│   ├── Qwen Edit 2511 - LineArt Align.json
│   └── Q image Edit 2511 - PixelPerfect.json
├── README.md
├── LICENSE
└── requirements.txt
```

## ⚠️ 注意事項

- コアファイル（`comfy_extras/nodes_qwen.py`）は **一切書き換えません**。ComfyUI の更新で本パッケージが消えることはありません。
- 事後補正 (AKAZE) は「画像全体が一様にズレている」場合に有効です。ポーズ変化や局所的な非剛体歪みには対応できません（その場合は画角維持とのトレードオフになります）。
- 事前予防 (PreCrop) はセンタークロップで画角が欠けます。`crop_anchor` 設定で欠ける方向を調整できます。
- VL 画像（384面積）経路のズレは事後補正で吸収しますが、事前予防では reference_latents 経路のみを完全にゼロにします（VL 経路の残差は微小です）。

## 📝 ライセンス

MIT License — 商用利用・改変・再配布すべて自由です。

---

## English Summary

A ComfyUI custom-node pack that enables **pixel-position-perfect editing** with Qwen-Image-Edit-2511.

Qwen-Image-Edit internally resizes input to a `1024×1024`-area grid with 8-pixel alignment, which introduces a few-pixel **scale + shift drift** between the generated image and the source. This pack solves it via **two complementary approaches**:

- **Corrective (AKAZE):** auto-align the Qwen output back onto the source using feature-point matching + RANSAC. Keeps the full frame; needs OpenCV.
- **Preventive (PreCrop):** pre-crop the source to a size that is a fixed point of Qwen's internal resize, so the resize becomes an identity and **zero drift** is mathematically guaranteed. No OpenCV needed, but the crop loses some of the frame.

Plus mask-based partial-editing nodes (`Qwen Image Edit+ (Mask)` + `Composite Masked (Hash-Perfect)`) that let you specify the edit region with a mask and guarantee the unmasked region is bit-identical to the source.

See the Japanese section above for node details, workflows, and the technical background.
