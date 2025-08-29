# sabi8_movie3.py
# -*- coding: utf-8 -*-

import os
import io
import time
import json
import math
import tempfile
import numpy as np
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms, models
from matplotlib import rcParams, font_manager
from matplotlib.font_manager import FontProperties

# --- 動画用（OpenCVは任意。無い場合は動画モードを使えません） ---
try:
    import cv2
except Exception:
    cv2 = None

# =========================================================
# 透明favicon（自転車アイコン完全非表示）を用意（なければ生成）
# =========================================================
FAVICON = "transparent.png"
def ensure_transparent_favicon(path=FAVICON):
    if not os.path.exists(path):
        try:
            img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            img.save(path)
        except Exception:
            pass
ensure_transparent_favicon()

# =========================================================
# ページ設定（この1回だけにする）
# =========================================================
st.set_page_config(
    page_title="金属腐食診断システム",
    page_icon=FAVICON,
    layout="wide",
    menu_items={"Get Help": None, "Report a bug": None, "About": None}
)

# =========================================================
# 右上の自転車/ランナー（ステータスインジケータ）をこのアプリだけ消す
# =========================================================
st.markdown("""
<style>
[data-testid="stStatusWidget"] { display: none !important; }
header [data-testid="stStatusWidget"] { display: none !important; }
.block-container img, .block-container canvas, .block-container svg {
  max-width: 100% !important;
  height: auto !important;
}
.section-gap { margin-top: 24px; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 日本語フォント設定
# =========================================================
_JP_FONT_CANDIDATES = [
    "IPAexGothic", "IPAGothic",
    "Noto Sans CJK JP", "Noto Sans JP",
    "Hiragino Sans", "Hiragino Kaku Gothic ProN",
    "Yu Gothic", "Meiryo",
]
def set_japanese_font():
    found = None
    for fam in _JP_FONT_CANDIDATES:
        try:
            fp = FontProperties(family=fam)
            path = font_manager.findfont(fp, fallback_to_default=False)
            if path and os.path.exists(path):
                found = fam
                break
        except Exception:
            pass
    if found:
        rcParams["font.family"] = found
    rcParams["axes.unicode_minus"] = False
set_japanese_font()

st.title("金属腐食診断システム")

# =========================================================
# ユーティリティ
# =========================================================
MODEL_EXTS = (".pth", ".pt", ".bin")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

def fit_square_canvas(img: Image.Image, size_px: int, bg=(255, 255, 255), inner_ratio: float = 0.85):
    inner_ratio = max(0.1, min(1.0, float(inner_ratio)))
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    max_side = int(size_px * inner_ratio)
    scale = min(max_side / w, max_side / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    resized = img.resize((new_w, new_h), resample)
    canvas = Image.new("RGB", (size_px, size_px), color=bg)
    off_x = (size_px - new_w) // 2
    off_y = (size_px - new_h) // 2
    canvas.paste(resized, (off_x, off_y))
    return canvas

def draw_red_border(img: Image.Image, width: int = 4, color=(255, 0, 0)) -> Image.Image:
    bordered = img.copy()
    draw = ImageDraw.Draw(bordered)
    w, h = bordered.size
    draw.rectangle([(1, 1), (w - 2, h - 2)], outline=color, width=width)
    return bordered

def make_square_bar_figure(
    labels, values, size_px: int,
    title: str = "", red_border: bool = False,
    hatch_pattern: str | None = None, hatch_facecolor=None
):
    dpi = 100
    size_in = size_px / dpi
    fig, ax = plt.subplots(figsize=(size_in, size_in), dpi=dpi)
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title, fontsize=11, pad=6)
    for i, v in enumerate(values):
        ax.text(i, min(0.98, v + 0.03), f"{v:.2f}", ha='center', fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    if hatch_pattern is not None:
        for b in bars:
            if hatch_facecolor is not None:
                b.set_facecolor(hatch_facecolor)
            b.set_hatch(hatch_pattern)
    if red_border:
        for s in ax.spines.values():
            s.set_edgecolor("red")
            s.set_linewidth(3)
    plt.tight_layout()
    return fig

def walk_dirs_with_files(roots, want_exts, max_depth=2, limit=200):
    seen, out = set(), []
    cwd = os.path.abspath(".")
    for base in roots:
        if not os.path.isdir(base):
            continue
        base_abs = os.path.abspath(base)
        for dirpath, dirnames, filenames in os.walk(base_abs):
            rel = os.path.relpath(dirpath, base_abs)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > max_depth:
                dirnames[:] = []
                continue
            if any(str(f).lower().endswith(want_exts) for f in filenames):
                rel_to_cwd = os.path.relpath(dirpath, cwd)
                norm = os.path.normpath(rel_to_cwd)
                if norm not in seen:
                    seen.add(norm)
                    out.append(norm)
                    if len(out) >= limit:
                        return sorted(out)
    return sorted(out)

def list_files_with_ext(dirpath, exts):
    try:
        return sorted([f for f in os.listdir(dirpath) if f.lower().endswith(exts)])
    except Exception:
        return []

def format_time_label(seconds: float) -> str:
    if seconds < 0: seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - (h*3600 + m*60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:04.1f}"
    else:
        return f"{m:d}:{s:04.1f}"

def crop_center_horizontal_fraction(img: Image.Image, fraction: float = 1/3) -> Image.Image:
    f = max(0.05, min(1.0, float(fraction)))
    w, h = img.size
    crop_w = max(1, int(round(w * f)))
    left = max(0, int(round((w - crop_w) / 2)))
    right = min(w, left + crop_w)
    if right <= left:
        return img
    return img.crop((left, 0, right, h))

def _pil_to_cv_gray(pil_img: Image.Image, resize_max_w: int | None = 960):
    if cv2 is None:
        return None, None, None
    rgb = np.array(pil_img.convert("RGB"))
    h, w = rgb.shape[:2]
    scale = 1.0
    if resize_max_w is not None and w > resize_max_w:
        scale = resize_max_w / float(w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        rgb_small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        rgb_small = rgb
    gray = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY)
    return gray, rgb_small, scale

def detect_pole_line(pil_img: Image.Image,
                     angle_tol_deg: float = 25.0,
                     min_length_ratio: float = 0.35,
                     canny1: int = 60, canny2: int = 180,
                     hough_thresh: int = 60,
                     max_line_gap_px: int = 10,
                     resize_max_w: int | None = 960):
    if cv2 is None:
        return None
    gray, _, scale = _pil_to_cv_gray(pil_img, resize_max_w=resize_max_w)
    if gray is None:
        return None
    h_s, w_s = gray.shape[:2]
    min_len_px = int(max(1, min_length_ratio * h_s))

    edges = cv2.Canny(gray, canny1, canny2)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=hough_thresh,
                            minLineLength=min_len_px, maxLineGap=max_line_gap_px)
    if lines is None or len(lines) == 0:
        return None

    best = None
    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx, dy = (x2 - x1), (y2 - y1)
        length = math.hypot(dx, dy)
        if length < min_len_px:
            continue
        angle_deg_signed = math.degrees(math.atan2(dy, dx))
        if 90 - angle_tol_deg <= abs(angle_deg_signed) <= 90 + angle_tol_deg:
            if (best is None) or (length > best[0]):
                best = (length, (x1, y1, x2, y2, angle_deg_signed))

    if best is None:
        return None

    _, (x1, y1, x2, y2, angle_deg_signed) = best
    if scale != 0:
        inv = 1.0 / scale
        x1, y1, x2, y2 = int(round(x1*inv)), int(round(y1*inv)), int(round(x2*inv)), int(round(y2*inv))
    return (x1, y1, x2, y2, angle_deg_signed)

def _rotate_cv_keep_bounds(rgb: np.ndarray, angle_deg: float, border_value=(255, 255, 255)):
    (h, w) = rgb.shape[:2]
    c = (w/2.0, h/2.0)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w/2.0) - c[0]
    M[1, 2] += (new_h/2.0) - c[1]
    rotated = cv2.warpAffine(rgb, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=border_value)
    return rotated, M

def _apply_affine_to_points(M, pts_xy):
    pts = np.asarray(pts_xy, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts, ones], axis=1)
    out = (M @ pts_h.T).T
    return out

def crop_to_pole_deskew(pil_img: Image.Image, width_fraction: float = 0.33, **detect_kwargs) -> Image.Image | None:
    if cv2 is None:
        return None
    res = detect_pole_line(pil_img, **detect_kwargs)
    if res is None:
        return None
    x1, y1, x2, y2, ang = res
    rot_deg = 90.0 - ang
    rgb = np.array(pil_img.convert("RGB"))
    rotated, M = _rotate_cv_keep_bounds(rgb, rot_deg, border_value=(255, 255, 255))
    pts_rot = _apply_affine_to_points(M, np.array([[x1, y1], [x2, y2]], dtype=np.float32))
    cx = float(pts_rot[:, 0].mean())
    f = max(0.05, min(1.0, float(width_fraction)))
    H, W = rotated.shape[:2]
    crop_w = max(1, int(round(W * f)))
    left = int(round(cx - crop_w/2))
    right = left + crop_w
    if left < 0:
        right -= left; left = 0
    if right > W:
        left -= (right - W); right = W; left = max(0, left)
    if right <= left:
        return None
    roi = rotated[:, left:right, :]
    return Image.fromarray(roi)

def detect_pole_center_x(pil_img: Image.Image,
                         angle_tol_deg: float = 12.0,
                         min_length_ratio: float = 0.35,
                         canny1: int = 60, canny2: int = 180,
                         hough_thresh: int = 60,
                         max_line_gap_px: int = 10,
                         resize_max_w: int | None = 960) -> int | None:
    res = detect_pole_line(pil_img, angle_tol_deg, min_length_ratio, canny1, canny2,
                           hough_thresh, max_line_gap_px, resize_max_w)
    if res is None:
        return None
    x1, y1, x2, y2, _ = res
    cx_full = int(round((x1 + x2) / 2.0))
    cx_full = int(np.clip(cx_full, 0, pil_img.size[0]-1))
    return cx_full

def crop_to_pole(pil_img: Image.Image, width_fraction: float = 0.33, **detect_kwargs) -> Image.Image | None:
    cx = detect_pole_center_x(pil_img, **detect_kwargs)
    if cx is None:
        return None
    f = max(0.05, min(1.0, float(width_fraction)))
    w, h = pil_img.size
    crop_w = max(1, int(round(w * f)))
    left = int(round(cx - crop_w / 2))
    right = left + crop_w
    if left < 0:
        right -= left; left = 0
    if right > w:
        left -= (right - w); right = w; left = max(0, left)
    if right <= left:
        return None
    return pil_img.crop((left, 0, right, h))

def extract_frames_from_video(video_path: str, every_sec: float = 2.0, max_frames: int | None = None, crop_fraction: float = 1.0):
    if cv2 is None:
        raise RuntimeError("OpenCV(cv2)がインストールされていません。pip install opencv-python")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けません: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    next_sample_ms = 0.0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_ms is None or pos_ms <= 0:
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            pos_ms = (frame_idx / fps) * 1000.0

        if pos_ms + 1e-6 >= next_sample_ms:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            if crop_fraction < 0.999:
                pil_img = crop_center_horizontal_fraction(pil_img, crop_fraction)
            frames.append((pil_img, pos_ms / 1000.0))
            next_sample_ms += every_sec * 1000.0
            if max_frames is not None and len(frames) >= max_frames:
                break

    cap.release()
    return frames

# =========================================================
# モデル定義・前処理
# =========================================================
class CustomResNet18(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CustomResNet18, self).__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.fc_class = nn.Linear(base_model.fc.in_features, num_classes)  # 3分類
        self.fc_regression = nn.Linear(base_model.fc.in_features, 1)       # 回帰(0-1)
        self.sigmoid = nn.Sigmoid()  # 回帰出力を0〜1に
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        class_logits = self.fc_class(x)
        regression_output = self.sigmoid(self.fc_regression(x))
        return class_logits, regression_output

class_names = ["Corrosion", "no-Corrosion", "base"]

def to_display_name(label: str) -> str:
    low = label.strip().lower()
    if low in ("base",):
        return "new"
    if low in ("no-corrosion", "no-corossion"):
        return "no-corrosion"
    if low in ("corrosion",):
        return "corrosion"
    return label

def norm_label(s: str) -> str:
    return s.lower().replace("_", "-").replace(" ", "")

@st.cache_resource
def load_model(path, num_classes=3):
    try:
        base_model = models.resnet18(weights=None)
    except TypeError:
        base_model = models.resnet18(pretrained=False)
    model = CustomResNet18(base_model, num_classes)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def infer_image(model, image: Image.Image):
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        class_logits, regression_output = model(input_tensor)
        logits = class_logits.squeeze().numpy()
        class_scores = torch.softmax(class_logits, dim=1).squeeze().numpy()
        predicted_class = int(torch.argmax(class_logits, 1).item())
        regression_value = float(regression_output.item())
    return logits, class_scores, predicted_class, regression_value

# =========================================================
# サイドバー（常時表示：モデル選択・入力モード）
# =========================================================
SEARCH_ROOTS = [
    ".", "saved_models", "models", "weights", "checkpoints",
    "sample", "samples", "images", "dataset",
]
VIDEO_SEARCH_ROOTS = ["movie", "movies", "video", "videos", "."]

with st.sidebar:
    st.header("1) モデルの選択（常時表示）")

    # ディレクトリ候補
    candidate_model_dirs = walk_dirs_with_files(SEARCH_ROOTS, MODEL_EXTS, max_depth=2, limit=200)
    if not candidate_model_dirs:
        st.info("💡 モデルが入ったフォルダが見つかりません。下のアップロードをご利用ください。")
    model_dir = st.selectbox(
        "モデルディレクトリ",
        candidate_model_dirs if candidate_model_dirs else ["（見つかりませんでした）"],
        index=0
    )

    # ファイル候補
    model_candidates = list_files_with_ext(model_dir, MODEL_EXTS) if candidate_model_dirs else []
    model_file = st.selectbox(
        "モデルファイル",
        model_candidates if model_candidates else ["（選択なし）"],
        index=0 if model_candidates else 0
    )
    picked_model_path = None
    if model_candidates:
        picked_model_path = os.path.join(model_dir, model_file)

    st.caption("またはアップロード：")
    up_model = st.file_uploader("モデルファイル（.pth/.pt/.bin）をアップロード", type=[e.replace(".", "") for e in MODEL_EXTS])
    if up_model is not None:
        # 一時ファイルに保存
        suffix = os.path.splitext(up_model.name)[1]
        tmp_path = os.path.join(tempfile.gettempdir(), f"uploaded_model{suffix}")
        with open(tmp_path, "wb") as f:
            f.write(up_model.read())
        picked_model_path = tmp_path
        st.success(f"アップロード済み：{up_model.name}")

    st.divider()
    st.header("2) 入力データの選択（常時表示）")
    INPUT_MODE = st.radio("入力データ", ["画像フォルダ or アップロード", "動画フォルダ or アップロード"], index=0)

    st.caption("表示設定")
    TILE_SIZE = st.slider("タイル高さ（px）", 200, 540, 320, step=10)
    IMAGE_SHRINK = st.slider("画像の縮小率（内側に余白）", 0.60, 1.00, 0.85, 0.01)
    regression_mode = st.selectbox("回帰の意味", ["劣化度スコア (0-1)", "腐食面積率 (0-1)"])
    layout_mode = st.radio("レイアウト", ["3列（画像+分類+回帰）", "2列（画像+図）"], index=0)

    if INPUT_MODE.startswith("動画"):
        if cv2 is None:
            st.warning("動画処理には OpenCV が必要です。`pip install opencv-python` を実行してください。")
        FRAME_EVERY_SEC = st.slider("フレーム間隔（秒）", 0.5, 10.0, 2.0, 0.5)
        MAX_FRAMES_PER_VIDEO = st.number_input("最大フレーム/動画", min_value=1, max_value=2000, value=200, step=10)

        CROP_MODE_VIDEO = st.radio("フレーム切り出し方法", ["なし", "中央", "ポール自動", "ポール自動＋回転補正"], index=3)
        if CROP_MODE_VIDEO == "中央":
            CENTER_CROP_FRACTION = st.slider("中央トリミング幅（横割合）", 0.10, 1.00, 0.33, 0.01)
        else:
            CENTER_CROP_FRACTION = 1.0

        if CROP_MODE_VIDEO in ("ポール自動", "ポール自動＋回転補正"):
            POLE_CROP_FRACTION = st.slider("ポール切り出し幅（横割合）", 0.10, 1.00, 0.33, 0.01)
            with st.expander("ポール検出の詳細設定", expanded=False):
                ANG_TOL = st.slider("垂直とみなす角度±（度）", 5, 40, 25, 1)
                MIN_LEN_RATIO = st.slider("最小縦線長（画像高さ比）", 0.10, 0.90, 0.35, 0.05)
                CANNY1 = st.slider("Canny閾値1", 0, 255, 60, 1)
                CANNY2 = st.slider("Canny閾値2", 0, 255, 180, 1)
                HOUGH_THR = st.slider("Houghしきい値", 1, 200, 60, 1)
                MAX_GAP = st.slider("線分の最大ギャップ(px/縮小後)", 0, 50, 10, 1)
                RESIZE_MAX_W = st.slider("検出時の最大幅（高速化）", 320, 1920, 960, 10)
        else:
            POLE_CROP_FRACTION, ANG_TOL, MIN_LEN_RATIO = 0.33, 25, 0.35
            CANNY1, CANNY2, HOUGH_THR, MAX_GAP, RESIZE_MAX_W = 60, 180, 60, 10, 960
    else:
        APPLY_IMAGE_CROP = st.checkbox("画像にも切り出しを適用する", value=False)
        if APPLY_IMAGE_CROP:
            CROP_MODE_IMAGE = st.radio("画像の切り出し方法", ["中央", "ポール自動", "ポール自動＋回転補正"], index=2)
            if CROP_MODE_IMAGE == "中央":
                IMG_CENTER_FRACTION = st.slider("中央トリミング幅（横割合/画像）", 0.10, 1.00, 0.33, 0.01)
            elif CROP_MODE_IMAGE == "ポール自動":
                IMG_POLE_FRACTION = st.slider("ポール切り出し幅（横割合/画像）", 0.10, 1.00, 0.33, 0.01)
            else:
                IMG_POLE_FRACTION = st.slider("ポール切り出し幅（横割合/画像）", 0.10, 1.00, 0.33, 0.01)
                if cv2 is None:
                    st.warning("回転補正には OpenCV が必要です。インストールが無い場合は回転補正なしで処理します。")
        else:
            CROP_MODE_IMAGE = "中央"
            IMG_CENTER_FRACTION = 1.0
            IMG_POLE_FRACTION = 0.33

# 常時：モデル準備
model = None
model_status = st.empty()
if 'picked_model_path' in locals() and picked_model_path:
    try:
        model = load_model(picked_model_path, num_classes=len(class_names))
        model_status.success(f"✅ モデル読込：{os.path.basename(picked_model_path)}")
    except Exception as e:
        model_status.error(f"❌ モデルの読み込みに失敗しました: {e}")
else:
    model_status.info("ℹ️ モデルを選択するかアップロードしてください。")

# =========================================================
# 入力ファイルUI（常時表示）
# =========================================================
regression_results = []

# 画像 UI
selected_images = []
uploaded_images = []

# 動画 UI
selected_videos = []
uploaded_videos = []

if INPUT_MODE.startswith("画像"):
    st.subheader("画像入力")
    candidate_image_dirs = walk_dirs_with_files(SEARCH_ROOTS, IMAGE_EXTS, max_depth=2, limit=200)
    if candidate_image_dirs:
        image_dir = st.selectbox("画像フォルダを選択", candidate_image_dirs, index=0, key="img_dir")
        image_files = list_files_with_ext(image_dir, IMAGE_EXTS)
        if image_files:
            default_pick = image_files
            selected_images = st.multiselect("表示する画像を選択（複数可）", image_files, default=default_pick, key="img_pick")
        else:
            st.warning("⚠️ フォルダ内に画像が見つかりません。")
    else:
        st.info("💡 画像フォルダが見つかりません。下でアップロードできます。")

    up_imgs = st.file_uploader("画像をアップロード（複数可）", type=[e.replace(".", "") for e in IMAGE_EXTS], accept_multiple_files=True, key="img_upl")
    if up_imgs:
        for f in up_imgs:
            try:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
                uploaded_images.append((f.name, img))
            except Exception as e:
                st.error(f"{f.name} の読み込みに失敗：{e}")

else:
    st.subheader("動画入力")
    candidate_video_dirs = walk_dirs_with_files(VIDEO_SEARCH_ROOTS, VIDEO_EXTS, max_depth=2, limit=200)
    if candidate_video_dirs:
        video_dir = st.selectbox("動画フォルダを選択", candidate_video_dirs, index=0, key="vid_dir")
        video_files = list_files_with_ext(video_dir, VIDEO_EXTS)
        video_files = sorted(video_files, key=lambda x: (not x.lower().endswith(".mp4"), x.lower()))
        if video_files:
            default_pick = [f for f in video_files if f.lower().endswith(".mp4")] or video_files
            selected_videos = st.multiselect("処理する動画を選択（複数可）", video_files, default=default_pick, key="vid_pick")
        else:
            st.warning("⚠️ フォルダ内に動画が見つかりません。")
    else:
        st.info("💡 動画フォルダが見つかりません。下でアップロードできます。")

    up_vids = st.file_uploader("動画をアップロード（複数可）", type=[e.replace(".", "") for e in VIDEO_EXTS], accept_multiple_files=True, key="vid_upl")
    if up_vids:
        for f in up_vids:
            # 一時ファイルに保存（OpenCVはメモリから直接は扱いづらい）
            tmp_path = os.path.join(tempfile.gettempdir(), f.name)
            with open(tmp_path, "wb") as g:
                g.write(f.read())
            uploaded_videos.append(tmp_path)
        if uploaded_videos:
            st.success(f"{len(uploaded_videos)} 本の動画を一時保存しました。")

# =========================================================
# 実行ボタン（UIを常に残しつつ必要条件が揃ったら実行）
# =========================================================
run = st.button("▶ 推論を実行")

if run:
    if model is None:
        st.error("モデルが未選択です。モデルを選択／アップロードしてください。")
    else:
        # ====== 画像モード ======
        if INPUT_MODE.startswith("画像"):
            DISPLAY_ORDER_INTERNAL = ["Corrosion", "no-Corrosion", "base"]
            DISPLAY_ORDER_CANON = [norm_label(x) for x in DISPLAY_ORDER_INTERNAL]
            norm_to_idx = {norm_label(lbl): i for i, lbl in enumerate(class_names)}

            # フォルダ選択分を先に実行
            if selected_images:
                image_dir = locals().get("image_dir", ".")
                for image_file in selected_images:
                    image_path = os.path.join(image_dir, image_file)
                    try:
                        image = Image.open(image_path).convert("RGB")

                        # 切り出し（任意）
                        if 'APPLY_IMAGE_CROP' in locals() and APPLY_IMAGE_CROP:
                            if CROP_MODE_IMAGE == "中央":
                                image = crop_center_horizontal_fraction(image, IMG_CENTER_FRACTION)
                            elif CROP_MODE_IMAGE == "ポール自動":
                                cropped = crop_to_pole(
                                    image, width_fraction=IMG_POLE_FRACTION,
                                    angle_tol_deg=25.0, min_length_ratio=0.35,
                                    canny1=60, canny2=180, hough_thresh=60, max_line_gap_px=10, resize_max_w=960
                                )
                                if cropped is not None:
                                    image = cropped
                            else:  # ポール自動＋回転補正
                                cropped = crop_to_pole_deskew(
                                    image, width_fraction=IMG_POLE_FRACTION,
                                    angle_tol_deg=25.0, min_length_ratio=0.35,
                                    canny1=60, canny2=180, hough_thresh=60, max_line_gap_px=10, resize_max_w=960
                                ) if cv2 is not None else None
                                if cropped is not None:
                                    image = cropped
                                elif cv2 is None:
                                    cropped2 = crop_to_pole(
                                        image, width_fraction=IMG_POLE_FRACTION,
                                        angle_tol_deg=25.0, min_length_ratio=0.35,
                                        canny1=60, canny2=180, hough_thresh=60, max_line_gap_px=10, resize_max_w=960
                                    )
                                    if cropped2 is not None:
                                        image = cropped2

                        logits, class_scores, pred_class_idx, reg_value = infer_image(model, image)

                        top_idx = int(np.argmax(class_scores))
                        idx_cor = norm_to_idx["corrosion"]
                        is_corrosion_top = (top_idx == idx_cor)
                        idx_new = norm_to_idx["base"]
                        if top_idx == idx_new:
                            reg_value = 0.0

                        pred_internal = class_names[pred_class_idx]
                        pred_display = to_display_name(norm_label(pred_internal))
                        is_corrosion = (pred_display.lower() == "corrosion")

                        tile_img = fit_square_canvas(image, TILE_SIZE, bg=(255, 255, 255), inner_ratio=IMAGE_SHRINK)
                        if is_corrosion:
                            tile_img = draw_red_border(tile_img, width=4, color=(255, 0, 0))

                        ordered_scores, ordered_labels_display = [], []
                        for canon in DISPLAY_ORDER_CANON:
                            idx = norm_to_idx.get(canon, None)
                            score = class_scores[idx] if idx is not None else 0.0
                            disp = to_display_name(canon)
                            ordered_scores.append(float(score))
                            ordered_labels_display.append(disp)
                        fig_cls = make_square_bar_figure(
                            ordered_labels_display, ordered_scores, size_px=TILE_SIZE,
                            title="分類スコア", red_border=is_corrosion
                        )

                        apply_gray_hatch = (not is_corrosion_top)
                        reg_title = "劣化度スコア (0-1)" if "劣化度" in regression_mode else "腐食面積率 (0-1)"
                        fig_reg = make_square_bar_figure(
                            ["劣化度" if "劣化度" in regression_mode else "面積率"], [reg_value], size_px=TILE_SIZE,
                            title=reg_title, red_border=is_corrosion,
                            hatch_pattern=("////" if apply_gray_hatch else None),
                            hatch_facecolor=((0.85, 0.85, 0.85) if apply_gray_hatch else None)
                        )

                        if layout_mode.startswith("3列"):
                            c1, c2, c3 = st.columns([1, 1, 1], gap="medium", vertical_alignment="top")
                            with c1:
                                st.markdown(f"**📷 {image_file}**")
                                st.image(tile_img, caption=f"Predicted: {pred_display}", use_container_width=True)
                            with c2:
                                st.pyplot(fig_cls, clear_figure=True, use_container_width=True)
                            with c3:
                                st.pyplot(fig_reg, clear_figure=True, use_container_width=True)
                        else:
                            left, right = st.columns([1, 1], gap="large", vertical_alignment="top")
                            with left:
                                st.markdown(f"**📷 {image_file}**")
                                st.image(tile_img, caption=f"Predicted: {pred_display}", use_container_width=True)
                                st.pyplot(fig_cls, clear_figure=True, use_container_width=True)
                            with right:
                                st.pyplot(fig_reg, clear_figure=True, use_container_width=True)

                        plt.close(fig_cls); plt.close(fig_reg)
                        regression_results.append((image_file, float(reg_value)))
                        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"{image_file} の処理中にエラー: {e}")

            # アップロード分
            if uploaded_images:
                DISPLAY_ORDER_INTERNAL = ["Corrosion", "no-Corrosion", "base"]
                DISPLAY_ORDER_CANON = [norm_label(x) for x in DISPLAY_ORDER_INTERNAL]
                norm_to_idx = {norm_label(lbl): i for i, lbl in enumerate(class_names)}

                for fname, image in uploaded_images:
                    try:
                        img = image

                        if 'APPLY_IMAGE_CROP' in locals() and APPLY_IMAGE_CROP:
                            if CROP_MODE_IMAGE == "中央":
                                img = crop_center_horizontal_fraction(img, IMG_CENTER_FRACTION)
                            elif CROP_MODE_IMAGE == "ポール自動":
                                cropped = crop_to_pole(
                                    img, width_fraction=IMG_POLE_FRACTION,
                                    angle_tol_deg=25.0, min_length_ratio=0.35,
                                    canny1=60, canny2=180, hough_thresh=60, max_line_gap_px=10, resize_max_w=960
                                )
                                if cropped is not None:
                                    img = cropped
                            else:
                                cropped = crop_to_pole_deskew(
                                    img, width_fraction=IMG_POLE_FRACTION,
                                    angle_tol_deg=25.0, min_length_ratio=0.35,
                                    canny1=60, canny2=180, hough_thresh=60, max_line_gap_px=10, resize_max_w=960
                                ) if cv2 is not None else None
                                if cropped is not None:
                                    img = cropped
                                elif cv2 is None:
                                    cropped2 = crop_to_pole(
                                        img, width_fraction=IMG_POLE_FRACTION,
                                        angle_tol_deg=25.0, min_length_ratio=0.35,
                                        canny1=60, canny2=180, hough_thresh=60, max_line_gap_px=10, resize_max_w=960
                                    )
                                    if cropped2 is not None:
                                        img = cropped2

                        logits, class_scores, pred_class_idx, reg_value = infer_image(model, img)

                        top_idx = int(np.argmax(class_scores))
                        idx_cor = norm_to_idx["corrosion"]
                        is_corrosion_top = (top_idx == idx_cor)
                        idx_new = norm_to_idx["base"]
                        if top_idx == idx_new:
                            reg_value = 0.0

                        pred_internal = class_names[pred_class_idx]
                        pred_display = to_display_name(norm_label(pred_internal))
                        is_corrosion = (pred_display.lower() == "corrosion")

                        tile_img = fit_square_canvas(img, TILE_SIZE, bg=(255, 255, 255), inner_ratio=IMAGE_SHRINK)
                        if is_corrosion:
                            tile_img = draw_red_border(tile_img, width=4, color=(255, 0, 0))

                        ordered_scores, ordered_labels_display = [], []
                        for canon in DISPLAY_ORDER_CANON:
                            idx = norm_to_idx.get(canon, None)
                            score = class_scores[idx] if idx is not None else 0.0
                            disp = to_display_name(canon)
                            ordered_scores.append(float(score))
                            ordered_labels_display.append(disp)
                        fig_cls = make_square_bar_figure(
                            ordered_labels_display, ordered_scores, size_px=TILE_SIZE,
                            title="分類スコア", red_border=is_corrosion
                        )

                        apply_gray_hatch = (not is_corrosion_top)
                        reg_title = "劣化度スコア (0-1)" if "劣化度" in regression_mode else "腐食面積率 (0-1)"
                        fig_reg = make_square_bar_figure(
                            ["劣化度" if "劣化度" in regression_mode else "面積率"], [reg_value], size_px=TILE_SIZE,
                            title=reg_title, red_border=is_corrosion,
                            hatch_pattern=("////" if apply_gray_hatch else None),
                            hatch_facecolor=((0.85, 0.85, 0.85) if apply_gray_hatch else None)
                        )

                        if layout_mode.startswith("3列"):
                            c1, c2, c3 = st.columns([1, 1, 1], gap="medium", vertical_alignment="top")
                            with c1:
                                st.markdown(f"**📷 {fname}（アップロード）**")
                                st.image(tile_img, caption=f"Predicted: {pred_display}", use_container_width=True)
                            with c2:
                                st.pyplot(fig_cls, clear_figure=True, use_container_width=True)
                            with c3:
                                st.pyplot(fig_reg, clear_figure=True, use_container_width=True)
                        else:
                            left, right = st.columns([1, 1], gap="large", vertical_alignment="top")
                            with left:
                                st.markdown(f"**📷 {fname}（アップロード）**")
                                st.image(tile_img, caption=f"Predicted: {pred_display}", use_container_width=True)
                                st.pyplot(fig_cls, clear_figure=True, use_container_width=True)
                            with right:
                                st.pyplot(fig_reg, clear_figure=True, use_container_width=True)

                        plt.close(fig_cls); plt.close(fig_reg)
                        regression_results.append((fname, float(reg_value)))
                        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"{fname} の処理中にエラー: {e}")

        # ====== 動画モード ======
        else:
            if cv2 is None:
                st.error("OpenCV が無いので動画処理は実行できません。`pip install opencv-python` を行ってください。")
            else:
                DISPLAY_ORDER_INTERNAL = ["Corrosion", "no-Corrosion", "base"]
                DISPLAY_ORDER_CANON = [norm_label(x) for x in DISPLAY_ORDER_INTERNAL]
                norm_to_idx = {norm_label(lbl): i for i, lbl in enumerate(class_names)}

                # 1) フォルダ選択分
                for video_file in selected_videos:
                    video_path = os.path.join(locals().get("video_dir", "."), video_file)
                    try:
                        with st.spinner(f"フレーム抽出中: {video_file}（{FRAME_EVERY_SEC:.1f}s間隔, 最大{MAX_FRAMES_PER_VIDEO}枚, 切り出し:{locals().get('CROP_MODE_VIDEO','')}）"):
                            frames = extract_frames_from_video(
                                video_path,
                                every_sec=float(locals().get("FRAME_EVERY_SEC", 2.0)),
                                max_frames=int(locals().get("MAX_FRAMES_PER_VIDEO", 200)),
                                crop_fraction=float(locals().get("CENTER_CROP_FRACTION", 1.0))
                            )
                        if not frames:
                            st.warning(f"⚠️ 抽出フレームなし: {video_file}")
                            continue

                        for pil_img, sec in frames:
                            if locals().get("CROP_MODE_VIDEO") == "ポール自動":
                                cropped = crop_to_pole(
                                    pil_img, width_fraction=locals().get("POLE_CROP_FRACTION", 0.33),
                                    angle_tol_deg=locals().get("ANG_TOL", 25),
                                    min_length_ratio=locals().get("MIN_LEN_RATIO", 0.35),
                                    canny1=locals().get("CANNY1", 60),
                                    canny2=locals().get("CANNY2", 180),
                                    hough_thresh=locals().get("HOUGH_THR", 60),
                                    max_line_gap_px=locals().get("MAX_GAP", 10),
                                    resize_max_w=locals().get("RESIZE_MAX_W", 960)
                                )
                                if cropped is not None:
                                    pil_img = cropped
                                else:
                                    pil_img = crop_center_horizontal_fraction(pil_img, locals().get("POLE_CROP_FRACTION", 0.33))
                            elif locals().get("CROP_MODE_VIDEO") == "ポール自動＋回転補正":
                                cropped = crop_to_pole_deskew(
                                    pil_img, width_fraction=locals().get("POLE_CROP_FRACTION", 0.33),
                                    angle_tol_deg=locals().get("ANG_TOL", 25),
                                    min_length_ratio=locals().get("MIN_LEN_RATIO", 0.35),
                                    canny1=locals().get("CANNY1", 60),
                                    canny2=locals().get("CANNY2", 180),
                                    hough_thresh=locals().get("HOUGH_THR", 60),
                                    max_line_gap_px=locals().get("MAX_GAP", 10),
                                    resize_max_w=locals().get("RESIZE_MAX_W", 960)
                                )
                                if cropped is not None:
                                    pil_img = cropped
                                else:
                                    pil_img = crop_center_horizontal_fraction(pil_img, locals().get("POLE_CROP_FRACTION", 0.33))

                            display_id = f"{video_file} @ {format_time_label(sec)}"

                            logits, class_scores, pred_class_idx, reg_value = infer_image(model, pil_img)

                            top_idx = int(np.argmax(class_scores))
                            idx_cor = norm_to_idx["corrosion"]
                            is_corrosion_top = (top_idx == idx_cor)
                            idx_new = norm_to_idx["base"]
                            if top_idx == idx_new:
                                reg_value = 0.0

                            pred_internal = class_names[pred_class_idx]
                            pred_display = to_display_name(norm_label(pred_internal))
                            is_corrosion = (pred_display.lower() == "corrosion")

                            tile_img = fit_square_canvas(pil_img, TILE_SIZE, bg=(255, 255, 255), inner_ratio=IMAGE_SHRINK)
                            if is_corrosion:
                                tile_img = draw_red_border(tile_img, width=4, color=(255, 0, 0))

                            ordered_scores, ordered_labels_display = [], []
                            for canon in DISPLAY_ORDER_CANON:
                                idx = norm_to_idx.get(canon, None)
                                score = class_scores[idx] if idx is not None else 0.0
                                disp = to_display_name(canon)
                                ordered_scores.append(float(score))
                                ordered_labels_display.append(disp)
                            fig_cls = make_square_bar_figure(
                                ordered_labels_display, ordered_scores, size_px=TILE_SIZE,
                                title="分類スコア", red_border=is_corrosion
                            )

                            apply_gray_hatch = (not is_corrosion_top)
                            reg_title = "劣化度スコア (0-1)" if "劣化度" in regression_mode else "腐食面積率 (0-1)"
                            fig_reg = make_square_bar_figure(
                                ["劣化度" if "劣化度" in regression_mode else "面積率"], [reg_value], size_px=TILE_SIZE,
                                title=reg_title, red_border=is_corrosion,
                                hatch_pattern=("////" if apply_gray_hatch else None),
                                hatch_facecolor=((0.85, 0.85, 0.85) if apply_gray_hatch else None)
                            )

                            if layout_mode.startswith("3列"):
                                c1, c2, c3 = st.columns([1, 1, 1], gap="medium", vertical_alignment="top")
                                with c1:
                                    st.markdown(f"**🎞 {display_id}**")
                                    st.image(tile_img, caption=f"Predicted: {pred_display}", use_container_width=True)
                                with c2:
                                    st.pyplot(fig_cls, clear_figure=True, use_container_width=True)
                                with c3:
                                    st.pyplot(fig_reg, clear_figure=True, use_container_width=True)
                            else:
                                left, right = st.columns([1, 1], gap="large", vertical_alignment="top")
                                with left:
                                    st.markdown(f"**🎞 {display_id}**")
                                    st.image(tile_img, caption=f"Predicted: {pred_display}", use_container_width=True)
                                    st.pyplot(fig_cls, clear_figure=True, use_container_width=True)
                                with right:
                                    st.pyplot(fig_reg, clear_figure=True, use_container_width=True)

                            plt.close(fig_cls); plt.close(fig_reg)
                            regression_results.append((display_id, float(reg_value)))
                            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"{video_file} の処理中にエラー: {e}")

                # 2) アップロード分
                for tmp_path in uploaded_videos:
                    try:
                        base = os.path.basename(tmp_path)
                        with st.spinner(f"フレーム抽出中: {base}（アップロード）"):
                            frames = extract_frames_from_video(
                                tmp_path,
                                every_sec=float(locals().get("FRAME_EVERY_SEC", 2.0)),
                                max_frames=int(locals().get("MAX_FRAMES_PER_VIDEO", 200)),
                                crop_fraction=float(locals().get("CENTER_CROP_FRACTION", 1.0))
                            )
                        if not frames:
                            st.warning(f"⚠️ 抽出フレームなし: {base}")
                            continue

                        for pil_img, sec in frames:
                            if locals().get("CROP_MODE_VIDEO") == "ポール自動":
                                cropped = crop_to_pole(
                                    pil_img, width_fraction=locals().get("POLE_CROP_FRACTION", 0.33),
                                    angle_tol_deg=locals().get("ANG_TOL", 25),
                                    min_length_ratio=locals().get("MIN_LEN_RATIO", 0.35),
                                    canny1=locals().get("CANNY1", 60),
                                    canny2=locals().get("CANNY2", 180),
                                    hough_thresh=locals().get("HOUGH_THR", 60),
                                    max_line_gap_px=locals().get("MAX_GAP", 10),
                                    resize_max_w=locals().get("RESIZE_MAX_W", 960)
                                )
                                if cropped is not None:
                                    pil_img = cropped
                                else:
                                    pil_img = crop_center_horizontal_fraction(pil_img, locals().get("POLE_CROP_FRACTION", 0.33))
                            elif locals().get("CROP_MODE_VIDEO") == "ポール自動＋回転補正":
                                cropped = crop_to_pole_deskew(
                                    pil_img, width_fraction=locals().get("POLE_CROP_FRACTION", 0.33),
                                    angle_tol_deg=locals().get("ANG_TOL", 25),
                                    min_length_ratio=locals().get("MIN_LEN_RATIO", 0.35),
                                    canny1=locals().get("CANNY1", 60),
                                    canny2=locals().get("CANNY2", 180),
                                    hough_thresh=locals().get("HOUGH_THR", 60),
                                    max_line_gap_px=locals().get("MAX_GAP", 10),
                                    resize_max_w=locals().get("RESIZE_MAX_W", 960)
                                )
                                if cropped is not None:
                                    pil_img = cropped
                                else:
                                    pil_img = crop_center_horizontal_fraction(pil_img, locals().get("POLE_CROP_FRACTION", 0.33))

                            display_id = f"{base} @ {format_time_label(sec)}"

                            logits, class_scores, pred_class_idx, reg_value = infer_image(model, pil_img)

                            top_idx = int(np.argmax(class_scores))
                            idx_cor = norm_to_idx["corrosion"]
                            is_corrosion_top = (top_idx == idx_cor)
                            idx_new = norm_to_idx["base"]
                            if top_idx == idx_new:
                                reg_value = 0.0

                            pred_internal = class_names[pred_class_idx]
                            pred_display = to_display_name(norm_label(pred_internal))
                            is_corrosion = (pred_display.lower() == "corrosion")

                            tile_img = fit_square_canvas(pil_img, TILE_SIZE, bg=(255, 255, 255), inner_ratio=IMAGE_SHRINK)
                            if is_corrosion:
                                tile_img = draw_red_border(tile_img, width=4, color=(255, 0, 0))

                            ordered_scores, ordered_labels_display = [], []
                            for canon in DISPLAY_ORDER_CANON:
                                idx = norm_to_idx.get(canon, None)
                                score = class_scores[idx] if idx is not None else 0.0
                                disp = to_display_name(canon)
                                ordered_scores.append(float(score))
                                ordered_labels_display.append(disp)
                            fig_cls = make_square_bar_figure(
                                ordered_labels_display, ordered_scores, size_px=TILE_SIZE,
                                title="分類スコア", red_border=is_corrosion
                            )

                            apply_gray_hatch = (not is_corrosion_top)
                            reg_title = "劣化度スコア (0-1)" if "劣化度" in regression_mode else "腐食面積率 (0-1)"
                            fig_reg = make_square_bar_figure(
                                ["劣化度" if "劣化度" in regression_mode else "面積率"], [reg_value], size_px=TILE_SIZE,
                                title=reg_title, red_border=is_corrosion,
                                hatch_pattern=("////" if apply_gray_hatch else None),
                                hatch_facecolor=((0.85, 0.85, 0.85) if apply_gray_hatch else None)
                            )

                            if layout_mode.startswith("3列"):
                                c1, c2, c3 = st.columns([1, 1, 1], gap="medium", vertical_alignment="top")
                                with c1:
                                    st.markdown(f"**🎞 {display_id}（アップロード）**")
                                    st.image(tile_img, caption=f"Predicted: {pred_display}", use_container_width=True)
                                with c2:
                                    st.pyplot(fig_cls, clear_figure=True, use_container_width=True)
                                with c3:
                                    st.pyplot(fig_reg, clear_figure=True, use_container_width=True)
                            else:
                                left, right = st.columns([1, 1], gap="large", vertical_alignment="top")
                                with left:
                                    st.markdown(f"**🎞 {display_id}（アップロード）**")
                                    st.image(tile_img, caption=f"Predicted: {pred_display}", use_container_width=True)
                                    st.pyplot(fig_cls, clear_figure=True, use_container_width=True)
                                with right:
                                    st.pyplot(fig_reg, clear_figure=True, use_container_width=True)

                            plt.close(fig_cls); plt.close(fig_reg)
                            regression_results.append((display_id, float(reg_value)))
                            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"{os.path.basename(tmp_path)} の処理中にエラー: {e}")

# =========================================================
# 結果まとめ（任意表示）
# =========================================================
if regression_results:
    st.subheader("劣化度スコアまとめ")
    for name, val in regression_results:
        st.write(f"- **{name}**: {val:.3f}")
