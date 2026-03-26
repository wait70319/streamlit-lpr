import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
import re
import io

st.set_page_config(page_title="車牌自動特寫與輸出", page_icon="📸", layout="wide")

# OCR.space API 設定
OCR_SPACE_API_KEY = "K88343483088957"
OCR_SPACE_URL = "https://api.ocr.space/parse/image"

# 縮圖函數：維持 1280px，確保輸出的合成圖畫質夠好
def resize_image(image, max_width=1280):
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        resized_img = cv2.resize(image, (max_width, new_h))
        return resized_img
    return image

def call_ocr_space(image_np):
    """
    將 numpy 圖片送到 OCR.space API，回傳帶有座標的辨識結果。
    isOverlayRequired=True 才會回傳每個字的 bounding box。
    """
    # 將 numpy 陣列壓縮成 JPEG bytes
    pil_img = Image.fromarray(image_np)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    buf.seek(0)

    payload = {
        "apikey": OCR_SPACE_API_KEY,
        "language": "eng",
        "isOverlayRequired": "true",   # 取得每個字的座標
        "detectOrientation": "false",
        "scale": "true",               # 自動縮放提升準確率
        "OCREngine": "2",              # Engine 2 對車牌效果較好
    }

    try:
        response = requests.post(
            OCR_SPACE_URL,
            files={"file": ("plate.jpg", buf, "image/jpeg")},
            data=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ 呼叫 OCR.space API 失敗：{e}")
        return None

def parse_ocr_results(ocr_json, img_h, img_w):
    """
    從 OCR.space 回傳的 JSON 中，解析每個字的文字與座標，
    回傳格式與原本 EasyOCR 相同：List of (bbox, text, prob)
      bbox = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    """
    results = []

    if not ocr_json:
        return results

    parsed_results = ocr_json.get("ParsedResults", [])
    if not parsed_results:
        return results

    for page in parsed_results:
        overlay = page.get("TextOverlay", {})
        lines = overlay.get("Lines", [])
        for line in lines:
            words = line.get("Words", [])
            if not words:
                continue

            # 將同一行的所有字合併為一個候選車牌
            line_text = "".join(w.get("WordText", "") for w in words)

            # 計算整行的 bounding box
            lefts   = [w["Left"]              for w in words]
            tops    = [w["Top"]               for w in words]
            rights  = [w["Left"] + w["Width"] for w in words]
            bottoms = [w["Top"] + w["Height"] for w in words]

            x1, y1 = min(lefts), min(tops)
            x2, y2 = max(rights), max(bottoms)

            # OCR.space 座標是相對於送出的圖片，直接使用
            bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            results.append((bbox, line_text, 1.0))  # prob 固定 1.0（API 未提供）

            # 也對每個單字單獨輸出，提高短文字的命中率
            for w in words:
                wx1 = w["Left"]
                wy1 = w["Top"]
                wx2 = wx1 + w["Width"]
                wy2 = wy1 + w["Height"]
                wbbox = [[wx1, wy1], [wx2, wy1], [wx2, wy2], [wx1, wy2]]
                results.append((wbbox, w.get("WordText", ""), 1.0))

    return results

# ──────────────────────────── UI ────────────────────────────

st.title("📸 車牌自動定位與特寫圖輸出")
st.write("系統透過 OCR.space API 辨識車牌，合成「畫中畫」放大特寫並提供高畫質下載。")

uploaded_file = st.file_uploader("選擇圖片檔案...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    original_img = np.array(image.convert("RGB"))

    img_np = resize_image(original_img, max_width=1280)
    img_h, img_w, _ = img_np.shape

    # 這是我們要作畫與輸出的最終畫布
    final_output_img = img_np.copy()

    with st.spinner("⏳ 正在呼叫 OCR.space 辨識車牌，請稍候..."):
        ocr_json = call_ocr_space(img_np)
        results = parse_ocr_results(ocr_json, img_h, img_w)

    found_plate = False

    for (bbox, text, prob) in results:
        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
        x2, y2 = int(bbox[2][0]), int(bbox[2][1])
        center_y = (y1 + y2) / 2

        # --- 位置與格式過濾 ---
        if center_y > (img_h * 0.90) or center_y < (img_h * 0.10):
            continue

        text = text.upper()
        text = re.sub(r"\s+", "", text).strip("-")

        if not re.search(r"^[A-Z0-9]{2,4}-?[A-Z0-9]{2,4}$", text):
            continue

        if len(text) < 5 or len(text) > 8:
            continue

        # 通過過濾 → 找到車牌
        found_plate = True

        # --- 裁切車牌 ---
        padding = 5
        crop_y1 = max(0, y1 - padding)
        crop_y2 = min(img_h, y2 + padding)
        crop_x1 = max(0, x1 - padding)
        crop_x2 = min(img_w, x2 + padding)
        clean_cropped_plate = img_np[crop_y1:crop_y2, crop_x1:crop_x2]

        # ==========================================
        # 繪製畫中畫特寫 (Picture-in-Picture)
        # ==========================================
        display_scale = 4.0
        pip_w = int((crop_x2 - crop_x1) * display_scale)
        pip_h = int((crop_y2 - crop_y1) * display_scale)
        pip_img = cv2.resize(clean_cropped_plate, (pip_w, pip_h), interpolation=cv2.INTER_CUBIC)

        pip_x1, pip_y1 = 30, 30
        pip_x2, pip_y2 = pip_x1 + pip_w, pip_y1 + pip_h

        final_output_img[pip_y1:pip_y2, pip_x1:pip_x2] = pip_img

        RED = (255, 0, 0)
        THICKNESS = 4

        cv2.rectangle(final_output_img, (x1, y1), (x2, y2), RED, THICKNESS)
        cv2.rectangle(final_output_img, (pip_x1, pip_y1), (pip_x2, pip_y2), RED, THICKNESS)

        pt_pip_bottom_right = (pip_x2, pip_y2)
        pt_plate_top_left = (x1, y1)
        cv2.line(final_output_img, pt_pip_bottom_right, pt_plate_top_left, RED, THICKNESS)

        # 在放大圖下方標示辨識結果
        label = f"辨識結果: {text}"
        cv2.putText(
            final_output_img, label,
            (pip_x1, pip_y2 + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2, cv2.LINE_AA,
        )

        break  # 只處理第一個符合條件的車牌

    # --- 輸出結果與下載按鈕 ---
    if not found_plate:
        st.warning("⚠️ 找不到符合標準的車牌位置。")

        # 顯示 OCR 原始辨識到的所有文字，方便除錯
        if ocr_json:
            raw_texts = [
                w.get("WordText", "")
                for page in ocr_json.get("ParsedResults", [])
                for line in page.get("TextOverlay", {}).get("Lines", [])
                for w in line.get("Words", [])
            ]
            if raw_texts:
                st.info("OCR 辨識到的所有文字：" + "、".join(raw_texts))
    else:
        st.success("✅ 成功產生特寫圖！")
        st.image(final_output_img, use_column_width=True)

        result_pil = Image.fromarray(final_output_img)
        buf = io.BytesIO()
        result_pil.save(buf, format="JPEG", quality=95)
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 下載完整合成圖",
            data=byte_im,
            file_name="license_plate_zoomed.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )
