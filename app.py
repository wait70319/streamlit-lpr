import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import re

st.set_page_config(page_title="車牌辨識與特寫輸出", page_icon="📸", layout="wide")

@st.cache_resource(show_spinner="📥 系統正在喚醒 AI 模型，請耐心等候...")
def load_model():
    return easyocr.Reader(['en'], gpu=False)

reader = load_model()

# 維持 1280px 解析度，確保合成出來的圖片夠清晰
def resize_image(image, max_width=1280):
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        resized_img = cv2.resize(image, (max_width, new_h))
        return resized_img
    return image

st.title("📸 車牌照片自動辨識與畫中畫特寫")
st.write("精準辨識車牌，並自動生成帶有「紅框與引導線」的專業特寫合成圖。")

uploaded_file = st.file_uploader("選擇圖片檔案...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # 確保轉為 RGB 格式 (Streamlit 顯示需要 RGB)
    original_img = np.array(image.convert('RGB'))
    
    img_np = resize_image(original_img, max_width=1280)
    img_h, img_w, _ = img_np.shape
    
    # 建立一個畫布 (複製原圖)，我們將在這個畫布上作畫
    final_output_img = img_np.copy()
    
    with st.spinner('⏳ AI 正在深度掃描並繪製特寫圖，請稍候...'):
        # 【第一階段】：找車牌位置
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        results = reader.readtext(gray_img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
        
    if not results:
        st.warning("⚠️ 找不到任何符合的車牌。")
    else:
        valid_detections = []

        for (bbox, text, prob) in results:
            (tl, tr, br, bl) = bbox
            # 取得車牌座標 (加上 int 確保為整數)
            x1, y1 = int(tl[0]), int(tl[1])
            x2, y2 = int(br[0]), int(br[1])
            
            center_y = (y1 + y2) / 2
            
            # 排除浮水印 (上下邊緣過濾)
            if center_y > (img_h * 0.85) or center_y < (img_h * 0.10):
                continue
            
            # 格式檢查
            text = text.upper().strip('-')
            if not re.search(r'^[A-Z0-9]{2,4}-[A-Z0-9]{2,4}$', text):
                continue
            if prob < 0.2:
                continue

            # --- 裁切乾淨的車牌 ---
            padding = 5
            crop_y1 = max(0, y1 - padding)
            crop_y2 = min(img_h, y2 + padding)
            crop_x1 = max(0, x1 - padding)
            crop_x2 = min(img_w, x2 + padding)
            
            clean_cropped_plate = img_np[crop_y1:crop_y2, crop_x1:crop_x2]

            # ==========================================
            # 【第二階段】：AI 二值化字體瘦身 (維持最高準確率)
            # ==========================================
            zoom_plate = cv2.resize(clean_cropped_plate, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            zoom_gray = cv2.cvtColor(zoom_plate, cv2.COLOR_RGB2GRAY)
            _, binary_plate = cv2.threshold(zoom_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            thinned_plate = cv2.dilate(binary_plate, kernel, iterations=1)
            final_feed = cv2.cvtColor(thinned_plate, cv2.COLOR_GRAY2RGB)
            
            final_text_result = reader.readtext(final_feed, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
            final_text = final_text_result[0] if len(final_text_result) > 0 else text

            valid_detections.append(final_text)

            # ==========================================
            # 【第三階段】：繪製畫中畫 (Picture-in-Picture) 視覺特效
            # ==========================================
            
            # 1. 決定放大圖的尺寸與位置
            # 將裁切下來的車牌放大 4 倍作為顯示用
            display_scale = 4.0
            pip_w = int((crop_x2 - crop_x1) * display_scale)
            pip_h = int((crop_y2 - crop_y1) * display_scale)
            pip_img = cv2.resize(clean_cropped_plate, (pip_w, pip_h), interpolation=cv2.INTER_CUBIC)
            
            # 設定畫中畫放在左上角 (留 30px 的邊距)
            pip_x1, pip_y1 = 30, 30
            pip_x2, pip_y2 = pip_x1 + pip_w, pip_y1 + pip_h
            
            # 2. 將放大圖貼到主畫布上
            final_output_img[pip_y1:pip_y2, pip_x1:pip_x2] = pip_img
            
            # 3. 定義紅色 (RGB 格式為 255, 0, 0) 與線條粗細
            RED = (255, 0, 0)
            THICKNESS = 4
            
            # 4. 畫框與連接線
            # 畫小車牌的紅框
            cv2.rectangle(final_output_img, (x1, y1), (x2, y2), RED, THICKNESS)
            # 畫左上角放大圖的紅框
            cv2.rectangle(final_output_img, (pip_x1, pip_y1), (pip_x2, pip_y2), RED, THICKNESS)
            
            # 畫引導連接線 (從放大圖的右下角，連到原車牌的左上角)
            pt_pip_bottom_right = (pip_x2, pip_y2)
            pt_plate_top_left = (x1, y1)
            cv2.line(final_output_img, pt_pip_bottom_right, pt_plate_top_left, RED, THICKNESS)

        # --- 單一畫面輸出 ---
        if len(valid_detections) == 0:
            st.info("沒有找到符合標準的車牌。")
        else:
            # 顯示這張充滿科技感的合成大圖
            st.image(final_output_img, use_column_width=True, caption="自動特寫合成圖")
            
            # 在圖片下方用醒目的字體列出辨識結果
            for text in valid_detections:
                st.success(f"🎯 **AI 最終辨識結果： {text}**")
