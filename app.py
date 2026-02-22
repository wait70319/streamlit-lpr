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

def resize_image(image, max_width=1280):
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        resized_img = cv2.resize(image, (max_width, new_h))
        return resized_img
    return image

st.title("📸 車牌照片自動辨識與畫中畫特寫")
st.write("精準辨識車牌，具備夜視增強與抗斷字技術。")

uploaded_file = st.file_uploader("選擇圖片檔案...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    original_img = np.array(image.convert('RGB'))
    
    img_np = resize_image(original_img, max_width=1280)
    img_h, img_w, _ = img_np.shape
    final_output_img = img_np.copy()
    
    with st.spinner('⏳ AI 正在深度掃描並繪製特寫圖，請稍候...'):
        
        # ==========================================
        # 【第一階段】：找車牌位置 (加入夜視增強與防斷字)
        # ==========================================
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # 1. 針對陰天/傍晚的畫面，先做一次全局對比強化 (讓黑底白字更明顯)
        clahe_global = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_img = clahe_global.apply(gray_img)
        
        # 2. 加入 width_ths=0.7 參數，強制 AI 把 BFY 跟 3805 視為同一行文字，不要切斷！
        results = reader.readtext(
            gray_img, 
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ', 
            width_ths=0.7
        )
        
    if not results:
        st.warning("⚠️ 找不到任何符合的車牌。")
    else:
        valid_detections = []

        for (bbox, text, prob) in results:
            x1, y1 = int(bbox[0][0]), int(bbox[0][1])
            x2, y2 = int(bbox[2][0]), int(bbox[2][1])
            center_y = (y1 + y2) / 2
            
            # --- 放寬浮水印排除區 ---
            # 因為有些車牌較低，我們將底部排除區從 15% 縮小到 10% (0.90)
            if center_y > (img_h * 0.90) or center_y < (img_h * 0.10):
                continue
            
            # --- 寬容的格式過濾器 (非常關鍵) ---
            text = text.upper()
            # 1. 強制清除所有空白 (解決 AI 讀成 BFY - 3805 的問題)
            text = re.sub(r'\s+', '', text)
            # 2. 去除頭尾可能誤判的橫槓
            text = text.strip('-')
            
            # 3. 檢查格式：允許中間的橫槓「存在」或「不存在」 (-?)
            # 這樣即使 AI 在第一階段沒看到橫槓 (BFY3805)，也不會被丟掉！
            if not re.search(r'^[A-Z0-9]{2,4}-?[A-Z0-9]{2,4}$', text):
                continue
                
            # 字元長度太短的雜訊過濾
            if len(text) < 5 or len(text) > 8:
                continue

            # --- 裁切乾淨的車牌 ---
            padding = 5
            crop_y1 = max(0, y1 - padding)
            crop_y2 = min(img_h, y2 + padding)
            crop_x1 = max(0, x1 - padding)
            crop_x2 = min(img_w, x2 + padding)
            clean_cropped_plate = img_np[crop_y1:crop_y2, crop_x1:crop_x2]

            # ==========================================
            # 【第二階段】：AI 二值化字體瘦身
            # ==========================================
            zoom_plate = cv2.resize(clean_cropped_plate, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            zoom_gray = cv2.cvtColor(zoom_plate, cv2.COLOR_RGB2GRAY)
            _, binary_plate = cv2.threshold(zoom_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            thinned_plate = cv2.dilate(binary_plate, kernel, iterations=1)
            final_feed = cv2.cvtColor(thinned_plate, cv2.COLOR_GRAY2RGB)
            
            # 逼 AI 重讀超清晰版
            final_text_result = reader.readtext(final_feed, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
            final_text = final_text_result[0] if len(final_text_result) > 0 else text

            valid_detections.append(final_text)

            # ==========================================
            # 【第三階段】：繪製畫中畫特寫
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

        # --- 輸出結果 ---
        if len(valid_detections) == 0:
            st.info("沒有找到符合標準的車牌。")
        else:
            st.image(final_output_img, use_column_width=True, caption="自動特寫合成圖")
            for text in valid_detections:
                st.success(f"🎯 **AI 最終辨識結果： {text}**")
