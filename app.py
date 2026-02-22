import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import re

# 1. 設定網頁標題
st.set_page_config(page_title="車牌辨識與自動放大系統", page_icon="🚗", layout="wide")

# 2. 載入模型
@st.cache_resource(show_spinner="📥 系統正在喚醒 AI 模型，請耐心等候...")
def load_model():
    return easyocr.Reader(['en'], gpu=False)

reader = load_model()

# 3. 自動縮圖 (維持 1000px 安全尺寸)
def resize_image(image, max_width=1000):
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        resized_img = cv2.resize(image, (max_width, new_h))
        return resized_img
    return image

# --- 網頁介面 ---
st.title("📸 車牌照片自動辨識與放大系統 (兩段式超解析版)")
st.write("已導入專業級『局部超解析重辨識』技術，徹底解決 N/M 誤判問題。")

uploaded_file = st.file_uploader("選擇圖片檔案...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    original_img = np.array(image.convert('RGB'))
    
    # 縮小原圖
    img_np = resize_image(original_img, max_width=1000)
    img_h, img_w, _ = img_np.shape
    
    with st.spinner('⏳ AI 正在深度掃描並辨識車牌中，請稍候...'):
        # 【第一階段】：大範圍掃描，只為了找出車牌的「座標位置」
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        results = reader.readtext(gray_img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
        
    if not results:
        st.warning("⚠️ 找不到任何符合的車牌。")
    else:
        col1, col2 = st.columns([2, 1])
        
        img_with_boxes = img_np.copy()
        valid_detections = []

        for (bbox, text, prob) in results:
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            br = (int(br[0]), int(br[1]))
            
            center_y = (tl[1] + br[1]) / 2
            
            # --- 排除浮水印 ---
            if center_y > (img_h * 0.85) or center_y < (img_h * 0.10):
                continue
            
            # --- 格式過濾 ---
            text = text.upper().strip('-')
            if not re.search(r'^[A-Z0-9]{2,4}-[A-Z0-9]{2,4}$', text):
                continue
            if prob < 0.2:
                continue

            # --- 畫框 ---
            cv2.rectangle(img_with_boxes, tl, br, (0, 255, 0), 3)
            
            # --- 裁切車牌 ---
            padding = 8 
            y1 = max(0, tl[1] - padding)
            y2 = min(img_h, br[1] + padding)
            x1 = max(0, tl[0] - padding)
            x2 = min(img_w, br[0] + padding)
            
            cropped_plate = img_np[y1:y2, x1:x2]
            
            # ==========================================
            # 【第二階段】：局部超解析度重辨識 (殺手鐧)
            # ==========================================
            
            # 1. 將這張小車牌「無損放大 3 倍」
            zoom_plate = cv2.resize(cropped_plate, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            
            # 2. 轉灰階並強力提升對比度 (讓黑白分明，拉開 N 中間的縫隙)
            zoom_gray = cv2.cvtColor(zoom_plate, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            zoom_enhanced = clahe.apply(zoom_gray)
            
            # 3. 逼 AI 重新只看這張超清晰的放大車牌 (detail=0 代表只回傳文字)
            final_text_result = reader.readtext(
                zoom_enhanced, 
                detail=0, 
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
            )
            
            # 如果第二階段有讀到東西，就用第二階段的結果 (通常是最準的)
            # 如果沒讀到，就保留第一階段的結果
            final_text = final_text_result[0] if len(final_text_result) > 0 else text
            
            valid_detections.append({
                "cropped": zoom_plate, # 畫面直接秀出放大3倍的圖
                "text": final_text,
                "prob": prob # 保留原始信心度作為參考
            })

        # --- 顯示畫面 ---
        with col1:
            st.subheader("原始圖片 (自動定位)")
            st.image(img_with_boxes, use_column_width=True)

        with col2:
            st.subheader("🔍 放大車牌 & 辨識結果")
            if len(valid_detections) == 0:
                st.info("沒有找到符合標準的車牌。")
            else:
                for det in valid_detections:
                    st.image(det["cropped"], use_column_width=True)
                    st.success(f"**車牌號碼： {det['text']}**")
                    st.markdown("---")
