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

# 3. 自動縮圖 (設定在 1000px，兼顧效能與畫質)
def resize_image(image, max_width=1000):
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        resized_img = cv2.resize(image, (max_width, new_h))
        return resized_img
    return image

# 4. 影像銳化處理函數 (對抗模糊的 N, M, K)
def enhance_image_for_ocr(img):
    # 轉為灰階
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # 使用 CLAHE 提升對比度 (讓黑字更黑，白底更白)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(gray)
    
    # 建立「銳化遮罩 (Sharpening Kernel)」
    # 這一行的魔法能把模糊的邊緣變得銳利，N 就是 N，K 就是 K！
    kernel = np.array([[0, -1, 0], 
                       [-1, 5,-1], 
                       [0, -1, 0]])
    sharpened = cv2.filter2D(contrast_img, -1, kernel)
    
    return sharpened

# --- 網頁介面 ---
st.title("📸 車牌照片自動辨識與放大系統 (高畫質銳化版)")
st.write("已導入 OpenCV 邊緣銳化技術，大幅提升 N, M, K 等相似字母的辨識率。")

uploaded_file = st.file_uploader("選擇圖片檔案...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 讀取圖片並轉為 RGB
    image = Image.open(uploaded_file)
    original_img = np.array(image.convert('RGB'))
    
    # 縮小原圖以避免記憶體崩潰
    img_np = resize_image(original_img, max_width=1000)
    img_h, img_w, _ = img_np.shape
    
    # --- 執行影像強化 ---
    enhanced_img = enhance_image_for_ocr(img_np)
    
    with st.spinner('⏳ AI 正在深度掃描並辨識車牌中，請稍候...'):
        # 進行 OCR (關閉 mag_ratio 節省記憶體，因為圖片已經銳化過了)
        results = reader.readtext(
            enhanced_img, 
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        )
        
    if not results:
        st.warning("⚠️ 找不到任何符合的車牌。")
    else:
        col1, col2 = st.columns([2, 1])
        
        # 複製彩色圖來畫框
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
            
            # --- 正規表達式過濾 ---
            text = text.upper()
            text = text.strip('-')
            
            # 車牌格式
            if not re.search(r'^[A-Z0-9]{2,4}-[A-Z0-9]{2,4}$', text):
                continue
                
            # 信心度過濾
            if prob < 0.3:
                continue

            # --- 畫框 ---
            cv2.rectangle(img_with_boxes, tl, br, (0, 255, 0), 3)
            
            # --- 裁切放大圖 (使用高品質插值法放大裁切出來的車牌) ---
            padding = 10 
            y1 = max(0, tl[1] - padding)
            y2 = min(img_h, br[1] + padding)
            x1 = max(0, tl[0] - padding)
            x2 = min(img_w, br[0] + padding)
            
            cropped_img = img_np[y1:y2, x1:x2]
            # 讓右側顯示的圖片經過平滑放大，視覺上更舒服
            display_crop = cv2.resize(cropped_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
            valid_detections.append({
                "cropped": display_crop,
                "text": text,
                "prob": prob
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
                    st.caption(f"AI 信心度: {det['prob']*100:.1f}%")
                    st.markdown("---")
