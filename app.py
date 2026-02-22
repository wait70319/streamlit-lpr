import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import re  # 引入正規表達式模組，用來過濾車牌格式

# 1. 設定網頁標題與版面寬度
st.set_page_config(page_title="車牌辨識與自動放大系統", page_icon="🚗", layout="wide")

# 2. 載入模型 (快取加速)
@st.cache_resource(show_spinner="📥 系統正在喚醒 AI 模型 (初次執行約需 30~60 秒，請耐心等候)...")
def load_model():
    return easyocr.Reader(['en'], gpu=False)

reader = load_model()

# 3. 自動縮圖函數 (加速核心)
def resize_image(image, max_width=1000):
    """如果圖片太寬，依比例縮小，大幅加快 AI 辨識速度"""
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        resized_img = cv2.resize(image, (max_width, new_h))
        return resized_img
    return image

# --- 網頁介面開始 ---
st.title("📸 車牌照片自動辨識與放大系統 (高準度版)")
st.write("請上傳一張包含車牌的照片，系統會自動強化影像、過濾雜訊，並精準抓取車牌號碼。")

uploaded_file = st.file_uploader("選擇圖片檔案...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 讀取圖片並確保為 RGB 格式
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert('RGB'))
    
    # 執行縮圖以提升速度
    img_np = resize_image(img_np)
    
    # --- 影像強化 (讓 AI 看得更清楚) ---
    # 將圖片轉為灰階，並使用 CLAHE 提升對比度，克服背光或反光問題
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(gray_img)
    
    with st.spinner('⏳ AI 正在強化影像並辨識車牌中，請稍候...'):
        # 進行 OCR 辨識 (加入 allowlist 強制只辨識大寫英文、數字與連字號)
        results = reader.readtext(enhanced_img, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
        
    if not results:
        st.warning("⚠️ 找不到任何文字或車牌，請嘗試更清晰的照片。")
    else:
        col1, col2 = st.columns([2, 1])
        
        # 複製一張原圖用來畫綠色框框 (維持彩色)
        img_with_boxes = img_np.copy()
        valid_detections = []

        for (bbox, text, prob) in results:
            # 1. 信心度過濾：低於 40% 的雜訊直接忽略
            if prob < 0.4:
                continue
            
            # 2. 強制轉為大寫字串 (雙重保險)
            text = text.upper()
            
            # 3. 格式過濾 (Regex)：台灣車牌通常包含 '-'，且前後為 2~4 個英數字
            # 如果不符合這個格式 (例如左下角的時間浮水印)，就跳過不處理
            if not re.search(r'[A-Z0-9]{2,4}-[A-Z0-9]{2,4}', text):
                continue
                
            # --- 取得座標並畫框 ---
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            br = (int(br[0]), int(br[1]))
            
            cv2.rectangle(img_with_boxes, tl, br, (0, 255, 0), 3)
            
            # --- 自動裁切 (放大) ---
            h, w, _ = img_np.shape
            padding = 10  # 邊界留白
            y1 = max(0, tl[1] - padding)
            y2 = min(h, br[1] + padding)
            x1 = max(0, tl[0] - padding)
            x2 = min(w, br[0] + padding)
            
            # 從原圖 (彩色) 中切出車牌區域
            cropped_img = img_np[y1:y2, x1:x2]
            
            valid_detections.append({
                "cropped": cropped_img,
                "text": text,
                "prob": prob
            })

        # --- 顯示左側畫面 ---
        with col1:
            st.subheader("原始圖片 (自動定位)")
            st.image(img_with_boxes, use_column_width=True)

        # --- 顯示右側畫面 ---
        with col2:
            st.subheader("🔍 放大車牌 & 辨識結果")
            if len(valid_detections) == 0:
                st.info("沒有找到符合標準的車牌。 (已自動過濾掉日期與雜訊)")
            else:
                for det in valid_detections:
                    st.image(det["cropped"], use_column_width=True)
                    st.success(f"**車牌號碼： {det['text']}**")
                    st.caption(f"AI 信心度: {det['prob']*100:.1f}%")
                    st.markdown("---")
