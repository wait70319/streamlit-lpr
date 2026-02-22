import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import re

# 1. 設定網頁標題
st.set_page_config(page_title="車牌辨識與自動放大系統", page_icon="🚗", layout="wide")

# 2. 載入模型 (快取加速)
@st.cache_resource(show_spinner="📥 系統正在喚醒 AI 模型，請耐心等候...")
def load_model():
    return easyocr.Reader(['en'], gpu=False)

reader = load_model()

# 3. 自動縮圖函數 (最大寬度設為 1280 保持足夠細節)
def resize_image(image, max_width=1280):
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        resized_img = cv2.resize(image, (max_width, new_h))
        return resized_img
    return image

# --- 網頁介面 ---
st.title("📸 車牌照片自動辨識與放大系統 (終極精準版)")
st.write("已加入【浮水印空間排除】與【內部光學放大】技術。")

uploaded_file = st.file_uploader("選擇圖片檔案...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 讀取圖片並轉為 RGB
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert('RGB'))
    
    # 縮小原圖以加快速度
    img_np = resize_image(img_np)
    
    # 取得圖片的高度 (用來過濾浮水印)
    img_h, img_w, _ = img_np.shape
    
    with st.spinner('⏳ AI 正在深度掃描並辨識車牌中，請稍候...'):
        # --- 核心升級：加入高階辨識參數 ---
        # mag_ratio=2.5: 在 AI 辨識前先將圖片放大 2.5 倍 (專治 N/M 不分)
        # adjust_contrast=True: 讓 AI 自動修復背光問題
        results = reader.readtext(
            img_np, 
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-',
            mag_ratio=2.5,         
            adjust_contrast=True   
        )
        
    if not results:
        st.warning("⚠️ 找不到任何符合的車牌。")
    else:
        col1, col2 = st.columns([2, 1])
        
        img_with_boxes = img_np.copy()
        valid_detections = []

        for (bbox, text, prob) in results:
            # 取得座標
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            br = (int(br[0]), int(br[1]))
            
            # 計算這個文字的「中心點 Y 座標」
            center_y = (tl[1] + br[1]) / 2
            
            # --- 殺手鐧 1：空間位置過濾 (排除浮水印) ---
            # 如果文字出現在畫面最底部 15% 或最頂部 10% 區域，認定為行車紀錄器浮水印，直接丟棄！
            if center_y > (img_h * 0.85) or center_y < (img_h * 0.10):
                continue
            
            # --- 殺手鐧 2：嚴格的正規表達式 ---
            text = text.upper()
            
            # 清除可能誤判的開頭或結尾符號 (例如不小心把邊框認成 - )
            text = text.strip('-')
            
            # 檢查是否符合 車牌格式 (2~4碼英數 + 一個橫槓 + 2~4碼英數)
            if not re.search(r'^[A-Z0-9]{2,4}-[A-Z0-9]{2,4}$', text):
                continue
                
            # 信心度過濾
            if prob < 0.3:
                continue

            # --- 畫框與裁切 ---
            cv2.rectangle(img_with_boxes, tl, br, (0, 255, 0), 3)
            
            padding = 10 
            y1 = max(0, tl[1] - padding)
            y2 = min(img_h, br[1] + padding)
            x1 = max(0, tl[0] - padding)
            x2 = min(img_w, br[0] + padding)
            
            cropped_img = img_np[y1:y2, x1:x2]
            
            valid_detections.append({
                "cropped": cropped_img,
                "text": text,
                "prob": prob
            })

        # --- 顯示畫面 ---
        with col1:
            st.subheader("原始圖片 (自動過濾浮水印)")
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
