import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image

# 1. 設定網頁標題與版面寬度
st.set_page_config(page_title="車牌辨識與自動放大系統", page_icon="🚗", layout="wide")

# 2. 載入 EasyOCR 模型 (使用 st.cache_resource 讓模型只載入一次，加快後續速度)
@st.cache_resource
def load_model():
    # 雲端免費版通常沒有 GPU，所以設定 gpu=False
    return easyocr.Reader(['en'], gpu=False)

reader = load_model()

# 3. 網頁標題
st.title("📸 車牌照片自動辨識與放大系統")
st.write("請上傳一張包含車牌的照片，系統會自動標記位置、**裁切放大**並辨識號碼。")

# 4. 檔案上傳元件
uploaded_file = st.file_uploader("選擇圖片檔案...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 讀取圖片並轉為 NumPy 陣列 (OpenCV 格式)
    image = Image.open(uploaded_file)
    # 確保圖片為 RGB 格式
    img_np = np.array(image.convert('RGB'))
    
    # 顯示載入中動畫
    with st.spinner('AI 正在尋找車牌與辨識文字中，請稍候...'):
        # 進行 OCR 辨識
        results = reader.readtext(img_np)
        
    if not results:
        st.warning("⚠️ 找不到任何文字或車牌，請嘗試更清晰的照片。")
    else:
        # 切割版面：左邊顯示原圖(佔比2)，右邊顯示放大結果(佔比1)
        col1, col2 = st.columns([2, 1])
        
        # 複製一張圖用來畫框框
        img_with_boxes = img_np.copy()
        valid_detections = []

        # 處理辨識結果
        for (bbox, text, prob) in results:
            # 過濾掉信心度太低 (低於 30%) 的雜訊
            if prob < 0.3:
                continue
                
            # 取得四個頂點座標
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            br = (int(br[0]), int(br[1]))
            
            # 在原圖上畫綠色框框 (粗細度為 3)
            cv2.rectangle(img_with_boxes, tl, br, (0, 255, 0), 3)
            
            # --- 核心：自動裁切 (放大) ---
            h, w, _ = img_np.shape
            padding = 15 # 邊界留白
            y1 = max(0, tl[1] - padding)
            y2 = min(h, br[1] + padding)
            x1 = max(0, tl[0] - padding)
            x2 = min(w, br[0] + padding)
            
            # 切割陣列
            cropped_img = img_np[y1:y2, x1:x2]
            
            # 將結果儲存到列表
            valid_detections.append({
                "cropped": cropped_img,
                "text": text,
                "prob": prob
            })

        # --- 顯示左側畫面 (畫好框的原圖) ---
        with col1:
            st.subheader("原始圖片 (自動定位)")
            st.image(img_with_boxes, use_column_width=True)

        # --- 顯示右側畫面 (裁切放大的車牌) ---
        with col2:
            st.subheader("🔍 放大車牌 & 辨識結果")
            
            if len(valid_detections) == 0:
                st.info("沒有找到符合信心度標準的車牌。")
            else:
                for det in valid_detections:
                    st.image(det["cropped"], caption="自動放大截圖", use_column_width=True)
                    st.success(f"**車牌號碼： {det['text']}**")
                    st.caption(f"AI 信心度: {det['prob']*100:.1f}%")
                    st.markdown("---") # 分隔線
