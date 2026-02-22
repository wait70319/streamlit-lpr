import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import re

st.set_page_config(page_title="車牌辨識與自動放大系統", page_icon="🚗", layout="wide")

@st.cache_resource(show_spinner="📥 系統正在喚醒 AI 模型，請耐心等候...")
def load_model():
    return easyocr.Reader(['en'], gpu=False)

reader = load_model()

def resize_image(image, max_width=1000):
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        resized_img = cv2.resize(image, (max_width, new_h))
        return resized_img
    return image

st.title("📸 車牌照片自動辨識與放大系統 (極限瘦身版)")
st.write("已導入『二值化字體瘦身術』，強行切開糊在一起的 N 與 M！")

uploaded_file = st.file_uploader("選擇圖片檔案...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    original_img = np.array(image.convert('RGB'))
    
    img_np = resize_image(original_img, max_width=1000)
    img_h, img_w, _ = img_np.shape
    
    with st.spinner('⏳ AI 正在深度掃描並執行字體修復，請稍候...'):
        # 【第一階段】：找車牌位置
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
            
            # 排除浮水印
            if center_y > (img_h * 0.85) or center_y < (img_h * 0.10):
                continue
            
            # 格式檢查
            text = text.upper().strip('-')
            if not re.search(r'^[A-Z0-9]{2,4}-[A-Z0-9]{2,4}$', text):
                continue
            if prob < 0.2:
                continue

            cv2.rectangle(img_with_boxes, tl, br, (0, 255, 0), 3)
            
            padding = 5  # 減少 padding 避免干擾
            y1 = max(0, tl[1] - padding)
            y2 = min(img_h, br[1] + padding)
            x1 = max(0, tl[0] - padding)
            x2 = min(img_w, br[0] + padding)
            cropped_plate = img_np[y1:y2, x1:x2]
            
            # ==========================================
            # 【第二階段】：物理性削瘦字體 (魔改開始)
            # ==========================================
            
            # 1. 將車牌無損放大 3 倍
            zoom_plate = cv2.resize(cropped_plate, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            zoom_gray = cv2.cvtColor(zoom_plate, cv2.COLOR_RGB2GRAY)
            
            # 2. Otsu 二值化：強制把所有灰色漸層變成「純黑」與「純白」
            _, binary_plate = cv2.threshold(zoom_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 3. 字體瘦身 (Dilation)：擴張白色的背景，藉此把黑色的字體「壓細」
            # 這個 3x3 矩陣就是我們的手術刀，用來切開 N 中間黏住的地方
            kernel = np.ones((3, 3), np.uint8)
            thinned_plate = cv2.dilate(binary_plate, kernel, iterations=1)
            
            # 轉回 RGB 格式餵給 AI (EasyOCR 喜歡三通道圖片)
            final_feed = cv2.cvtColor(thinned_plate, cv2.COLOR_GRAY2RGB)
            
            # 4. 讓 AI 只讀這張「被削瘦過的黑白字體圖」
            final_text_result = reader.readtext(
                final_feed, 
                detail=0, 
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
            )
            
            # 如果還是讀不到，才退回第一階段的結果
            if len(final_text_result) > 0:
                final_text = final_text_result[0]
            else:
                final_text = text
            
            valid_detections.append({
                "cropped": final_feed, # 我們把動過手術的黑白圖秀在右邊，讓你看看效果！
                "text": final_text,
                "prob": prob
            })

        with col1:
            st.subheader("原始圖片 (自動定位)")
            st.image(img_with_boxes, use_column_width=True)

        with col2:
            st.subheader("🔍 字體瘦身 & 辨識結果")
            if len(valid_detections) == 0:
                st.info("沒有找到符合標準的車牌。")
            else:
                for det in valid_detections:
                    # 這裡顯示的會是純黑白的瘦身版車牌
                    st.image(det["cropped"], use_column_width=True, caption="二值化瘦身處理圖")
                    st.success(f"**車牌號碼： {det['text']}**")
                    st.markdown("---")
