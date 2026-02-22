import streamlit as st
import cv2
import numpy as np
import easyocr
from PIL import Image
import re
import io # 新增 io 模組，用來處理圖片下載

st.set_page_config(page_title="車牌自動特寫與輸出", page_icon="📸", layout="wide")

# 載入 AI 模型 (僅用來尋找車牌座標)
@st.cache_resource(show_spinner="📥 系統正在喚醒 AI 模型，請耐心等候...")
def load_model():
    return easyocr.Reader(['en'], gpu=False)

reader = load_model()

# 縮圖函數：維持 1280px，確保輸出的合成圖畫質夠好
def resize_image(image, max_width=1280):
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        resized_img = cv2.resize(image, (max_width, new_h))
        return resized_img
    return image

st.title("📸 車牌自動定位與特寫圖輸出")
st.write("系統會自動尋找車牌位置，合成「畫中畫」放大特寫，並提供高畫質下載。")

uploaded_file = st.file_uploader("選擇圖片檔案...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    original_img = np.array(image.convert('RGB'))
    
    img_np = resize_image(original_img, max_width=1280)
    img_h, img_w, _ = img_np.shape
    
    # 這是我們要作畫與輸出的最終畫布
    final_output_img = img_np.copy()
    
    with st.spinner('⏳ 正在尋找車牌並合成特寫圖，請稍候...'):
        
        # 影像前處理 (加強對比，讓 AI 更好找位置)
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe_global = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_img = clahe_global.apply(gray_img)
        
        # 尋找車牌座標
        results = reader.readtext(
            gray_img, 
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ', 
            width_ths=0.7
        )
        
    found_plate = False

    for (bbox, text, prob) in results:
        x1, y1 = int(bbox[0][0]), int(bbox[0][1])
        x2, y2 = int(bbox[2][0]), int(bbox[2][1])
        center_y = (y1 + y2) / 2
        
        # --- 位置與格式過濾 (確保抓到的是車牌而不是浮水印) ---
        if center_y > (img_h * 0.90) or center_y < (img_h * 0.10):
            continue
            
        text = text.upper()
        text = re.sub(r'\s+', '', text).strip('-')
        
        if not re.search(r'^[A-Z0-9]{2,4}-?[A-Z0-9]{2,4}$', text):
            continue
            
        if len(text) < 5 or len(text) > 8:
            continue

        # 如果通過過濾，代表成功找到車牌
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
        # 將切下的車牌無損放大 4 倍
        display_scale = 4.0
        pip_w = int((crop_x2 - crop_x1) * display_scale)
        pip_h = int((crop_y2 - crop_y1) * display_scale)
        pip_img = cv2.resize(clean_cropped_plate, (pip_w, pip_h), interpolation=cv2.INTER_CUBIC)
        
        # 設定放大圖放在左上角
        pip_x1, pip_y1 = 30, 30
        pip_x2, pip_y2 = pip_x1 + pip_w, pip_y1 + pip_h
        
        # 將放大圖覆蓋到主畫面上
        final_output_img[pip_y1:pip_y2, pip_x1:pip_x2] = pip_img
        
        # 定義紅色與粗細
        RED = (255, 0, 0)
        THICKNESS = 4
        
        # 畫原車牌紅框 & 左上角放大圖紅框
        cv2.rectangle(final_output_img, (x1, y1), (x2, y2), RED, THICKNESS)
        cv2.rectangle(final_output_img, (pip_x1, pip_y1), (pip_x2, pip_y2), RED, THICKNESS)
        
        # 畫斜線連接兩個框
        pt_pip_bottom_right = (pip_x2, pip_y2)
        pt_plate_top_left = (x1, y1)
        cv2.line(final_output_img, pt_pip_bottom_right, pt_plate_top_left, RED, THICKNESS)
        
        # 只處理第一個找到的車牌就結束 (避免畫面太亂)
        break

    # --- 輸出結果與下載按鈕 ---
    if not found_plate:
        st.warning("⚠️ 找不到符合標準的車牌位置。")
    else:
        st.success("✅ 成功產生特寫圖！")
        
        # 顯示圖片
        st.image(final_output_img, use_column_width=True)
        
        # --- 準備下載功能 ---
        # 1. 將 Numpy 陣列轉回 PIL 圖片格式
        result_pil = Image.fromarray(final_output_img)
        # 2. 建立記憶體緩衝區
        buf = io.BytesIO()
        # 3. 將圖片以高畫質 JPEG 存入緩衝區
        result_pil.save(buf, format="JPEG", quality=95)
        # 4. 取得圖片的位元組資料
        byte_im = buf.getvalue()
        
        # 5. 建立 Streamlit 下載按鈕
        st.download_button(
            label="📥 下載完整合成圖",
            data=byte_im,
            file_name="license_plate_zoomed.jpg",
            mime="image/jpeg",
            # 讓按鈕變大變明顯
            use_container_width=True 
        )
