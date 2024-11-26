import streamlit as st
import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image
import pandas as pd
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Western Blot Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def load_test_images():
    """
    Load test images from the 'pics' directory.

    Returns:
        list: List of image file names.
    """
    script_dir = Path(__file__).resolve().parent
    image_dir = script_dir / "pics"
    
    # Create directory if it doesn't exist
    if not image_dir.exists():
        image_dir.mkdir(parents=True)
        st.warning(f"Created images directory at {image_dir}")
        return []
    
    # List image files with proper error handling
    try:
        image_files = [f.name for f in image_dir.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
        return image_files
    except Exception as e:
        st.error(f"Error loading images: {str(e)}")
        return []

def safe_read_image(image_path):
    """
    Safely read an image with error handling.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Loaded image or None if an error occurred.
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image at {image_path}")
        return image
    except Exception as e:
        st.error(f"Error reading image: {str(e)}")
        return None

def standardize_image(image):
    """æ ‡å‡†åŒ–å›¾åƒå¤§å°å’Œè´¨é‡"""
    # è®¾å®šæ ‡å‡†å°ºå¯¸
    standard_height = 300  # è®¾ç½®ä¸€ä¸ªè¾ƒé«˜çš„æ ‡å‡†é«˜åº¦ä»¥ä¿æŒæ¸…æ™°åº¦
    
    # è®¡ç®—å®½åº¦ï¼Œä¿æŒåŽŸå§‹å®½é«˜æ¯”
    aspect_ratio = image.shape[1] / image.shape[0]
    standard_width = int(standard_height * aspect_ratio)
    
    # è°ƒæ•´å›¾åƒå¤§å°
    resized = cv2.resize(image, (standard_width, standard_height), 
                        interpolation=cv2.INTER_LANCZOS4)  # ä½¿ç”¨Lanczosæ’å€¼èŽ·å¾—æ›´å¥½çš„è´¨é‡
    
    # å›¾åƒé”åŒ–
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(resized, -1, kernel)
    
    # é™å™ªåŒæ—¶ä¿æŒè¾¹ç¼˜
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
    
    # è°ƒæ•´å¯¹æ¯”åº¦å’Œäº®åº¦
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def process_image(image):
    """å¤„ç†å›¾åƒçš„ä¸»å‡½æ•°"""
    # é¦–å…ˆè¿›è¡Œæ ‡å‡†åŒ–
    standardized = standardize_image(image)
    
    # è½¬æ¢ä¸ºç°åº¦å›¾
    gray = cv2.cvtColor(standardized, cv2.COLOR_BGR2GRAY)
    
    # è¿›ä¸€æ­¥çš„å›¾åƒå¢žå¼º
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # åŽ»å™ª
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # åè½¬å›¾åƒï¼ˆä½¿æ¡å¸¦ä¸ºç™½è‰²
    inverted = cv2.bitwise_not(denoised)
    
    return inverted, standardized

def detect_bands(image):
    """æ”¹è¿›çš„æ¡å¸¦æ£€æµ‹å‡½æ•°"""
    # é¢„å¤„ç†ä»¥å‡å°‘é˜´å½±å½±å“
    blur = cv2.GaussianBlur(image, (5,5), 0)
    
    # ä½¿ç”¨Otsué˜ˆå€¼åˆ†å‰²ï¼Œæ›´å¥½åœ°åˆ†ç¦»æ¡å¸¦å’ŒèƒŒæ™¯
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # å½¢æ€å­¦æ“ä½œåŽ»é™¤å™ªå£°å’Œé˜´å½±
    kernel_v = np.ones((15,1), np.uint8)  # åž‚ç›´æ–¹å‘çš„æ ¸
    kernel_h = np.ones((1,5), np.uint8)   # æ°´å¹³æ–¹å‘çš„æ ¸
    
    # å¼€è¿ç®—åŽ»é™¤å°å™ªç‚¹
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    # é—­è¿ç®—å¡«å……æ¡å¸¦å†…çš„ç©ºéš™
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_v)
    
    # å¯»æ‰¾è½®å»“
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # åˆå¹¶è¿‡è¿‘çš„è½®å»“
    merged_contours = []
    min_distance = 5  # æœ€å°è·ç¦»é˜ˆå€¼
    
    # æŒ‰xåæ ‡æŽ’åºè½®å»“
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    current_contour = None
    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # è¿‡æ»¤æŽ‰å¤ªå°çš„è½®å»“ï¼ˆå¯èƒ½æ˜¯å™ªå£°ï¼‰
        if area < 50:  # æœ€å°é¢ç§¯é˜ˆå€¼
            continue
            
        # è¿‡æ»¤æŽ‰é«˜å®½æ¯”å¼‚å¸¸çš„è½®å»“ï¼ˆå¯èƒ½æ˜¯é˜´å½±ï¼‰
        aspect_ratio = w / h
        if aspect_ratio > 5 or aspect_ratio < 0.2:  # é™åˆ¶é•¿å®½æ¯”
            continue
        
        if current_contour is None:
            current_contour = contour
        else:
            # èŽ·å–å½“å‰è½®å»“çš„è¾¹ç•Œæ¡†
            curr_x, _, curr_w, _ = cv2.boundingRect(current_contour)
            
            # å¦‚æžœä¸¤ä¸ªè½®å»“è¶³å¤Ÿè¿‘ï¼Œåˆå¹¶å®ƒä»¬
            if x - (curr_x + curr_w) < min_distance:
                # åˆ›å»ºåˆå¹¶çš„è½®å»“
                combined_contour = np.vstack((current_contour, contour))
                current_contour = combined_contour
            else:
                merged_contours.append(current_contour)
                current_contour = contour
    
    # æ·»åŠ æœ€åŽä¸€ä¸ªè½®å»“
    if current_contour is not None:
        merged_contours.append(current_contour)
    
    # å¯¹æ¯ä¸ªåˆå¹¶åŽçš„è½®å»“è¿›è¡Œä¼˜åŒ–
    final_contours = []
    for cnt in merged_contours:
        # èŽ·å–è½®å»“çš„è¾¹ç•Œæ¡†
        x, y, w, h = cv2.boundingRect(cnt)
        
        # ç¨å¾®æ‰©å¤§ROIåŒºåŸŸ
        roi = image[max(0, y-5):min(image.shape[0], y+h+5),
                   max(0, x-5):min(image.shape[1], x+w+5)]
        
        # å¯¹ROIé‡æ–°è¿›è¡Œé˜ˆå€¼åˆ†å‰²
        _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # åœ¨ROIä¸­æ‰¾åˆ°æœ€å¤§è½®å»“
        roi_contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
        
        if roi_contours:
            # é€‰æ‹©æœ€å¤§çš„è½®å»“
            max_cnt = max(roi_contours, key=cv2.contourArea)
            # è°ƒæ•´åæ ‡åˆ°åŽŸå›¾
            max_cnt = max_cnt + np.array([max(0, x-5), max(0, y-5)])[None, None, :]
            final_contours.append(max_cnt)
    
    return final_contours

def analyze_band_intensity(image, contours):
    """æ”¹è¿›çš„æ¡å¸¦å¼ºåº¦åˆ†æžå‡½æ•°"""
    results = []
    
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # æ‰©å¤§ROIåŒºåŸŸä»¥åŒ…å«å‘¨å›´èƒŒæ™¯
        roi_y1 = max(0, y - 10)
        roi_y2 = min(image.shape[0], y + h + 10)
        roi_x1 = max(0, x - 10)
        roi_x2 = min(image.shape[1], x + w + 10)
        
        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # åˆ›å»ºæŽ©ç 
        mask = np.zeros_like(roi)
        roi_cnt = cnt - np.array([roi_x1, roi_y1])[None, None, :]
        cv2.drawContours(mask, [roi_cnt], -1, 255, -1)
        
        # è®¡ç®—èƒŒæ™¯ï¼ˆä½¿ç”¨ROIè¾¹ç¼˜åŒºåŸŸï¼‰
        bg_mask = cv2.bitwise_not(mask)
        background = np.median(roi[bg_mask > 0])
        
        # è®¡ç®—æ¡å¸¦å¼ºåº¦
        band_pixels = roi[mask > 0]
        mean_intensity = np.mean(band_pixels) - background
        
        # è°ƒæ•´æ•°å€¼æ¯”ä¾‹
        mean_intensity = (mean_intensity / 100)  # å°†å¼ºåº¦é™¤ä»¥100
        area = cv2.contourArea(cnt) / 100  # å°†é¢ç§¯é™¤ä»¥100
        total_intensity = mean_intensity * len(band_pixels)
        
        # è®¡ç®—ç§¯åˆ†å¯†åº¦
        integrated_density = total_intensity * area
        
        results.append({
            'band_number': i+1,
            'area': area,
            'mean_intensity': mean_intensity,
            'total_intensity': total_intensity,
            'integrated_density': integrated_density
        })
    
    return results

# Update image path in main():
def main():
    st.title("Western Blot Band Analyzer")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Sidebar controls
    st.sidebar.header("Settings")
    test_images = load_test_images()
    
    if not test_images:
        st.error("No test images found in 'pics' directory")
        return
        
    selected_image = st.sidebar.selectbox("Select Test Image", test_images)
    
    # Construct proper path with os.path.join
    image_path = os.path.join(script_dir, "pics", selected_image)
    image = safe_read_image(image_path)
    
    if image is None:
        return
    
    if image is not None:
        # æ·»åŠ åŽŸå§‹å›¾åƒä¿¡æ¯æ˜¾ç¤º
        st.sidebar.subheader("Image Information")
        st.sidebar.text(f"Original Size: {image.shape[1]}x{image.shape[0]}")
        
        # å¤„ç†å›¾åƒ
        processed, standardized = process_image(image)
        bands = detect_bands(processed)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(standardized, channels="BGR", use_container_width=True)
            st.caption(f"Standardized Size: {standardized.shape[1]}x{standardized.shape[0]}")
        
        # ç»˜åˆ¶æ£€æµ‹ç»“æžœ
        result = standardized.copy()
        for i, cnt in enumerate(bands):
            # ç»˜åˆ¶è½®å»“
            cv2.drawContours(result, [cnt], -1, (0,255,255), 2)
            
            # ç»˜åˆ¶è¾¹ç•Œæ¡†å’Œæ ‡ç­¾
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x,y), (x+w,y+h), (0,255,255), 1)
            cv2.putText(result, f"#{i+1}", (x,y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        
        with col2:
            st.subheader("Detected Bands")
            st.image(result, channels="BGR", use_container_width=True)
        
        # åˆ†æžç»“æžœ
        if len(bands) > 0:
            st.subheader("Band Analysis Results")
            analysis_results = analyze_band_intensity(processed, bands)
            
            # åˆ›å»ºDataFrame
            df = pd.DataFrame(analysis_results)
            
            # æ·»åŠ ç›¸å¯¹ç§¯åˆ†å¯†åº¦
            max_density = df['integrated_density'].max()
            df['relative_density'] = (df['integrated_density'] / max_density * 100)
            
            # è®¾ç½®æ˜¾ç¤ºåˆ—å¹¶å››èˆäº”å…¥åˆ°ä¸¤ä½å°æ•°
            display_cols = [
                'band_number', 
                'area',
                'mean_intensity',
                'integrated_density',
                'relative_density'
            ]
            
            # å¯¹æ‰€æœ‰æ•°å€¼åˆ—å››èˆäº”å…¥åˆ°ä¸¤ä½å°æ•°
            df = df.round(2)
            
            st.dataframe(df[display_cols])
            
            # æ·»åŠ ç§¯åˆ†å¯†åº¦åˆ†å¸ƒå›¾
            st.subheader("Integrated Density Distribution")
            fig = go.Figure()
            
            # æ·»åŠ ç§¯åˆ†å¯†åº¦æŸ±çŠ¶å›¾
            fig.add_trace(go.Bar(
                x=df['band_number'],
                y=df['relative_density'],
                name='Relative Integrated Density',
                marker_color='rgb(55, 83, 109)'
            ))
            
            fig.update_layout(
                xaxis_title="Band Number",
                yaxis_title="Relative Integrated Density (%)",
                showlegend=True,
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()