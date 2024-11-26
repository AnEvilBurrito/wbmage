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
    """标准化图像大小和质量"""
    # 设定标准尺寸
    standard_height = 500  # 设置一个较高的标准高度以保持清晰度
    
    # 计算宽度，保持原始宽高比
    aspect_ratio = image.shape[1] / image.shape[0]
    standard_width = int(standard_height * aspect_ratio)
    
    # 调整图像大小
    resized = cv2.resize(image, (standard_width, standard_height), 
                        interpolation=cv2.INTER_LANCZOS4)  # 使用Lanczos插值获得更好的质量
    
    # 图像锐化
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(resized, -1, kernel)
    
    # 降噪同时保持边缘
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
    
    # 调整对比度和亮度
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def process_image(image):
    """处理图像的主函数"""
    # 首先进行标准化
    standardized = standardize_image(image)
    
    # 转换为灰度图
    gray = cv2.cvtColor(standardized, cv2.COLOR_BGR2GRAY)
    
    # 进一步的图像增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 去噪
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # 反转图像（使条带为白色
    inverted = cv2.bitwise_not(denoised)
    
    return inverted, standardized

def detect_bands(image):
    """改进的条带检测函数"""
    # 预处理以减少阴影影响
    blur = cv2.GaussianBlur(image, (5,5), 0)
    
    # 使用Otsu阈值分割，更好地分离条带和背景
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 形态学操作去除噪声和阴影
    kernel_v = np.ones((15,1), np.uint8)  # 垂直方向的核
    kernel_h = np.ones((1,5), np.uint8)   # 水平方向的核
    
    # 开运算去除小噪点
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    # 闭运算填充条带内的空隙
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_v)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 合并过近的轮廓
    merged_contours = []
    min_distance = 2  # 最小距离阈值
    
    # 按x坐标排序轮廓
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    current_contour = None
    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # 过滤掉太小的轮廓（可能是噪声）
        if area < 50:  # 最小面积阈值
            continue
            
        # 过滤掉高宽比异常的轮廓（可能是阴影）
        aspect_ratio = w / h
        if aspect_ratio > 5 or aspect_ratio < 0.2:  # 限制长宽比
            continue
        
        if current_contour is None:
            current_contour = contour
        else:
            # 获取当前轮廓的边界框
            curr_x, _, curr_w, _ = cv2.boundingRect(current_contour)
            
            # 如果两个轮廓足够近，合并它们
            if x - (curr_x + curr_w) < min_distance:
                # 创建合并的轮廓
                combined_contour = np.vstack((current_contour, contour))
                current_contour = combined_contour
            else:
                merged_contours.append(current_contour)
                current_contour = contour
    
    # 添加最后一个轮廓
    if current_contour is not None:
        merged_contours.append(current_contour)
    
    # 对每个合并后的轮廓进行优化
    final_contours = []
    for cnt in merged_contours:
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 稍微扩大ROI区域
        roi = image[max(0, y-5):min(image.shape[0], y+h+5),
                   max(0, x-5):min(image.shape[1], x+w+5)]
        
        # 对ROI重新进行阈值分割
        _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 在ROI中找到最大轮廓
        roi_contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
        
        if roi_contours:
            # 选择最大的轮廓
            max_cnt = max(roi_contours, key=cv2.contourArea)
            # 调整坐标到原图
            max_cnt = max_cnt + np.array([max(0, x-5), max(0, y-5)])[None, None, :]
            final_contours.append(max_cnt)
    
    return final_contours

def analyze_band_intensity(image, contours):
    """改进的条带强度分析函数"""
    results = []
    
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        
        # 扩大ROI区域以包含周围背景
        roi_y1 = max(0, y - 10)
        roi_y2 = min(image.shape[0], y + h + 10)
        roi_x1 = max(0, x - 10)
        roi_x2 = min(image.shape[1], x + w + 10)
        
        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
        
        # 创建掩码
        mask = np.zeros_like(roi)
        roi_cnt = cnt - np.array([roi_x1, roi_y1])[None, None, :]
        cv2.drawContours(mask, [roi_cnt], -1, 255, -1)
        
        # 计算背景（使用ROI边缘区域）
        bg_mask = cv2.bitwise_not(mask)
        background = np.median(roi[bg_mask > 0])
        
        # 计算条带强度
        band_pixels = roi[mask > 0]
        mean_intensity = np.mean(band_pixels) - background
        
        # 调整数值比例
        mean_intensity = (mean_intensity / 100)  # 将强度除以100
        area = cv2.contourArea(cnt) / 100  # 将面积除以100
        total_intensity = mean_intensity * len(band_pixels)
        
        # 计算积分密度
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
        # 添加原始图像信息显示
        st.sidebar.subheader("Image Information")
        st.sidebar.text(f"Original Size: {image.shape[1]}x{image.shape[0]}")
        
        # 处理图像
        processed, standardized = process_image(image)
        bands = detect_bands(processed)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(standardized, channels="BGR", use_container_width=True)
            st.caption(f"Standardized Size: {standardized.shape[1]}x{standardized.shape[0]}")
        
        # 绘制检测结果
        result = standardized.copy()
        for i, cnt in enumerate(bands):
            # 绘制轮廓
            cv2.drawContours(result, [cnt], -1, (0,255,255), 2)
            
            # 绘制边界框和标签
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x,y), (x+w,y+h), (0,255,255), 1)
            cv2.putText(result, f"#{i+1}", (x,y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        
        with col2:
            st.subheader("Detected Bands")
            st.image(result, channels="BGR", use_container_width=True)
        
        # 分析结果
        if len(bands) > 0:
            st.subheader("Band Analysis Results")
            analysis_results = analyze_band_intensity(processed, bands)
            
            # 创建DataFrame
            df = pd.DataFrame(analysis_results)
            
            # 添加相对积分密度
            max_density = df['integrated_density'].max()
            df['relative_density'] = (df['integrated_density'] / max_density * 100)
            
            # 设置显示列并四舍五入到两位小数
            display_cols = [
                'band_number', 
                'area',
                'mean_intensity',
                'integrated_density',
                'relative_density'
            ]
            
            # 对所有数值列四舍五入到两位小数
            df = df.round(2)
            
            st.dataframe(df[display_cols])
            
            # 添加积分密度分布图
            st.subheader("Integrated Density Distribution")
            fig = go.Figure()
            
            # 添加积分密度柱状图
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