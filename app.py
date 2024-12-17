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
        image_files = [f.name for f in image_dir.iterdir() if f.suffix.lower() in {
            '.png', '.jpg', '.jpeg'}]
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
    """Standardize image size and quality"""
    # Set the standard height
    standard_height = 500  # Set a higher standard height to maintain clarity

    # Calculate width while maintaining the original aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]
    standard_width = int(standard_height * aspect_ratio)

    # Resize the image
    resized = cv2.resize(image, (standard_width, standard_height),
                         interpolation=cv2.INTER_LANCZOS4)  # Use Lanczos interpolation for better quality

    # Sharpen the image
    kernel = np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]])
    sharpened = cv2.filter2D(resized, -1, kernel)

    # Denoise while preserving edges
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)

    # Adjust contrast and brightness
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return enhanced


def process_image(image, clip_limit=2.0, tile_grid_size=8):
    """Main function for processing the image"""
    # First, standardize the image
    standardized = standardize_image(image)

    # Convert to grayscale
    gray = cv2.cvtColor(standardized, cv2.COLOR_BGR2GRAY)

    # Further image enhancement
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(
        tile_grid_size, tile_grid_size))
    enhanced = clahe.apply(gray)

    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

    # Invert the image (to make bands white)
    inverted = cv2.bitwise_not(denoised)

    return inverted, standardized


def detect_bands(image, min_distance=2, min_area=50):
    """Improved band detection function"""
    # Preprocess to reduce shadow effects
    blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Use Otsu thresholding to better separate bands and background
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to remove noise and shadows
    kernel_v = np.ones((15, 1), np.uint8)  # Vertical kernel
    kernel_h = np.ones((1, 5), np.uint8)   # Horizontal kernel

    # Open operation to remove small noise
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    # Close operation to fill gaps inside bands
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_v)

    # Find contours
    contours, _ = cv2.findContours(
        closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

    # Merge contours that are too close
    merged_contours = []

    # Sort contours by x-coordinate
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    current_contour = None
    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Filter out contours that are too small (likely noise)
        if area < min_area:  # Minimum area threshold
            continue

        # Filter out contours with abnormal aspect ratios (likely shadows)
        aspect_ratio = w / h
        if aspect_ratio > 5 or aspect_ratio < 0.2:  # Limit aspect ratio
            continue

        if current_contour is None:
            current_contour = contour
        else:
            # Get the bounding box of the current contour
            curr_x, _, curr_w, _ = cv2.boundingRect(current_contour)

            # If two contours are close enough, merge them
            if x - (curr_x + curr_w) < min_distance:
                # Create a merged contour
                combined_contour = np.vstack((current_contour, contour))
                current_contour = combined_contour
            else:
                merged_contours.append(current_contour)
                current_contour = contour

    # Add the last contour
    if current_contour is not None:
        merged_contours.append(current_contour)

    # Optimize each merged contour
    final_contours = []
    for cnt in merged_contours:
        # Get the bounding box of the contour
        x, y, w, h = cv2.boundingRect(cnt)

        # Slightly expand the ROI area
        roi = image[max(0, y-5):min(image.shape[0], y+h+5),
                    max(0, x-5):min(image.shape[1], x+w+5)]

        # Reapply thresholding within the ROI
        _, roi_thresh = cv2.threshold(
            roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find the largest contour within the ROI
        roi_contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

        if roi_contours:
            # Choose the largest contour
            max_cnt = max(roi_contours, key=cv2.contourArea)
            # Adjust coordinates back to the original image
            max_cnt = max_cnt + \
                np.array([max(0, x-5), max(0, y-5)])[None, None, :]
            final_contours.append(max_cnt)

    return final_contours


def analyze_band_intensity(image, contours):
    """Improved band intensity analysis function"""
    results = []

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        # Expand ROI to include surrounding background
        roi_y1 = max(0, y - 10)
        roi_y2 = min(image.shape[0], y + h + 10)
        roi_x1 = max(0, x - 10)
        roi_x2 = min(image.shape[1], x + w + 10)

        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]

        # Create a mask
        mask = np.zeros_like(roi)
        roi_cnt = cnt - np.array([roi_x1, roi_y1])[None, None, :]
        cv2.drawContours(mask, [roi_cnt], -1, 255, -1)

        # Calculate background (using the edge region of the ROI)
        bg_mask = cv2.bitwise_not(mask)
        background = np.median(roi[bg_mask > 0])

        # Calculate band intensity
        band_pixels = roi[mask > 0]
        mean_intensity = np.mean(band_pixels) - background

        # Adjust numerical scale
        mean_intensity = (mean_intensity / 100)  # Divide intensity by 100
        area = cv2.contourArea(cnt) / 100  # Divide area by 100
        total_intensity = mean_intensity * len(band_pixels)

        # Calculate integrated density
        integrated_density = total_intensity * area

        results.append({
            'band_number': i+1,
            'area': area,
            'mean_intensity': mean_intensity,
            'total_intensity': total_intensity,
            'integrated_density': integrated_density
        })

    return results


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

    # Advanced settings expander
    advanced_settings = st.sidebar.expander(
        "Advanced Settings", expanded=False)
    with advanced_settings:
        min_distance = st.slider("Minimum Distance Between Contours", 1, 10, 2)
        min_area = st.slider("Minimum Contour Area", 10, 100, 50)
        clip_limit = st.slider("CLAHE Clip Limit", 1.0, 4.0, 2.0)
        tile_grid_size = st.slider("CLAHE Tile Grid Size", 1, 16, 8)

    # Construct proper path with os.path.join
    image_path = os.path.join(script_dir, "pics", selected_image)
    image = safe_read_image(image_path)

    if image is None:
        return

    if image is not None:
        # Add original image information display
        st.sidebar.subheader("Image Information")
        st.sidebar.text(f"Original Size: {image.shape[1]}x{image.shape[0]}")

        # Process the image
        processed, standardized = process_image(
            image, clip_limit, tile_grid_size)
        bands = detect_bands(processed, min_distance, min_area)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(standardized, channels="BGR", use_container_width=True)
            st.caption(
                f"Standardized Size: {standardized.shape[1]}x{standardized.shape[0]}")

        # Draw detection results
        result = standardized.copy()
        for i, cnt in enumerate(bands):
            # Draw contours
            cv2.drawContours(result, [cnt], -1, (0, 255, 255), 2)

            # Draw bounding boxes and labels
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 255), 1)
            cv2.putText(result, f"#{i+1}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        with col2:
            st.subheader("Detected Bands")
            st.image(result, channels="BGR", use_container_width=True)

        # Analysis results
        if len(bands) > 0:
            st.subheader("Band Analysis Results")
            analysis_results = analyze_band_intensity(processed, bands)

            # Create DataFrame
            df = pd.DataFrame(analysis_results)

            # Add relative integrated density
            max_density = df['integrated_density'].max()
            df['relative_density'] = (
                df['integrated_density'] / max_density * 100)

            # Set display columns and round to two decimal places
            display_cols = [
                'band_number',
                'area',
                'mean_intensity',
                'integrated_density',
                'relative_density'
            ]

            # Round all numerical columns to two decimal places
            df = df.round(2)

            st.dataframe(df[display_cols])

            # Add integrated density distribution plot
            st.subheader("Integrated Density Distribution")
            fig = go.Figure()

            # Add integrated density bar chart
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
