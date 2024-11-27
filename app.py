import streamlit as st
import cv2
import numpy as np
import os
from pathlib import Path
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks

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
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(resized, -1, kernel)
    
    # Denoise while preserving edges
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, 10, 10, 7, 21)
    
    # Adjust contrast and brightness
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced

def process_image(image):
    """Main function for processing the image"""
    # First, standardize the image
    standardized = standardize_image(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(standardized, cv2.COLOR_BGR2GRAY)
    
    # Further image enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise the image
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    # Invert the image (to make bands white)
    inverted = cv2.bitwise_not(denoised)
    
    return inverted, standardized

def locate_band_cutoff_x(image):
    '''
    Determines all x-coordinates where the bands are located.
    E.g. [0, 100, 200] means there are 2 bands which should be cropped at 0-100 and 100-200.
    '''
    
    # Preprocess to reduce shadow effects
    blur = cv2.GaussianBlur(image, (5,5), 0)
    
    # Use Otsu thresholding to better separate bands and background
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to remove noise and shadows
    kernel_v = np.ones((15,1), np.uint8)  # Vertical kernel
    kernel_h = np.ones((1,5), np.uint8)   # Horizontal kernel
    
    # Open operation to remove small noise
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    # Close operation to fill gaps inside bands
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_v)
    
    # Compute the whiteness of each column
    column_whiteness = np.mean(closed, axis=0)
    
    # create an image with the same shape as the original image
    whiteness_avg_img = np.zeros_like(image)
    # but with the column whiteness values
    for i, whiteness in enumerate(column_whiteness):
        whiteness_avg_img[:, i] = whiteness
    # convert to 3 channel image
    whiteness_avg_img = cv2.cvtColor(whiteness_avg_img, cv2.COLOR_GRAY2BGR)
    
    return column_whiteness, whiteness_avg_img

def detect_bands(image):
    """Improved band detection function"""
    # Preprocess to reduce shadow effects
    blur = cv2.GaussianBlur(image, (5,5), 0)
    
    # Use Otsu thresholding to better separate bands and background
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to remove noise and shadows
    kernel_v = np.ones((15,1), np.uint8)  # Vertical kernel
    kernel_h = np.ones((1,5), np.uint8)   # Horizontal kernel
    
    # Open operation to remove small noise
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
    # Close operation to fill gaps inside bands
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_v)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Merge contours that are too close
    merged_contours = []
    min_distance = 2  # Minimum distance threshold
    
    # Sort contours by x-coordinate
    sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    current_contour = None
    for contour in sorted_contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        
        # Filter out contours that are too small (likely noise)
        if area < 50:  # Minimum area threshold
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
        _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find the largest contour within the ROI
        roi_contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
        
        if roi_contours:
            # Choose the largest contour
            max_cnt = max(roi_contours, key=cv2.contourArea)
            # Adjust coordinates back to the original image
            max_cnt = max_cnt + np.array([max(0, x-5), max(0, y-5)])[None, None, :]
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
        # Add original image information display
        st.sidebar.subheader("Image Information")
        st.sidebar.text(f"Original Size: {image.shape[1]}x{image.shape[0]}")
        
        # Process the image
        processed, standardized = process_image(image)
        bands = detect_bands(processed)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(standardized, channels="BGR", use_container_width=True)
            st.caption(f"Standardized Size: {standardized.shape[1]}x{standardized.shape[0]}")
        
        # Draw detection results
        result = standardized.copy()
        for i, cnt in enumerate(bands):
            # Draw contours
            cv2.drawContours(result, [cnt], -1, (0,255,255), 2)
            
            # Draw bounding boxes and labels
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x,y), (x+w,y+h), (0,255,255), 1)
            cv2.putText(result, f"#{i+1}", (x,y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        
        with col2:
            st.subheader("Detected Bands")
            st.image(result, channels="BGR", use_container_width=True)
        
        # Analysis results
        if len(bands) > 0:
            
            ''' Debugging Band Detection '''
            # Create subheader 
            st.subheader("Band Detection Step-by-Step")
            
            # Create a copy of the standardized image
            debug_image = processed.copy()
            
            blur = cv2.GaussianBlur(debug_image, (5,5), 0)
    
            # Use Otsu thresholding to better separate bands and background
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to remove noise and shadows
            kernel_v = np.ones((15,1), np.uint8)  # Vertical kernel
            kernel_h = np.ones((1,5), np.uint8)   # Horizontal kernel
            
            # Open operation to remove small noise
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
            # Close operation to fill gaps inside bands
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_v)
            
            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Display the thresholded image
            # st.image(thresh, channels="BGR", use_container_width=True)
            # st.subheader(str(type(opened)))
            # print(type(thresh))
            
            st.subheader("Inverted")
            
            st.image(debug_image, use_container_width=True)
            
            st.subheader("Blurred")
            
            st.image(blur, use_container_width=True)
            
            st.subheader("Thresholded")
            
            st.image(thresh, use_container_width=True)
            
            st.subheader("Opened")
            
            st.image(opened, use_container_width=True)
            
            st.subheader("Closed")
            
            st.image(closed, use_container_width=True)
            
            thresh_cnt = thresh.copy()
            # make the image 3 channel
            thresh_cnt = cv2.cvtColor(thresh_cnt, cv2.COLOR_GRAY2BGR)
            for i, cnt in enumerate(contours):
                # Draw contours
                cv2.drawContours(thresh_cnt, [cnt], -1, (0, 255, 255), 2)

                # Draw bounding boxes and labels
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(thresh_cnt, (x, y), (x+w, y+h), (0, 255, 255), 1)
                cv2.putText(thresh_cnt, f"#{i+1}", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
            st.subheader("Contours")
            st.subheader(f"Contours Count: {len(contours)}")
            st.image(thresh_cnt, channels="BGR", use_container_width=True)
            
            # Merge contours that are too close
            merged_contours = []
            min_distance = 2  # Minimum distance threshold

            # Sort contours by x-coordinate
            sorted_contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

            current_contour = None
            for contour in sorted_contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)

                # Filter out contours that are too small (likely noise)
                if area < 50:  # Minimum area threshold
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
                
            # draw merged contours
            merged_cnt = thresh.copy()
            # make the image 3 channel
            merged_cnt = cv2.cvtColor(merged_cnt, cv2.COLOR_GRAY2BGR)
            for i, cnt in enumerate(merged_contours):
                # Draw contours
                cv2.drawContours(merged_cnt, [cnt], -1, (0, 255, 255), 2)

                # Draw bounding boxes and labels
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(merged_cnt, (x, y), (x+w, y+h), (0, 255, 255), 1)
                cv2.putText(merged_cnt, f"#{i+1}", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
            st.subheader("Merged Contours")
            st.subheader(f"Merged Contours Count: {len(merged_contours)}")
            st.image(merged_cnt, channels="BGR", use_container_width=True)
            
            final_contours = []
            for cnt in merged_contours:
                # Get the bounding box of the contour
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Slightly expand the ROI area
                roi = debug_image[max(0, y-5):min(debug_image.shape[0], y+h+5),
                                  max(0, x-5):min(debug_image.shape[1], x+w+5)]
                
                # Reapply thresholding within the ROI
                _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Find the largest contour within the ROI
                roi_contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, 
                                                cv2.CHAIN_APPROX_SIMPLE)
                
                if roi_contours:
                    # Choose the largest contour
                    max_cnt = max(roi_contours, key=cv2.contourArea)
                    # Adjust coordinates back to the original image
                    max_cnt = max_cnt + np.array([max(0, x-5), max(0, y-5)])[None, None, :]
                    final_contours.append(max_cnt)
            
            # Final contours
            final_cnt = debug_image.copy()
            # make the image 3 channel
            final_cnt = cv2.cvtColor(final_cnt, cv2.COLOR_GRAY2BGR)
            for i, cnt in enumerate(final_contours):
                # Draw contours
                cv2.drawContours(final_cnt, [cnt], -1, (0, 255, 255), 2)

                # Draw bounding boxes and labels
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(final_cnt, (x, y), (x+w, y+h), (0, 255, 255), 1)
                cv2.putText(final_cnt, f"#{i+1}", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
            st.subheader("Final Contours")
            st.subheader(f"Final Contours Count: {len(final_contours)}")
            st.image(final_cnt, channels="BGR", use_container_width=True)
            
            
            ''' Novel Algorithm'''
            
            # Compute the whiteness of each column
            column_whiteness = np.mean(closed, axis=0)
            
            # Compute the delta of change per column
            column_delta = np.diff(column_whiteness)
            
            # (For debugging) Display the column whiteness
            # create an image with the same shape as the original image
            whiteness_avg_img = np.zeros_like(closed)
            # but with the column whiteness values
            for i, whiteness in enumerate(column_whiteness):
                whiteness_avg_img[:, i] = whiteness
            # convert to 3 channel image
            whiteness_avg_img = cv2.cvtColor(whiteness_avg_img, cv2.COLOR_GRAY2BGR)
            st.subheader("Column Whiteness")
            st.image(whiteness_avg_img, channels="BGR", use_container_width=True)
            
            # (For debugging) Display the column delta
            delta_avg_img = np.zeros_like(closed)
            # but with the column delta values
            for i, delta in enumerate(column_delta):
                delta_avg_img[:, i] = delta
            # convert to 3 channel image
            delta_avg_img = cv2.cvtColor(delta_avg_img, cv2.COLOR_GRAY2BGR)
            st.subheader("Column Delta")
            st.image(delta_avg_img, channels="BGR", use_container_width=True)
            
            
            # plot the column whiteness
            inverted_column_whiteness = -column_whiteness+(-min(-column_whiteness))
            fig = go.Figure()
            fig.add_trace(go.Scatter
                          (x=np.arange(len(column_whiteness)), y=inverted_column_whiteness, mode='lines'))
            fig.update_layout(width=standardized.shape[1], height=1000)
            st.plotly_chart(fig, use_container_width=True)
            


            def thresholding_algo(y, lag, threshold, influence):
                signals = np.zeros(len(y))
                filteredY = np.array(y)
                avgFilter = [0]*len(y)
                stdFilter = [0]*len(y)
                avgFilter[lag - 1] = np.mean(y[0:lag])
                stdFilter[lag - 1] = np.std(y[0:lag])
                for i in range(lag, len(y)):
                    if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter[i-1]:
                        if y[i] > avgFilter[i-1]:
                            signals[i] = 1
                        else:
                            signals[i] = -1

                        filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
                        avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
                        stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
                    else:
                        signals[i] = 0
                        filteredY[i] = y[i]
                        avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
                        stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

                return dict(signals=np.asarray(signals),
                            avgFilter=np.asarray(avgFilter),
                            stdFilter=np.asarray(stdFilter))
            
           
           
            # results = thresholding_algo(inverted_column_whiteness, ) 
           
            high_points, _ = find_peaks(inverted_column_whiteness, prominence=25)
        
            
            # append 0 and the last column index to the high points
            high_points = np.append(high_points, [0, len(column_whiteness)-1])
            high_points = np.sort(high_points)
            st.subheader("Band Separation Points")
            st.write(high_points)
            st.write(len(high_points)-1)
            
            # insert these points into the image as a pink line with width 5
            for i in range(len(high_points)-1):
                cv2.line(whiteness_avg_img, (high_points[i], 0), (high_points[i], whiteness_avg_img.shape[0]), (255, 0, 255), 5)
                
            st.image(whiteness_avg_img, channels="BGR", use_container_width=True)
            
            # insert into original image
            for i in range(len(high_points)-1):
                cv2.line(standardized, (high_points[i], 0), (high_points[i], standardized.shape[0]), (255, 0, 255), 5)
                
            st.subheader("Detected Bands")
            st.image(standardized, channels="BGR", use_container_width=True)
            

        
                
            ''' Original code'''
            
            # Draw the original contours
            
            st.subheader("Band Analysis Results")
            analysis_results = analyze_band_intensity(processed, bands)
            
            # Create DataFrame
            df = pd.DataFrame(analysis_results)
            
            # Add relative integrated density
            max_density = df['integrated_density'].max()
            df['relative_density'] = (df['integrated_density'] / max_density * 100)
            
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
