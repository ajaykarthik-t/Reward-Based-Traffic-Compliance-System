import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import uuid
import easyocr
import pandas as pd
from datetime import datetime
import tempfile
from streamlit_webrtc import webrtc_streamer
import av
from ultralytics import YOLO
import csv
import random
import string
import names

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Constants and Configurations
TEMP_DIR = tempfile.mkdtemp()
FOLDER_PATH = "./detections/"
CSV_PATH = "./data/vehicle_records.csv"
LICENSE_MODEL_PATH = './models/license_plate_detector.pt'
HELMET_MODEL_PATH = './models/yolov8n.pt'

# Point system configuration
POINTS_FOR_HELMET = 5
POINTS_FOR_NO_HELMET = -10
INITIAL_POINTS = 100

# Create necessary directories
os.makedirs(FOLDER_PATH, exist_ok=True)
os.makedirs("./data", exist_ok=True)

# Create CSV file if it doesn't exist
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['timestamp', 'license_plate', 'username', 'points', 'confidence', 'violation_type', 'image_path'])

def generate_username(license_plate):
    """Generate a random username based on license plate"""
    # Use the license plate as a seed for consistent username generation
    random.seed(license_plate)
    return names.get_full_name()

def load_models():
    """
    Load YOLO models for license plate and helmet detection
    Returns:
        tuple: (license_plate_detector, helmet_detector)
    """
    try:
        # Load license plate detection model
        license_detector = YOLO(LICENSE_MODEL_PATH)
        
        # Load helmet detection model
        helmet_detector = YOLO(HELMET_MODEL_PATH)
        
        return license_detector, helmet_detector
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        raise Exception("Failed to load detection models")

def ensure_rgb(image):
    """
    Ensure the image is in RGB format
    Args:
        image: Input image
    Returns:
        numpy.ndarray: RGB image
    """
    if len(image.shape) == 2:  # Grayscale
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image

def read_license_plate(plate_crop):
    """
    Read text from license plate image using EasyOCR
    Args:
        plate_crop: Cropped image containing license plate
    Returns:
        tuple: (plate_text, confidence_score)
    """
    try:
        results = reader.readtext(plate_crop)
        if results:
            # Get the result with highest confidence
            best_result = max(results, key=lambda x: x[2])
            plate_text = best_result[1].upper().replace(' ', '')  # Standardize format
            return plate_text, best_result[2]  # text and confidence
        return None, 0.0
    except Exception as e:
        st.error(f"Error reading license plate: {str(e)}")
        return None, 0.0

def update_points(license_plate, has_helmet):
    """
    Update points for a license plate based on helmet detection
    Args:
        license_plate: The detected license plate number
        has_helmet: Boolean indicating if helmet was detected
    Returns:
        int: Updated points for this license plate
    """
    df = load_data()
    
    # Check if license plate exists
    if license_plate in df['license_plate'].values:
        # Get current points
        current_points = df.loc[df['license_plate'] == license_plate, 'points'].iloc[0]
        
        # Update points based on helmet detection
        if has_helmet:
            new_points = current_points + POINTS_FOR_HELMET
        else:
            new_points = max(0, current_points + POINTS_FOR_NO_HELMET)  # Don't go below zero
        
        # Update the points in the DataFrame
        df.loc[df['license_plate'] == license_plate, 'points'] = new_points
        df.to_csv(CSV_PATH, index=False)
        
        return new_points
    else:
        # New license plate - start with initial points
        if has_helmet:
            return INITIAL_POINTS + POINTS_FOR_HELMET
        else:
            return max(0, INITIAL_POINTS + POINTS_FOR_NO_HELMET)

def save_to_csv(data):
    """
    Save detection data to CSV file, ensuring unique license plates
    """
    df = load_data()
    
    # Extract the new data
    timestamp, license_plate, confidence, violation_type, img_path = data
    
    # If the license plate already exists, update the entry
    if license_plate in df['license_plate'].values:
        # Get current points and username
        username = df.loc[df['license_plate'] == license_plate, 'username'].iloc[0]
        
        # Update points based on violation type
        has_helmet = "No Helmet" not in violation_type
        points = update_points(license_plate, has_helmet)
        
        # Update the most recent timestamp and image
        df.loc[df['license_plate'] == license_plate, 'timestamp'] = timestamp
        df.loc[df['license_plate'] == license_plate, 'confidence'] = confidence
        df.loc[df['license_plate'] == license_plate, 'violation_type'] = violation_type
        df.loc[df['license_plate'] == license_plate, 'image_path'] = img_path
        df.loc[df['license_plate'] == license_plate, 'points'] = points
    else:
        # Generate a username for the new license plate
        username = generate_username(license_plate)
        
        # Calculate initial points based on violation type
        has_helmet = "No Helmet" not in violation_type
        points = update_points(license_plate, has_helmet)
        
        # Create a new row
        new_row = pd.DataFrame({
            'timestamp': [timestamp],
            'license_plate': [license_plate],
            'username': [username],
            'points': [points],
            'confidence': [confidence],
            'violation_type': [violation_type],
            'image_path': [img_path]
        })
        df = pd.concat([df, new_row], ignore_index=True)
    
    # Save the updated DataFrame
    df.to_csv(CSV_PATH, index=False)

def load_data():
    """Load data from CSV file"""
    try:
        return pd.read_csv(CSV_PATH)
    except:
        return pd.DataFrame(columns=['timestamp', 'license_plate', 'username', 'points', 'confidence', 'violation_type', 'image_path'])

def process_frame(img, license_detector, helmet_detector):
    """Process a single frame for helmet detection and license plate extraction with clear UI indicators"""
    img = ensure_rgb(img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    detections = []
    
    try:
        # Copy image for UI overlay
        ui_overlay = img_bgr.copy()
        
        # Step 1: Detect motorcycles and riders
        helmet_results = helmet_detector(img_bgr)[0]
        
        motorcycle_boxes = []
        rider_boxes = []
        helmet_boxes = []
        
        # Extract all detections by class
        for detection in helmet_results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) == 3:  # motorcycle
                motorcycle_boxes.append({
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'score': score
                })
            elif int(class_id) == 0:  # person
                rider_boxes.append({
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'score': score
                })
            elif int(class_id) == 2:  # helmet
                helmet_boxes.append({
                    'box': (int(x1), int(y1), int(x2), int(y2)),
                    'score': score
                })
        
        # Step 2: Associate riders with motorcycles
        riders_on_motorcycles = []
        
        for rider in rider_boxes:
            rider_box = rider['box']
            
            # Find if this rider is on any motorcycle
            for motorcycle in motorcycle_boxes:
                moto_box = motorcycle['box']
                
                if do_boxes_overlap(rider_box, moto_box, threshold=0.1):
                    riders_on_motorcycles.append({
                        'rider_box': rider_box,
                        'motorcycle_box': moto_box,
                        'has_helmet': False,  # Default to no helmet
                        'license_plate': None
                    })
                    break
        
        # Step 3: Check if riders have helmets
        for rider_data in riders_on_motorcycles:
            rider_box = rider_data['rider_box']
            
            # Approximate head area (top portion of rider box)
            head_height = (rider_box[3] - rider_box[1]) // 3  # Top third of rider
            head_box = (rider_box[0], rider_box[1], rider_box[2], rider_box[1] + head_height)
            
            # Check if any helmet overlaps with this rider's head
            for helmet in helmet_boxes:
                helmet_box = helmet['box']
                
                if do_boxes_overlap(head_box, helmet_box, threshold=0.1):
                    rider_data['has_helmet'] = True
                    break
        
        # Step 4: Detect license plates
        license_results = license_detector(img_bgr)[0]
        
        for detection in license_results.boxes.data.tolist():
            x1, y1, x2, y2, score, _ = detection
            license_box = (int(x1), int(y1), int(x2), int(y2))
            
            plate_crop = img_bgr[int(y1):int(y2), int(x1):int(x2)]
            if plate_crop.size > 0:
                plate_text, text_score = read_license_plate(plate_crop)
                
                if plate_text and text_score > 0.5:  # Only accept reasonably confident reads
                    # Find closest motorcycle and associate license plate
                    closest_rider = None
                    min_distance = float('inf')
                    
                    for rider_data in riders_on_motorcycles:
                        moto_box = rider_data['motorcycle_box']
                        moto_center = ((moto_box[0] + moto_box[2]) // 2, (moto_box[1] + moto_box[3]) // 2)
                        license_center = ((license_box[0] + license_box[2]) // 2, (license_box[1] + license_box[3]) // 2)
                        
                        # Calculate distance
                        distance = ((moto_center[0] - license_center[0])**2 + 
                                    (moto_center[1] - license_center[1])**2)**0.5
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_rider = rider_data
                    
                    # If the closest motorcycle is reasonably close, associate the license plate
                    if closest_rider and min_distance < 300:
                        closest_rider['license_plate'] = plate_text
        
        # Step 5: Create UI overlay with clear indicators
        for rider_data in riders_on_motorcycles:
            # Extract data
            rider_box = rider_data['rider_box']
            moto_box = rider_data['motorcycle_box']
            has_helmet = rider_data['has_helmet']
            license_plate = rider_data['license_plate']
            
            # Create a clear UI indicator box around the motorcycle and rider
            x_min = min(rider_box[0], moto_box[0])
            y_min = min(rider_box[1], moto_box[1])
            x_max = max(rider_box[2], moto_box[2])
            y_max = max(rider_box[3], moto_box[3])
            
            # Draw a prominent box around the entire motorcycle+rider
            cv2.rectangle(ui_overlay, (x_min-10, y_min-10), (x_max+10, y_max+10), 
                         (0, 0, 255) if not has_helmet else (0, 255, 0), 3)
            
            # Create text for UI display
            helmet_status = "HELMET: YES ✓" if has_helmet else "HELMET: NO ⚠"
            plate_text = f"PLATE: {license_plate}" if license_plate else "PLATE: Not detected"
            
            # Display helmet status in a prominent UI element
            # Create a semi-transparent background for text
            status_bg = ui_overlay.copy()
            cv2.rectangle(status_bg, (x_min-10, y_min-45), (x_min + 200, y_min-10), 
                         (0, 0, 0), -1)
            ui_overlay = cv2.addWeighted(status_bg, 0.7, ui_overlay, 0.3, 0)
            
            # Add helmet status text
            cv2.putText(ui_overlay, helmet_status, (x_min, y_min-25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if has_helmet else (0, 0, 255), 2)
            
            # Add license plate text
            plate_color = (255, 255, 255) if license_plate else (100, 100, 100)
            cv2.putText(ui_overlay, plate_text, (x_min, y_min-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, plate_color, 2)
            
            # If license plate is detected, highlight it in the image
            if license_plate:
                # Find the license plate box again to highlight it
                for detection in license_results.boxes.data.tolist():
                    x1, y1, x2, y2, score, _ = detection
                    plate_crop = img_bgr[int(y1):int(y2), int(x1):int(x2)]
                    if plate_crop.size > 0:
                        detected_text, _ = read_license_plate(plate_crop)
                        if detected_text == license_plate:
                            # Highlight the license plate
                            cv2.rectangle(ui_overlay, (int(x1), int(y1)), (int(x2), int(y2)), 
                                         (255, 255, 0), 2)
                            # Add a magnified view of the license plate
                            magnified_plate = cv2.resize(plate_crop, (0, 0), fx=2, fy=2)
                            h, w = magnified_plate.shape[:2]
                            # Place magnified view at the bottom right of the detection area
                            roi = ui_overlay[y_max:y_max+h, x_max-w:x_max]
                            if roi.shape[:2] == magnified_plate.shape[:2]:
                                ui_overlay[y_max:y_max+h, x_max-w:x_max] = magnified_plate
            
            # Save detection data for reporting
            points = update_points(license_plate, has_helmet) if license_plate else 0
            violation_type = 'Compliant - Helmet Detected' if has_helmet else 'Violation - No Helmet'
            
            if license_plate:
                # Save the license plate image if available
                plate_img_name = f"plate_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
                plate_img_path = os.path.join(FOLDER_PATH, plate_img_name)
                
                # Find the plate image and save it
                for detection in license_results.boxes.data.tolist():
                    x1, y1, x2, y2, score, _ = detection
                    plate_crop = img_bgr[int(y1):int(y2), int(x1):int(x2)]
                    detected_text, _ = read_license_plate(plate_crop)
                    if detected_text == license_plate and plate_crop.size > 0:
                        cv2.imwrite(plate_img_path, plate_crop)
                        break
                
                # Save to CSV
                save_to_csv([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    license_plate,
                    float(text_score) if 'text_score' in locals() else 0.9,
                    violation_type,
                    plate_img_path if 'plate_img_path' in locals() else ""
                ])
                
                # Add to detections for reporting
                detections.append({
                    'license_plate': license_plate,
                    'has_helmet': has_helmet,
                    'points': points,
                    'violation_type': violation_type
                })
        
        return cv2.cvtColor(ui_overlay, cv2.COLOR_BGR2RGB), detections
    
    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")
        traceback.print_exc()  # Print the full stack trace for debugging
        return img, []

def do_boxes_overlap(box1, box2, threshold=0.3):
    """
    Check if two bounding boxes overlap
    Args:
        box1, box2: Bounding boxes in format (x1, y1, x2, y2)
        threshold: Minimum overlap ratio to consider as overlapping
    Returns:
        bool: True if boxes overlap, False otherwise
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x1 < x2 and y1 < y2:
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Check if the intersection is significant enough
        if intersection / min(area1, area2) > threshold:
            return True
    
    return False

def find_closest_motorcycle(plate_box, motorcycles, max_distance=300):
    """
    Find the closest motorcycle to a license plate
    Args:
        plate_box: License plate bounding box (x1, y1, x2, y2)
        motorcycles: List of detected motorcycles
        max_distance: Maximum distance to consider for association
    Returns:
        dict or None: The closest motorcycle or None if none are close enough
    """
    plate_center = ((plate_box[0] + plate_box[2]) // 2, (plate_box[1] + plate_box[3]) // 2)
    
    min_distance = max_distance
    closest_motorcycle = None
    
    for motorcycle in motorcycles:
        moto_box = motorcycle['box']
        moto_center = ((moto_box[0] + moto_box[2]) // 2, (moto_box[1] + moto_box[3]) // 2)
        
        # Calculate Euclidean distance between centers
        distance = ((plate_center[0] - moto_center[0])**2 + 
                    (plate_center[1] - moto_center[1])**2)**0.5
        
        if distance < min_distance:
            min_distance = distance
            closest_motorcycle = motorcycle
    
    return closest_motorcycle

class VideoProcessor:
    def __init__(self):
        self.license_detector = YOLO(LICENSE_MODEL_PATH)
        self.helmet_detector = YOLO(HELMET_MODEL_PATH)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img, detections = process_frame(img, self.license_detector, self.helmet_detector)
        return av.VideoFrame.from_ndarray(processed_img, format="rgb24")

def create_dashboard():
    """Create a dashboard with detection statistics and leaderboard"""
    df = load_data()
    
    if len(df) == 0:
        st.info("No data available yet. Start detection to collect data.")
        return
    
    # Dashboard metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Unique Vehicles", df['license_plate'].nunique())
    with col2:
        compliance_rate = (df['violation_type'].str.contains('Compliant').sum() / len(df)) * 100
        st.metric("Compliance Rate", f"{compliance_rate:.1f}%")
    with col3:
        st.metric("Today's Detections", 
                 len(df[df['timestamp'].str.contains(datetime.now().strftime("%Y-%m-%d"))]))
    
    # Leaderboard
    st.subheader("🏆 Points Leaderboard")
    leaderboard = df.sort_values('points', ascending=False).drop_duplicates('license_plate')
    leaderboard_display = leaderboard[['username', 'license_plate', 'points']].reset_index(drop=True)
    leaderboard_display.index = leaderboard_display.index + 1  # Start from 1 instead of 0
    st.dataframe(leaderboard_display, use_container_width=True)
    
    # Recent detections
    st.subheader("Recent Detections")
    recent_df = df.sort_values('timestamp', ascending=False).head(10)
    st.dataframe(recent_df[['timestamp', 'username', 'license_plate', 'points', 'violation_type']], 
                use_container_width=True)
    
    # Compliance over time
    st.subheader("Compliance Timeline")
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    df['is_compliant'] = df['violation_type'].str.contains('Compliant')
    
    timeline_data = df.groupby(['date', 'is_compliant']).size().unstack(fill_value=0)
    if not timeline_data.empty and len(timeline_data.columns) > 1:
        timeline_data.columns = ['Non-Compliant', 'Compliant']
        timeline_data_norm = timeline_data.divide(timeline_data.sum(axis=1), axis=0) * 100
        
        st.area_chart(timeline_data_norm)
    
    # License plate search
    st.subheader("🔍 Search Vehicle")
    search_plate = st.text_input("Enter license plate number:")
    if search_plate:
        results = df[df['license_plate'].str.contains(search_plate.upper(), na=False)]
        if not results.empty:
            st.dataframe(results[['timestamp', 'username', 'license_plate', 'points', 'violation_type']], 
                        use_container_width=True)
            
            # Show the most recent image for this license plate
            most_recent = results.iloc[0]
            if os.path.exists(most_recent['image_path']):
                st.image(most_recent['image_path'], caption=f"Most recent detection of {most_recent['license_plate']}")
        else:
            st.info("No matching records found.")

def main():
    st.set_page_config(page_title="Reward-Based Traffic Compliance System", layout="wide")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Live Detection", "Leaderboard & Stats"])
    
    if page == "Live Detection":
        st.title("🚗 Reward-Based Traffic Compliance System")
        st.subheader("Detection & Reward Tracking")
        
        # Load models
        license_detector, helmet_detector = load_models()
        
        # Mode selection
        mode = st.radio("Select Detection Mode:", 
                        ["Upload Image", "Camera", "Live Stream"], 
                        horizontal=True)
        
        if mode == "Upload Image":
            img_file = st.file_uploader("Upload an Image:", type=["jpg", "jpeg", "png"])
            if img_file:
                image = np.array(Image.open(img_file).convert('RGB'))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original Image")
                
                if st.button("Detect & Award Points"):
                    processed_img, detections = process_frame(image.copy(), license_detector, helmet_detector)
                    with col2:
                        st.image(processed_img, caption="Processed Image")
                    
                    # Display results summary
                    if detections:
                        st.success(f"Processed {len(detections)} vehicle(s)")
                        for detection in detections:
                            status = "✅ COMPLIANT" if detection['has_helmet'] else "⚠️ NON-COMPLIANT"
                            points_msg = f"+{POINTS_FOR_HELMET}" if detection['has_helmet'] else f"{POINTS_FOR_NO_HELMET}"
                            st.info(f"{status} | Plate: {detection['license_plate']} | Points: {detection['points']} ({points_msg})")
                    else:
                        st.info("No vehicles or license plates detected.")
        
        elif mode == "Camera":
            img_file = st.camera_input("Take a Photo")
            if img_file:
                image = np.array(Image.open(img_file).convert('RGB'))
                processed_img, detections = process_frame(image.copy(), license_detector, helmet_detector)
                st.image(processed_img, caption="Processed Image")
                
                # Display results summary
                if detections:
                    st.success(f"Processed {len(detections)} vehicle(s)")
                    for detection in detections:
                        status = "✅ COMPLIANT" if detection['has_helmet'] else "⚠️ NON-COMPLIANT"
                        points_msg = f"+{POINTS_FOR_HELMET}" if detection['has_helmet'] else f"{POINTS_FOR_NO_HELMET}"
                        st.info(f"{status} | Plate: {detection['license_plate']} | Points: {detection['points']} ({points_msg})")
                else:
                    st.info("No vehicles or license plates detected.")
        
        else:  # Live Stream
            st.warning("Live stream mode will process frames continuously and update the point system in real-time.")
            webrtc_streamer(
                key="traffic_monitor",
                video_processor_factory=VideoProcessor,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
    
    else:  # Leaderboard & Stats page
        st.title("📊 Traffic Compliance Leaderboard")
        create_dashboard()

if __name__ == "__main__":
    main()