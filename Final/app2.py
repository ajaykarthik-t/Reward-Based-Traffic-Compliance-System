import streamlit as st
import numpy as np
from PIL import Image
import cv2
import easyocr
from ultralytics import YOLO
import os
import uuid

# Load YOLO model
LICENSE_MODEL_DETECTION_DIR = './models/license_plate_detector.pt'
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)
reader = easyocr.Reader(['en'], gpu=False)

st.title("License Plate Detection")

# Upload image
uploaded_image = st.file_uploader("Upload a Car Image", type=["png", "jpg", "jpeg"])

# Function to detect license plate and extract text
def detect_license_plate(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    detections = license_plate_detector(img)[0]
    plates = []
    
    for box in detections.boxes.data.tolist():
        x1, y1, x2, y2, _, _ = box
        license_plate_crop = img[int(y1):int(y2), int(x1): int(x2)]
        license_plate_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        text_detections = reader.readtext(license_plate_gray)
        texts = [result[1].upper() for result in text_detections]
        
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        plates.append((license_plate_crop, ' '.join(texts)))
    
    return image, plates

if uploaded_image:
    image = np.array(Image.open(uploaded_image))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Detect License Plate"):
        result_image, detected_plates = detect_license_plate(image)
        st.image(result_image, caption="Detected License Plate", use_column_width=True)
        
        for i, (plate_img, text) in enumerate(detected_plates):
            st.image(plate_img, caption=f"License Plate {i+1}", width=250)
            st.success(f"Detected Text: {text}")
