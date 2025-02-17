import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
import re

class LicensePlateDetector:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
    
    def preprocess_image(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            
        return image

    def detect_plate(self, image):
        img = self.preprocess_image(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(bilateral, 30, 200)
        contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        plate_contour = None
        
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 2.0 <= aspect_ratio <= 5.5:
                    plate_contour = approx
                    break
        
        if plate_contour is not None:
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [plate_contour], 0, 255, -1)
            (x, y) = np.where(mask == 255)
            (x1, y1) = (np.min(y), np.min(x))
            (x2, y2) = (np.max(y), np.max(x))
            padding = 5
            plate_region = gray[max(y1-padding, 0):min(y2+padding, gray.shape[0]),
                              max(x1-padding, 0):min(x2+padding, gray.shape[1])]
            return plate_region, (x1, y1, x2, y2)
        return None, None

    def enhance_plate(self, plate_img):
        if plate_img.shape[1] < 200:
            plate_img = cv2.resize(plate_img, (200, int(200 * plate_img.shape[0] / plate_img.shape[1])))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        plate_img = clahe.apply(plate_img)
        _, plate_img = cv2.threshold(plate_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        plate_img = cv2.fastNlMeansDenoising(plate_img)
        return plate_img

    def read_plate(self, image):
        original_img = self.preprocess_image(image)
        plate_region, coords = self.detect_plate(original_img)
        
        if plate_region is not None:
            enhanced_plate = self.enhance_plate(plate_region)
            try:
                results = self.reader.readtext(enhanced_plate)
                if results:
                    plate_text = ""
                    max_conf = 0
                    for _, text, conf in results:
                        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                        if len(cleaned) >= 4 and conf > max_conf:
                            plate_text = cleaned
                            max_conf = conf
                    if plate_text:
                        x1, y1, x2, y2 = coords
                        cv2.rectangle(original_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(original_img, plate_text, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        return plate_text, original_img, max_conf
            except Exception as e:
                st.error(f"Error reading text: {str(e)}")
        return "No plate detected", original_img, 0.0

def main():
    st.set_page_config(page_title="License Plate Detector", layout="wide")
    st.title("🚗 License Plate Detection System")
    st.write("Upload a vehicle image to detect and read the license plate")
    
    @st.cache_resource
    def load_detector():
        return LicensePlateDetector()
    
    detector = load_detector()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file)
        col1.subheader("Original Image")
        col1.image(image, use_column_width=True)
        
        if col1.button("Detect License Plate"):
            with st.spinner("Processing..."):
                try:
                    plate_text, processed_image, confidence = detector.read_plate(image)
                    col2.subheader("Processed Image")
                    col2.image(processed_image, channels="BGR", use_column_width=True)
                    if plate_text != "No plate detected":
                        st.success(f"Detected License Plate: {plate_text}")
                        st.info(f"Confidence: {confidence:.2%}")
                    else:
                        st.warning("No license plate detected. Please try a different image.")
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
                    st.write("Please try a different image or check if the image is corrupted.")

if __name__ == "__main__":
    main()
