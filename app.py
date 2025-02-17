import streamlit as st
import cv2
import numpy as np
import imutils
import easyocr
from PIL import Image
import matplotlib.pyplot as plt

st.title("OCR Image Processing")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply filter and find edges
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    
    # Display Edge Detection result
    st.image(edged, caption='Edge Detection', use_column_width=True, channels='GRAY')
    
    # Find Contours and Apply Mask
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break
    
    if location is not None:
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)
        
        # Crop the region of interest
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2+1, y1:y2+1]
        
        st.image(cropped_image, caption='Cropped Image', use_column_width=True, channels='GRAY')
        
        # Use EasyOCR to read text
        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)
        
        if result:
            text = result[0][-2]
            st.write(f"**Extracted Text:** {text}")
            
            # Render the result on the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1] + 60), 
                              fontFace=font, fontScale=1, color=(0, 255, 0), 
                              thickness=2, lineType=cv2.LINE_AA)
            res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)
            
            st.image(res, caption='Final Result with OCR Text', use_column_width=True)
        else:
            st.write("No text detected.")
