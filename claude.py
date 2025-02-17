import streamlit as st
import subprocess
import sys
import os
import time
import wget
import zipfile
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import uuid
import csv

def install_package(package):
    """Install a single package with pip and handle errors"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error installing {package}: {str(e)}")
        return False

def setup_environment():
    """Setup the complete environment including packages and Object Detection API"""
    # Install required packages
    packages = [
        'tensorflow==2.4.1',
        'tensorflow-gpu==2.4.1',
        'opencv-python',
        'easyocr',
        'pillow',
        'numpy',
        'wget',
        'protobuf==3.20.0',
        'matplotlib==3.2'
    ]
    
    for package in packages:
        if not install_package(package):
            st.error(f"Failed to install {package}")
            return False

    try:
        # Clone TensorFlow models repository if it doesn't exist
        if not os.path.exists('models'):
            subprocess.check_call(['git', 'clone', 'https://github.com/tensorflow/models.git'])
        
        # Setup protoc
        if os.name == 'nt':  # Windows
            if not os.path.exists('protoc'):
                os.makedirs('protoc')
            
            # Download protoc
            protoc_url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
            protoc_path = os.path.join('protoc', 'protoc-3.15.6-win64.zip')
            
            if not os.path.exists(protoc_path):
                wget.download(protoc_url, protoc_path)
            
            with zipfile.ZipFile(protoc_path, 'r') as zip_ref:
                zip_ref.extractall('protoc')
            
            os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join('protoc', 'bin'))
        
        # Compile protobufs
        os.chdir('models/research')
        subprocess.check_call(['protoc', 'object_detection/protos/*.proto', '--python_out=.'])
        
        # Install Object Detection API
        subprocess.check_call([sys.executable, 'setup.py', 'build'])
        subprocess.check_call([sys.executable, 'setup.py', 'install'])
        
        os.chdir('../..')
        return True
        
    except Exception as e:
        st.error(f"Error during setup: {str(e)}")
        return False

def initialize_model():
    """Initialize and return the detection model"""
    # Constants
    CUSTOM_MODEL_NAME = 'my_ssd_mobnet'
    PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
    
    # Setup paths
    paths = {
        'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
        'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
        'APIMODEL_PATH': os.path.join('Tensorflow','models'),
        'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
        'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
        'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME),
    }
    
    files = {
        'PIPELINE_CONFIG': os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], 'label_map.pbtxt')
    }
    
    # Create directories
    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path)
    
    # Initialize model
    try:
        from object_detection.utils import label_map_util
        from object_detection.utils import visualization_utils as viz_utils
        from object_detection.builders import model_builder
        from object_detection.utils import config_util
        
        # Prevent GPU memory consumption
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
            except RuntimeError as e:
                st.error(f"GPU configuration error: {e}")
        
        # Load model config and build detection model
        configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
        detection_model = model_builder.build(model_config=configs['model'], is_training=False)
        
        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
        ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()
        
        return detection_model, label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    
    except Exception as e:
        st.error(f"Error initializing model: {e}")
        return None, None

@tf.function
def detect_fn(image, detection_model):
    """Detect objects in image"""
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def process_image(image_np, detection_model, category_index):
    """Process image and return detections"""
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor, detection_model)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                 for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
    image_np_with_detections = image_np.copy()
    
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes']+1,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.8,
        agnostic_mode=False
    )
    
    return image_np_with_detections, detections

def perform_ocr(image, detections, detection_threshold=0.7, region_threshold=0.6):
    """Perform OCR on detected regions"""
    import easyocr
    
    scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
    boxes = detections['detection_boxes'][:len(scores)]
    
    width = image.shape[1]
    height = image.shape[0]
    
    for idx, box in enumerate(boxes):
        roi = box*[height, width, height, width]
        region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
        reader = easyocr.Reader(['en'])
        ocr_result = reader.readtext(region)
        
        # Filter text based on region size
        rectangle_size = region.shape[0]*region.shape[1]
        plate = []
        for result in ocr_result:
            length = np.sum(np.subtract(result[0][1], result[0][0]))
            height = np.sum(np.subtract(result[0][2], result[0][1]))
            if length*height / rectangle_size > region_threshold:
                plate.append(result[1])
        
        if plate:
            return plate, region
    
    return None, None

def main():
    st.title("License Plate Detection and Recognition")
    
    # Setup state
    if 'setup_complete' not in st.session_state:
        st.session_state.setup_complete = False
    
    # Perform setup if needed
    if not st.session_state.setup_complete:
        with st.spinner("Setting up environment... This may take a few minutes."):
            if setup_environment():
                st.session_state.setup_complete = True
                st.success("Setup completed successfully!")
            else:
                st.error("Setup failed. Please check the error messages above.")
                return
    
    # Initialize model
    detection_model, category_index = initialize_model()
    if detection_model is None:
        st.error("Failed to initialize model. Please ensure setup completed successfully.")
        return
    
    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process button
        if st.button("Detect License Plate"):
            with st.spinner("Processing image..."):
                try:
                    # Convert image and detect
                    image_np = np.array(image)
                    processed_image, detections = process_image(image_np, detection_model, category_index)
                    
                    # Display processed image
                    st.image(processed_image, caption="Detected License Plate", use_column_width=True)
                    
                    # Perform OCR
                    text, region = perform_ocr(processed_image, detections)
                    if text:
                        st.success(f"Detected Text: {text}")
                        
                        # Save results
                        if not os.path.exists('Detection_Images'):
                            os.makedirs('Detection_Images')
                        
                        img_name = f'{uuid.uuid1()}.jpg'
                        cv2.imwrite(os.path.join('Detection_Images', img_name), region)
                        
                        with open('detection_results.csv', mode='a', newline='') as f:
                            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            csv_writer.writerow([img_name, text])
                        
                        st.info("Results saved to detection_results.csv")
                    else:
                        st.warning("No text detected in the license plate region")
                
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()