import subprocess
import sys
import os
import time

def install_package(package):
    """Install a single package with pip and handle errors"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}: {str(e)}")
        return False

def install_requirements():
    """Install required packages with proper error handling"""
    packages = [
        'streamlit',
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
    
    failed_packages = []
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        raise Exception(f"Failed to install: {', '.join(failed_packages)}")

def setup_object_detection():
    """Setup TensorFlow Object Detection API"""
    try:
        # Clone TensorFlow models repository if it doesn't exist
        if not os.path.exists('models'):
            subprocess.check_call(['git', 'clone', 'https://github.com/tensorflow/models.git'])
        
        # Install protobuf
        if os.name == 'nt':  # Windows
            if not os.path.exists('protoc'):
                os.makedirs('protoc')
            
            # Download protoc for Windows
            protoc_url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
            protoc_path = os.path.join('protoc', 'protoc-3.15.6-win64.zip')
            
            if not os.path.exists(protoc_path):
                import wget
                wget.download(protoc_url, protoc_path)
            
            # Extract protoc
            import zipfile
            with zipfile.ZipFile(protoc_path, 'r') as zip_ref:
                zip_ref.extractall('protoc')
            
            # Add protoc to PATH
            os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join('protoc', 'bin'))
        
        # Compile protobufs
        os.chdir('models/research')
        if os.name == 'nt':  # Windows
            subprocess.check_call(['protoc', 'object_detection/protos/*.proto', '--python_out=.'])
        else:  # Linux/Mac
            subprocess.check_call(['protoc', 'object_detection/protos/*.proto', '--python_out=.'])
        
        # Copy setup.py if needed
        if not os.path.exists('setup.py'):
            subprocess.check_call(['copy' if os.name == 'nt' else 'cp', 
                                 'object_detection/packages/tf2/setup.py', 'setup.py'])
        
        # Install Object Detection API
        subprocess.check_call([sys.executable, 'setup.py', 'build'])
        subprocess.check_call([sys.executable, 'setup.py', 'install'])
        
        os.chdir('../..')
        return True
        
    except Exception as e:
        print(f"Error during Object Detection API setup: {str(e)}")
        return False