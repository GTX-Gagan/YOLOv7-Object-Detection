#!/usr/bin/env python3
"""
YOLOv7 Web Application with Detection API
"""
import os
import io
import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import cv2
import numpy as np
from models.experimental import attempt_load
from utils.general import (non_max_suppression, scale_coords, 
                           check_img_size, set_logging, increment_path)
from utils.datasets import letterbox
from utils.torch_utils import select_device, time_synchronized

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Configuration
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global model variables
model = None
device = None
img_size = 640
conf_thres = 0.25
iou_thres = 0.45
stride = 32

def initialize_model():
    """Initialize YOLOv7 model"""
    global model, device, stride
    set_logging()
    device = select_device('cpu')
    
    weights = './yolov7-tiny.pt'
    if not Path(weights).exists():
        return False, f"Model weights not found: {weights}"
    
    try:
        model = attempt_load(weights, map_location=device)
        stride = int(model.stride.max())
        model.eval()
        
        # Run once to warm up
        with torch.no_grad():
            model(torch.zeros(1, 3, img_size, img_size).to(device))
        
        return True, "Model loaded successfully"
    except Exception as e:
        return False, f"Error loading model: {str(e)}"

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """
    Process image with YOLOv7 detection
    Returns: (image_with_boxes, detections)
    """
    try:
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None, []
        
        im0 = img.copy()
        h0, w0 = img.shape[:2]
        
        # Letterbox
        img = letterbox(img, img_size, stride=stride)[0]
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        # Inference
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        with torch.no_grad():
            pred = model(img)[0]
        
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        
        detections = []
        
        # Process detections
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                # Draw boxes and collect detections
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = map(int, xyxy)
                    confidence = float(conf)
                    class_id = int(cls)
                    
                    # Draw rectangle
                    cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Put label
                    label = f'Object {class_id}: {confidence:.2f}'
                    cv2.putText(im0, label, (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    detections.append({
                        'class': class_id,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'width': x2 - x1,
                        'height': y2 - y1
                    })
        
        return im0, detections
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, []

def cv2_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode()
    return image_base64

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

@app.route('/api/detect', methods=['POST'])
def detect():
    """Main detection endpoint"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save uploaded file
        filename = Path(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))
        
        # Process image
        result_image, detections = process_image(filepath)
        
        if result_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Convert to base64
        image_base64 = cv2_to_base64(result_image)
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{image_base64}',
            'detections': detections,
            'count': len(detections)
        })
    
    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect-url', methods=['POST'])
def detect_from_url():
    """Detection endpoint accepting image URL or base64"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        image_data = data['image']
        
        # Handle base64 image
        if image_data.startswith('data:image'):
            # Remove data URI prefix
            image_data = image_data.split(',')[1]
        
        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Save temporarily
            temp_path = UPLOAD_FOLDER / 'temp_detect.png'
            image.save(str(temp_path))
            
            # Process image
            result_image, detections = process_image(temp_path)
            
            if result_image is None:
                return jsonify({'error': 'Failed to process image'}), 400
            
            # Convert to base64
            image_base64 = cv2_to_base64(result_image)
            
            return jsonify({
                'success': True,
                'image': f'data:image/jpeg;base64,{image_base64}',
                'detections': detections,
                'count': len(detections)
            })
        
        except Exception as e:
            return jsonify({'error': f'Invalid image data: {str(e)}'}), 400
    
    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/settings', methods=['GET', 'POST'])
def settings():
    """Get or update detection settings"""
    global conf_thres, iou_thres, img_size
    
    if request.method == 'GET':
        return jsonify({
            'confidence_threshold': conf_thres,
            'iou_threshold': iou_thres,
            'image_size': img_size
        })
    
    elif request.method == 'POST':
        data = request.get_json()
        
        if 'confidence_threshold' in data:
            conf_thres = float(data['confidence_threshold'])
        if 'iou_threshold' in data:
            iou_thres = float(data['iou_threshold'])
        
        return jsonify({
            'success': True,
            'confidence_threshold': conf_thres,
            'iou_threshold': iou_thres
        })

@app.route('/api/info', methods=['GET'])
def info():
    """Get model and system info"""
    return jsonify({
        'model': 'YOLOv7-Tiny',
        'device': str(device),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'image_size': img_size
    })

if __name__ == '__main__':
    print("Initializing YOLOv7 model...")
    success, message = initialize_model()
    print(message)
    
    if success:
        print("Starting Flask server on http://127.0.0.1:5000")
        app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)
    else:
        print("Failed to start server")
