from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import uuid
from datetime import datetime
import threading
from werkzeug.utils import secure_filename
import cv2
from detection_service import DetectionService
from grouping_service import GroupingService
import logging
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

detection_service = DetectionService()
grouping_service = GroupingService()

request_queue = queue.Queue()
results_store = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image_async(request_id, image_path):
    try:
        # Step 1: Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not load image")
        
        # Step 2: Product Detection
        logger.info(f"Processing detection for request {request_id}")
        detections = detection_service.detect_products(image)
        
        # Step 3: Product Grouping
        logger.info(f"Processing grouping for request {request_id}")
        grouped_results = grouping_service.group_products(detections, image)
        
        # Step 4: Generate visualization
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{request_id}_output.jpg")
        visualization = grouping_service.visualize_results(image, grouped_results, output_path)
        
        # Step 5: Prepare response
        response_data = {
            'request_id': request_id,
            'status': 'completed',
            'detections': grouped_results,
            'visualization_path': output_path,
            'total_objects': len(grouped_results),
            'unique_groups': len(set([r['group_id'] for r in grouped_results])),
            'timestamp': datetime.now().isoformat()
        }
        
        results_store[request_id] = response_data
        logger.info(f"Processing completed for request {request_id}")
        
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {str(e)}")
        results_store[request_id] = {
            'request_id': request_id,
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

@app.route('/')
def index():
    """Serve the main interface"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_filename = f"{timestamp}_{request_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)
        
        # Start async processing
        thread = threading.Thread(target=process_image_async, args=(request_id, filepath))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'request_id': request_id,
            'status': 'processing',
            'message': 'Image uploaded successfully. Processing started.'
        }), 202
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/status/<request_id>', methods=['GET'])
def get_status(request_id):
    """Get processing status for a request"""
    if request_id in results_store:
        result = results_store[request_id].copy()
        # Remove visualization path from JSON response
        if 'visualization_path' in result:
            result['visualization_available'] = True
            del result['visualization_path']
        return jsonify(result)
    else:
        return jsonify({
            'request_id': request_id,
            'status': 'pending',
            'message': 'Request is still processing or not found'
        }), 202

@app.route('/result/<request_id>', methods=['GET'])
def get_result(request_id):
    """Get the complete result for a processed request"""
    if request_id not in results_store:
        return jsonify({'error': 'Request not found'}), 404
    
    result = results_store[request_id]
    if result.get('status') != 'completed':
        return jsonify({'error': 'Processing not completed yet'}), 202
    
    response_result = result.copy()
    if 'visualization_path' in response_result:
        del response_result['visualization_path']
    
    return jsonify(response_result)

@app.route('/visualization/<request_id>', methods=['GET'])
def get_visualization(request_id):
    """Get the visualization image for a request"""
    if request_id not in results_store:
        return jsonify({'error': 'Request not found'}), 404
    
    result = results_store[request_id]
    if 'visualization_path' not in result or not os.path.exists(result['visualization_path']):
        return jsonify({'error': 'Visualization not available'}), 404
    
    return send_file(result['visualization_path'], mimetype='image/jpeg')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)