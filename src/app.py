from flask import Flask, render_template, request, flash, redirect, url_for, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import cv2
import os
import json
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile app
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'  # Secret key for flash messages

# Define allowed extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the classification labels from a JSON file
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'labels.json'), 'r') as f:
    CLASSIFICATION_LABELS = json.load(f)

# Global variable for lazy-loaded model
_classification_model = None

def get_model():
    """Lazy load model - only loads when first prediction is requested"""
    global _classification_model
    
    if _classification_model is not None:
        return _classification_model
    
    print("Loading model for the first time...")
    
    # Import TensorFlow only when needed (saves startup memory)
    from tensorflow.keras.models import load_model
    
    # Path to the model is absolute to ensuring finding it
    local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models/flagship_model.keras')
    
    print(f"Loading model from: {local_model_path}")
    
    if os.path.exists(local_model_path):
        try:
            _classification_model = load_model(local_model_path)
            print("Model loaded successfully!")
            return _classification_model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model from {local_model_path}: {e}")
    else:
        print(f"ERROR: Model file not found at {local_model_path}")
        # As a fallback, we could try downloading, but for now we expect the file to be present
        # via git since we're pushing it.
        raise FileNotFoundError(f"Model file missing: {local_model_path}")

def allowed_file(filename):
    """
    Checks if a given filename has an allowed image extension.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """
    API root - returns API information and available endpoints.
    """
    return jsonify({
        'name': 'WheatGuard API',
        'version': '1.0.0',
        'description': 'Wheat Disease Detection API',
        'endpoints': {
            '/': 'API info (this page)',
            '/health': 'Health check',
            '/api/predict': 'POST - Upload image for prediction'
        },
        'status': 'ready',
        'model_loaded': _classification_model is not None
    })

@app.route('/health')
def health():
    """Health check endpoint for Render"""
    return jsonify({'status': 'healthy', 'model_loaded': _classification_model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image uploads, preprocesses the image, makes a prediction using the
    classification model, and displays the result.
    """
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    # Process the file if it exists and is allowed
    if file and allowed_file(file.filename):
        # Get model (lazy load)
        model = get_model()
        
        # Read the image file into a BytesIO object
        img = Image.open(io.BytesIO(file.read()))
        img_np = np.array(img)
        
        # Convert RGB image to BGR for OpenCV compatibility (if needed for other operations)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Preprocess the image for the classification model (Expected input: 260x260 for loaded model)
        img_resized_classification = cv2.resize(img_np, (260, 260))  # Resize to model's expected input
        img_reshaped_classification = np.reshape(img_resized_classification, (1, 260, 260, 3)) # Reshape for model input
        
        # EfficientNetV2B3 handles normalization internally (expects 0-255 inputs)
        # So we just pass the resized image directly
        img_preprocessed = img_reshaped_classification

        # Run the classification model to get predictions
        prediction = model.predict(img_preprocessed)
        label_index = np.argmax(prediction)  # Get the index of the highest probability class
        label = CLASSIFICATION_LABELS[label_index]  # Get the corresponding label string

        # Generate a unique filename for the output image using a timestamp
        timestamp = str(int(time.time()))
        output_image_filename = f'output_{timestamp}.jpg'
        # Define the path to save the output image in the static folder
        output_image_path = os.path.join('static', output_image_filename)
        # Save the processed image (original BGR version) to the static folder
        cv2.imwrite(output_image_path, img_bgr)

        # Cleanup old images (older than 1 hour)
        cleanup_old_images()

        # Render the result page with the predicted label and image path
        return render_template('result.html', image_path=output_image_filename, label=label, timestamp=timestamp)
    else:
        # Flash an error message for invalid file types and redirect to the index page
        flash('Invalid file type. Please upload an image (png, jpg, jpeg).')
        return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    JSON API endpoint for mobile app predictions.
    Returns: JSON with label and confidence
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Get model (lazy load)
            model = get_model()
            
            img = Image.open(io.BytesIO(file.read()))
            img_np = np.array(img)
            
            # Preprocess image for classification model (loaded model expects 260x260)
            img_resized = cv2.resize(img_np, (260, 260))
            img_reshaped = np.reshape(img_resized, (1, 260, 260, 3))
            
            # Run prediction
            prediction = model.predict(img_reshaped)
            label_index = np.argmax(prediction)
            label = CLASSIFICATION_LABELS[label_index]
            confidence = float(prediction[0][label_index])
            
            return jsonify({
                'label': label,
                'confidence': confidence
            })
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

def cleanup_old_images(folder='static', age_seconds=3600):
    """
    Removes files in the specified folder that are older than age_seconds.
    """
    try:
        current_time = time.time()
        folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder)
        for filename in os.listdir(folder_path):
            if filename.startswith('output_') and filename.endswith('.jpg'):
                file_path = os.path.join(folder_path, filename)
                file_creation_time = os.path.getmtime(file_path)
                if current_time - file_creation_time > age_seconds:
                    os.remove(file_path)
                    print(f"Deleted old image: {filename}")
    except Exception as e:
        print(f"Error cleaning up images: {e}")

if __name__ == '__main__':
    # Get the port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    # Run the Flask application
    app.run(host='0.0.0.0', port=port, debug=True)
