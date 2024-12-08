import os
import json
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import hashlib

# Initialize Flask app
app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure the folders exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(RESULT_FOLDER):
    os.makedirs(RESULT_FOLDER)

# Load YOLO model
model = YOLO('yolov10n.pt')

# Helper function to check valid file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to generate a unique color for each label
label_color_map = {}

def generate_color_for_label(label):
    """Generate a unique color for each label using hash."""
    if label not in label_color_map:
        hash_value = int(hashlib.md5(label.encode('utf-8')).hexdigest(), 16)
        color = ((hash_value >> 16) & 0xFF, (hash_value >> 8) & 0xFF, hash_value & 0xFF)
        label_color_map[label] = color
    return label_color_map[label]

def create_image_with_bboxes(img, json_path):
    img = np.array(img)  # Convert PIL image to numpy array (defaults to RGB)

    # Load predictions from JSON file
    with open(json_path, "r") as f:
        prediction_data = json.load(f)
    
    for item in prediction_data:
        label = item['label']
        score = item['score']
        box = item['bbox']
        
        # Generate a unique color for each label
        color = generate_color_for_label(label)
        
        xmin, ymin, xmax, ymax = map(int, box)
        
        # Draw the bounding box and label
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        img = cv2.putText(img, f"{label} {score:.2f}", 
                          (xmin, ymin - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Convert BGR (OpenCV format) back to RGB (PIL format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def save_predictions_as_json(results, img_path):
    # Get prediction data
    prediction_data = []
    boxes = results[0].boxes.xywh.cpu().numpy() 
    class_indices = results[0].boxes.cls.cpu().numpy().astype(int)
    labels = [results[0].names[idx] for idx in class_indices]  
    scores = results[0].boxes.conf.cpu().numpy()
    
    # Convert boxes from xywh format to xmin, ymin, xmax, ymax
    boxes = [((x - w / 2), (y - h / 2), (x + w / 2), (y + h / 2)) for x, y, w, h in boxes]
    
    for label, score, box in zip(labels, scores, boxes):
        prediction_data.append({
            "label": label,
            "score": float(score),  # Convert score to float
            "bbox": [float(coord) for coord in box]  # Convert bounding box coordinates to float
        })
    
    # Define where to save predictions.json
    result_dir = os.path.join('static', 'results')  # Define folder to save JSON file
    if not os.path.exists(result_dir):  # Make sure the directory exists
        os.makedirs(result_dir)  # Create directory if it doesn't exist
    
    # Define full path for predictions.json
    json_filename = "predictions.json"
    json_filepath = os.path.join(result_dir, json_filename)
    
    # Save the predictions as a JSON file
    with open(json_filepath, "w") as json_file:
        json.dump(prediction_data, json_file, indent=4)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the post request has the file part
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']
        
        if file.filename == '':
            return "No selected file"
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            # Predict objects in the image
            results = model(img_path)  # Predictions
            
            # Save predictions to JSON
            save_predictions_as_json(results, img_path)
            
            # Create an image with bounding boxes
            json_path = os.path.join('static', 'results', 'predictions.json')  # Corrected path for JSON file
            img = Image.open(img_path)
            img_with_bbox = create_image_with_bboxes(np.array(img), json_path)
            
            result_image_path = os.path.join(RESULT_FOLDER, f"result_{filename}")
            cv2.imwrite(result_image_path, img_with_bbox)  # Save result image with bounding boxes
            
            return render_template('index.html', filename=filename, result_image=result_image_path)
    return render_template('index.html', filename=None, result_image=None)

# Route to serve the images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
