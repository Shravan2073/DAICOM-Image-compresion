from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from compression import compress_image, save_compressed_image
import cv2
from utils import log_status

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
COMPRESSED_FOLDER = 'static/compressed/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['COMPRESSED_FOLDER'] = COMPRESSED_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        log_status("No file part found in the request.")
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        log_status("No file selected.")
        return redirect(url_for('index'))
    
    if file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)
        log_status(f"File {file.filename} uploaded successfully.")
        
        # Read image and compress
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        log_status(f"Processing file {file.filename}...")
        compressed_important, compressed_background = compress_image(image)
        
        compressed_path = os.path.join(app.config['COMPRESSED_FOLDER'], 'compressed_' + file.filename)
        save_compressed_image(image, compressed_path)
        
        log_status(f"File {file.filename} compression completed. Returning compressed file.")
        return send_from_directory(app.config['COMPRESSED_FOLDER'], 'compressed_' + file.filename)

if __name__ == '__main__':
    app.run(debug=True)
