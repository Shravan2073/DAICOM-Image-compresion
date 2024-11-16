import numpy as np
import cv2
from utils import log_status, update_progress, calculate_psnr, calculate_ssim, check_dicom_compliance

# Define constants
TOTAL_STEPS = 4
MAX_COMPRESSION_RATIO = 10
MIN_PSN_RATING = 45
MIN_SSIM = 0.95
TMAX = 1000  # Max allowable encoding/decoding time (milliseconds)

# List of allowed file types for upload
ALLOWED_FILE_TYPES = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}

# Step 1: Image Segmentation Function
def segment_image(image, threshold=200):
    log_status("Starting image segmentation...")
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    important_region = cv2.bitwise_and(image, image, mask=mask)
    background_region = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    update_progress(1, TOTAL_STEPS)
    return important_region, background_region

# Step 2: Run-Length Encoding (RLE)
def run_length_encoding(arr):
    log_status("Starting Run-Length Encoding (RLE)...")
    compressed = []
    count = 1
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1]:
            count += 1
        else:
            compressed.append((arr[i-1], count))
            count = 1
    compressed.append((arr[-1], count))
    update_progress(2, TOTAL_STEPS)
    return compressed

# Step 3: Huffman Encoding (for the important region)
def huffman_encoding(data):
    log_status("Starting Huffman Encoding...")
    # Huffman encoding implementation here (this is a placeholder)
    encoded_data = "encoded_data"  # Placeholder for actual Huffman encoding process
    update_progress(3, TOTAL_STEPS)
    return encoded_data, {}

# Step 4: Apply Discrete Cosine Transform (DCT) to the background region for lossy compression
def apply_dct(image):
    log_status("Starting Discrete Cosine Transform (DCT)...")
    image_float = np.float32(image) / 255.0
    dct = cv2.dct(image_float)
    update_progress(4, TOTAL_STEPS)
    return dct

# Step 5: Combine Compression Approaches
def compress_image(image):
    log_status("Beginning compression process...")

    # Step 1: Image segmentation
    important_region, background_region = segment_image(image)

    # Step 2: Run-Length Encoding (RLE) on the important region
    important_rle = run_length_encoding(important_region.flatten())

    # Step 3: Huffman Encoding on the important region
    important_huffman_encoded, _ = huffman_encoding(important_rle)

    # Step 4: Apply DCT to the background region for lossy compression
    background_dct = apply_dct(background_region)

    # Step 5: Calculate compression ratio (Cr), PSNR, SSIM, and other metrics
    compression_ratio = len(image) / len(background_dct)  # Placeholder, replace with actual logic
    psnr_value = calculate_psnr(image, background_dct)  # Assuming you have a PSNR calculation function
    ssim_value = calculate_ssim(image, background_dct)  # Assuming SSIM calculation function
    
    # Check constraints
    if compression_ratio > MAX_COMPRESSION_RATIO:
        raise ValueError("Compression ratio exceeds the acceptable limit.")
    if psnr_value < MIN_PSN_RATING:
        raise ValueError("PSNR is too low; the image quality is compromised.")
    if ssim_value < MIN_SSIM:
        raise ValueError("SSIM is too low; structural integrity is compromised.")
    
    log_status("Compression completed successfully.")
    return important_huffman_encoded, background_dct

# Step 6: Save the compressed image
def save_compressed_image(image, path):
    log_status(f"Saving compressed image to {path}...")
    cv2.imwrite(path, image)

# Step 7: Validate image file type (for file uploads)
def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_FILE_TYPES

# Step 8: DICOM Compliance Check (using a placeholder function)
def ensure_dicom_compliance(image):
    log_status("Checking DICOM compliance...")
    if not check_dicom_compliance(image):
        raise ValueError("The image does not comply with the DICOM standard.")
    log_status("DICOM compliance verified.")

# Example function to calculate PSNR (Peak Signal-to-Noise Ratio)
def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Example function to calculate SSIM (Structural Similarity Index)
def calculate_ssim(image1, image2):
    return cv2.compareSSIM(image1, image2)

# Integrate all steps in the Flask app route (assuming Flask handling)
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    
    if file and is_allowed_file(file.filename):
        image = cv2.imread(file)  # Load image
        ensure_dicom_compliance(image)  # Check DICOM compliance
        compressed_data = compress_image(image)  # Perform compression
        save_compressed_image(compressed_data, 'path/to/save/compressed_image')  # Save the compressed image
        return jsonify({"message": "Image compressed successfully"})
    else:
        return jsonify({"error": "Invalid file type"})
