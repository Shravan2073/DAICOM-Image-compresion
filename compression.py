import numpy as np
import cv2
from utils import log_status, update_progress

# Assuming 4 steps for compression progress: segmentation, RLE, Huffman, and DCT
TOTAL_STEPS = 4

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

# Step 3: Huffman Encoding
# Similar Huffman encoding implementation as before
def huffman_encoding(data):
    log_status("Starting Huffman Encoding...")
    # Huffman implementation goes here
    update_progress(3, TOTAL_STEPS)
    return "encoded_data", {}

# Step 4: DCT for lossy compression
def apply_dct(image):
    log_status("Starting Discrete Cosine Transform (DCT)...")
    image_float = np.float32(image) / 255.0
    dct = cv2.dct(image_float)
    update_progress(4, TOTAL_STEPS)
    return dct

# Step 5: Combine Compression Approaches
def compress_image(image):
    log_status("Beginning compression process...")
    important_region, background_region = segment_image(image)
    important_rle = run_length_encoding(important_region.flatten())
    important_huffman_encoded, _ = huffman_encoding(important_rle)
    background_dct = apply_dct(background_region)
    log_status("Compression completed successfully.")
    return important_huffman_encoded, background_dct

# Step 6: Save the compressed image
def save_compressed_image(image, path):
    log_status(f"Saving compressed image to {path}...")
    cv2.imwrite(path, image)
