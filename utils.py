import logging
import os

LOG_FOLDER = 'logs/'
LOG_FILE = os.path.join(LOG_FOLDER, 'compression.log')

# Ensure the log folder exists
os.makedirs(LOG_FOLDER, exist_ok=True)

# Setup logging
logging.basicConfig(filename=LOG_FILE, 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_status(message):
    """Log status to both the terminal and log file."""
    logging.info(message)
    print(message)

def update_progress(step, total_steps):
    """Calculate and log the progress of the compression process."""
    progress = (step / total_steps) * 100
    log_status(f"Compression progress: {progress:.2f}%")
