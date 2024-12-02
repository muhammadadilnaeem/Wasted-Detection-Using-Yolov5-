import os  # Import the OS module for interacting with the operating system
import logging  # Import the logging module for logging messages
from datetime import datetime  # Import datetime for timestamping log files
from from_root import from_root  # Import from_root to get the project's root directory

# Create a log file name based on the current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the path for the log file by joining the root directory with the log folder and log file name
log_path = os.path.join(from_root(), "log", LOG_FILE)

# Create the log directory if it doesn't already exist
os.makedirs(log_path, exist_ok=True)

# Combine the log directory path with the log file name to get the full log file path
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

# Configure the logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Set the log file path
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",  # Define the log message format
    level=logging.INFO  # Set the logging level to INFO
)