import os  # Importing the os module for operating system functionalities
# import sys  # Importing the sys module for system-specific parameters and functions
from waste_detection.logger import logging  # Importing logging module for logging events
from waste_detection.exception import AppException  # Importing custom exception handling
from waste_detection.pipeline.training_pipeline import TrainPipeline  # Importing the TrainPipeline class for training operations

# Create an instance of the TrainPipeline class
load_data = TrainPipeline()

# Execute the run_pipeline method to start the data ingestion process
load_data.run_pipeline()