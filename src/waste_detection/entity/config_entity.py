import os  # Import the OS module for interacting with the operating system
from datetime import datetime  # Import datetime for handling date and time
from dataclasses import dataclass  # Import dataclass for creating data classes
from waste_detection.constant.training_pipeline import *  # Import constants related to the training pipeline

@dataclass
class TrainingPipelineConfig:
    # Data class to hold configuration for the training pipeline
    artifacts_dir: str = ARTIFACTS_DIR  # Directory for storing artifacts

# Create an instance of TrainingPipelineConfig
training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig() 

@dataclass
class DataIngestionConfig:
    # Data class to hold configuration for data ingestion
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, DATA_INGESTION_DIR_NAME  # Directory for data ingestion
    )

    feature_store_file_path: str = os.path.join(
        data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR  # Path for the feature store file
    )

    data_download_url: str = DATA_DOWNLOAD_URL  # URL for downloading data

@dataclass
class DataValidationConfig:
    # Data class to hold configuration for data validation
    data_validation_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, DATA_VALIDATION_DIR_NAME  # Directory for data validation
    )

    valid_status_file_dir: str = os.path.join(data_validation_dir, DATA_VALIDATION_STATUS_FILE)  # Path for validation status file

    required_file_list = DATA_VALIDATION_ALL_REQUIRED_FILES  # List of required files for validation

@dataclass
class ModelTrainerConfig:
    # Data class to hold configuration for model training
    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifacts_dir, MODEL_TRAINER_DIR_NAME  # Directory for model training
    )

    weight_name = MODEL_TRAINER_PRETRAINED_WEIGHT_NAME  # Name of the pre-trained weights file

    no_epochs = MODEL_TRAINER_NO_EPOCHS  # Number of epochs for training

    batch_size = MODEL_TRAINER_BATCH_SIZE  # Batch size for training