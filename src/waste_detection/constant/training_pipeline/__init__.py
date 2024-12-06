
# Define the directory name for storing artifacts
ARTIFACTS_DIR: str = "artifacts"

"""
Data Ingestion related constants start with DATA_INGESTION variable name
"""
# Define the directory name for data ingestion
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

# Define the directory name for storing features
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"

# URL for downloading the dataset from Google Drive
DATA_DOWNLOAD_URL: str = "https://drive.google.com/file/d/1ECfl3dtYyfivY8kYPq7RHUBTjC-2vf61/view?usp=share_link"

"""
Data Validation related constants start with DATA_VALIDATION variable name
"""
# Define the directory name for data validation
DATA_VALIDATION_DIR_NAME: str = "data_validation"

# Name of the file that stores the validation status
DATA_VALIDATION_STATUS_FILE = 'status.txt'

# List of all required files for data validation
DATA_VALIDATION_ALL_REQUIRED_FILES = ["train", "valid", "data.yaml"]

"""
MODEL TRAINER related constants start with MODEL_TRAINER variable name
"""
# Define the directory name for the model trainer
MODEL_TRAINER_DIR_NAME: str = "model_trainer"

# Name of the pre-trained weights file for the model
MODEL_TRAINER_PRETRAINED_WEIGHT_NAME: str = "yolov5s.pt"

# Number of epochs for model training
MODEL_TRAINER_NO_EPOCHS: int = 10

# Batch size to be used during model training
MODEL_TRAINER_BATCH_SIZE: int = 16