import os  # Importing the os module for interacting with the operating system
import sys 
import shutil  # Importing the shutil module for file operations
from waste_detection.logger import logging  # Importing logging module for logging events
from waste_detection.exception import AppException  # Importing custom exception handling
from waste_detection.entity.config_entity import DataValidationConfig  # Importing configuration for data validation
from waste_detection.entity.artifacts_entity import DataIngestionArtifact, DataValidationArtifact  # Importing artifact entities for data ingestion and validation

class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        """
        Initializes the DataValidation class with ingestion artifact and validation configuration.
        
        Parameters:
            data_ingestion_artifact (DataIngestionArtifact): Artifact containing information about the ingested data.
            data_validation_config (DataValidationConfig): Configuration for data validation.
        """
        try:
            # Assigning the ingestion artifact and validation configuration to instance variables
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

        except Exception as e:
            # Handle exceptions by raising a custom application exception
            raise AppException(e, sys) 

    
    def validate_all_files_exist(self) -> bool:
        """
        Validates that all required files exist in the feature store path.
        
        Returns:
            bool: True if all required files are present, False otherwise.
        """
        try:
            validation_status = None  # Initialize validation status

            # List all files in the feature store path
            all_files = os.listdir(self.data_ingestion_artifact.feature_store_path)

            # Check each file against the required file list
            for file in all_files:
                if file not in self.data_validation_config.required_file_list:
                    # If a required file is missing, set validation status to False
                    validation_status = False
                    os.makedirs(self.data_validation_config.data_validation_dir, exist_ok=True)  # Create validation directory if it doesn't exist
                    # Write validation status to the specified file
                    with open(self.data_validation_config.valid_status_file_dir, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    # If the file is present, set validation status to True
                    validation_status = True
                    os.makedirs(self.data_validation_config.data_validation_dir, exist_ok=True)  # Ensure validation directory exists
                    # Write validation status to the specified file
                    with open(self.data_validation_config.valid_status_file_dir, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status  # Return the final validation status

        except Exception as e:
            # Handle exceptions by raising a custom application exception
            raise AppException(e, sys)

    
    def initiate_data_validation(self) -> DataValidationArtifact: 
        """
        Initiates the data validation process and returns a validation artifact.
        
        Returns:
            DataValidationArtifact: An artifact containing the validation status.
        """
        logging.info("Entered initiate_data_validation method of DataValidation class")
        try:
            # Validate files and get the validation status
            status = self.validate_all_files_exist()
            # Create a validation artifact with the status
            data_validation_artifact = DataValidationArtifact(validation_status=status)

            logging.info("Exited initiate_data_validation method of DataValidation class")
            logging.info(f"Data validation artifact: {data_validation_artifact}")

            # If validation is successful, copy the data zip file to the current working directory
            if status:
                shutil.copy(self.data_ingestion_artifact.data_zip_file_path, os.getcwd())

            return data_validation_artifact  # Return the created validation artifact

        except Exception as e:
            # Handle exceptions by raising a custom application exception
            raise AppException(e, sys)