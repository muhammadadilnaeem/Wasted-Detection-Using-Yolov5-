import os
import sys
from waste_detection.logger import logging  # Importing logging module for logging events
from waste_detection.exception import AppException  # Importing custom exception handling
from waste_detection.components.data_ingestion import DataIngestion  # Importing DataIngestion class for data handling
from waste_detection.components.data_validation import DataValidation
from waste_detection.entity.config_entity import (DataIngestionConfig,
                                                  DataValidationConfig)  # Importing configuration entity for data ingestion
from waste_detection.entity.artifacts_entity import (DataIngestionArtifact,
                                                    DataValidationArtifact)  # Importing artifact entity for data ingestion outputs


class TrainPipeline:
    def __init__(self):
        # Initializing the data ingestion and data validation configuration
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process and returns the data ingestion artifact.
        
        Returns:
            DataIngestionArtifact: An artifact containing information about the ingested data.
        """
        try: 
            # Log entering the method
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data from URL")

            # Instantiate DataIngestion with the defined configuration
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            # Initiate the data ingestion process
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            # Log successful data retrieval
            logging.info("Got the data from URL")
            # Log exiting the method
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")

            return data_ingestion_artifact  # Return the artifact containing the ingested data

        except Exception as e:
            # Handle exceptions by raising a custom application exception
            raise AppException(e, sys)
        
    def start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """
        Initiates the data validation process using the provided data ingestion artifact.
        
        Parameters:
            data_ingestion_artifact (DataIngestionArtifact): The artifact containing information about the ingested data.
            
        Returns:
            DataValidationArtifact: An artifact containing the validation status after processing.
        """
        logging.info("Entered the start_data_validation method of TrainPipeline class")

        try:
            # Create an instance of DataValidation using the provided artifact and validation configuration
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )

            # Initiate the data validation process
            data_validation_artifact = data_validation.initiate_data_validation()

            # Log the completion of the data validation operation
            logging.info("Performed the data validation operation")

            logging.info("Exited the start_data_validation method of TrainPipeline class")

            return data_validation_artifact  # Return the resulting validation artifact

        except Exception as e:
            # Handle exceptions by raising a custom application exception, preserving the original exception context
            raise AppException(e, sys) from e
        

    def run_pipeline(self) -> None:
        """

        This method initiates the data ingestion process and handles any exceptions
        that may arise during execution.
        
        """
        try:
            # Start the data ingestion process and store the resulting artifact
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )
        
        except Exception as e:
            # Handle exceptions by raising a custom application exception
            raise AppException(e, sys)