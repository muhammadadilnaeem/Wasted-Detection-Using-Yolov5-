import os
import sys
from waste_detection.logger import logging  # Importing logging module for logging events
from waste_detection.exception import AppException  # Importing custom exception handling
from waste_detection.components.data_ingestion import DataIngestion  # Importing DataIngestion class for data handling
from waste_detection.entity.config_entity import DataIngestionConfig  # Importing configuration entity for data ingestion
from waste_detection.entity.artifacts_entity import DataIngestionArtifact  # Importing artifact entity for data ingestion outputs


class TrainPipeline:
    def __init__(self):
        # Initializing the data ingestion configuration
        self.data_ingestion_config = DataIngestionConfig()

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

    def run_pipeline(self) -> None:
        """

        This method initiates the data ingestion process and handles any exceptions
        that may arise during execution.
        
        """
        try:
            # Start the data ingestion process and store the resulting artifact
            data_ingestion_artifact = self.start_data_ingestion()
        
        except Exception as e:
            # Handle exceptions by raising a custom application exception
            raise AppException(e, sys)