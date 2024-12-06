import os
import sys
from waste_detection.logger import logging  # Importing logging module for logging events
from waste_detection.exception import AppException  # Importing custom exception handling
from waste_detection.components.data_ingestion import DataIngestion  # Importing DataIngestion class for data handling
from waste_detection.components.data_validation import DataValidation
from waste_detection.entity.config_entity import (DataIngestionConfig,
                                                  DataValidationConfig,
                                                  ModelTrainerConfig)  # Importing configuration entity for data ingestion
from waste_detection.entity.artifacts_entity import (DataIngestionArtifact,
                                                    DataValidationArtifact,
                                                    ModelTrainerArtifact)  # Importing artifact entity for data ingestion outputs
from waste_detection.components.model_trainer import ModelTrainer



class TrainPipeline:
    def __init__(self):
        # Initializing the data ingestion and data validation configuration
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.model_trainer_config = ModelTrainerConfig()

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
        
    def start_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiates the model training process.

        Returns:
        ModelTrainerArtifact: An artifact containing the results of the model training process.

        Raises:
        AppException: If an error occurs during the model training initiation.
        """
        try:
            # Create an instance of the ModelTrainer with the configuration provided
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
            )
            
            # Start the model training process and store the resulting artifact
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            
            # Return the artifact containing the results of the model training
            return model_trainer_artifact

        except Exception as e:
            # Raise a custom AppException if an error occurs
            raise AppException(e, sys)
        


    def run_pipeline(self) -> None:
        """
        Executes the data processing pipeline, which includes data ingestion, 
        data validation, and model training.

        Raises:
            AppException: If an error occurs during any stage of the pipeline.
        """
        try:
            # Start the data ingestion process and store the resulting artifact
            data_ingestion_artifact = self.start_data_ingestion()
            
            # Start the data validation process using the ingestion artifact
            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact=data_ingestion_artifact
            )

            # Check if the data validation was successful
            if data_validation_artifact.validation_status == True:
                # Start the model training process if data is valid
                model_trainer_artifact = self.start_model_trainer()
            
            else:
                # Raise an exception if the data format is incorrect
                raise Exception("Your data is not in correct format")

        except Exception as e:
            # Raise a custom AppException if an error occurs
            raise AppException(e, sys)