# Import required libraries
import os
import sys
import gdown
import zipfile
from waste_detection.logger import logging
from waste_detection.exception import AppException
from waste_detection.entity.config_entity import DataIngestionConfig
from waste_detection.entity.artifacts_entity import DataIngestionArtifact


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        Initializes the DataIngestion class with a configuration object.

        Args:
            data_ingestion_config (DataIngestionConfig): Configuration for data ingestion.
        
        Raises:
            AppException: If there is an error during initialization.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise AppException(e, sys)

        
    def download_data(self) -> str:
        """
        Fetch data from the specified URL.

        Returns:
            str: The path to the downloaded zip file.

        Raises:
            AppException: If there is an error during data download.
        """
        try:
            # Retrieve the dataset URL and download directory from config
            dataset_url = self.data_ingestion_config.data_download_url
            zip_download_dir = self.data_ingestion_config.data_ingestion_dir
            
            # Create the download directory if it doesn't exist
            os.makedirs(zip_download_dir, exist_ok=True)
            
            data_file_name = "data.zip"
            zip_file_path = os.path.join(zip_download_dir, data_file_name)
            logging.info(f"Downloading data from {dataset_url} into file {zip_file_path}")

            # Extract the file ID from the dataset URL
            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            
            # Download the dataset using gdown
            gdown.download(prefix + file_id, zip_file_path)

            logging.info(f"Downloaded data from {dataset_url} into file {zip_file_path}")

            return zip_file_path

        except Exception as e:
            raise AppException(e, sys)
        

    def extract_zip_file(self, zip_file_path: str) -> str:
        """
        Extracts the zip file into the specified data directory.

        Args:
            zip_file_path (str): The path to the zip file to be extracted.

        Returns:
            str: The path to the directory where the files were extracted.

        Raises:
            AppException: If there is an error during extraction.
        """
        try:
            # Get feature store path from config and create it if necessary
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(feature_store_path, exist_ok=True)

            # Extract the contents of the zip file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(feature_store_path)
            logging.info(f"Extracting zip file: {zip_file_path} into dir: {feature_store_path}")

            return feature_store_path

        except Exception as e:
            raise AppException(e, sys)
        

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process by downloading and extracting data.

        Returns:
            DataIngestionArtifact: An artifact containing paths to the downloaded and extracted data.

        Raises:
            AppException: If there is an error during the ingestion process.
        """
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")
        try:
            # Download data and extract it
            zip_file_path = self.download_data()
            feature_store_path = self.extract_zip_file(zip_file_path)

            # Create an artifact containing the paths
            data_ingestion_artifact = DataIngestionArtifact(
                data_zip_file_path=zip_file_path,
                feature_store_path=feature_store_path
            )

            logging.info("Exited initiate_data_ingestion method of DataIngestion class")
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            raise AppException(e, sys)