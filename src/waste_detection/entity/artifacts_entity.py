
from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """
    A dataclass to hold information about the data ingestion process.
    
    Attributes:
        data_zip_file_path (str): The file path of the zipped data.
        feature_store_path (str): The path where the features are stored.
    """
    data_zip_file_path: str
    feature_store_path: str


@dataclass
class DataValidationArtifact:
    """
    A dataclass to represent the output of the data validation process.
    
    Attributes:
        validation_status (bool): Status indicating whether the data validation passed or failed.
    """
    validation_status: bool


# @dataclass
# class ModelTrainerArtifact:
#     """
#     A dataclass to hold information about the trained model.
    
#     Attributes:
#         trained_model_file_path (str): The file path where the trained model is saved.
#     """
#     trained_model_file_path: str