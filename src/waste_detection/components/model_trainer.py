
# importing ibraries
import os 
import sys
from waste_detection.logger import logging
from waste_detection.exception import AppException
from waste_detection.utils.main_utils import *
from waste_detection.entity.config_entity import ModelTrainerConfig
from waste_detection.entity.artifacts_entity import ModelTrainerArtifact

class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,  # Configuration for the model trainer
    ):
        self.model_trainer_config = model_trainer_config  # Store the configuration

    def initiate_model_trainer(self,) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        try:
            logging.info("Unzipping data")  # Log that data unzipping is starting
            os.system("unzip data.zip")  # Unzip the data file
            os.system("rm data.zip")  # Remove the zip file after extraction

            # Load the number of classes from the data.yaml configuration file
            with open("data.yaml", 'r') as stream:
                num_classes = str(yaml.safe_load(stream)['nc'])

            # Extract the model configuration file name without the extension
            model_config_file_name = self.model_trainer_config.weight_name.split(".")[0]
            print(model_config_file_name)  # Print the model configuration name for reference

            # Read the model configuration from a YAML file
            config = read_yaml_file(f"yolov5/models/{model_config_file_name}.yaml")

            # Update the number of classes in the configuration
            config['nc'] = int(num_classes)

            # Save the updated configuration to a new YAML file
            with open(f'yolov5/models/custom_{model_config_file_name}.yaml', 'w') as f:
                yaml.dump(config, f)

            # Train the model using the YOLOv5 training script with specified parameters
            os.system(f"cd yolov5/ && python train.py --img 416 --batch {self.model_trainer_config.batch_size} --epochs {self.model_trainer_config.no_epochs} --data ../data.yaml --cfg ./models/custom_yolov5s.yaml --weights {self.model_trainer_config.weight_name} --name yolov5s_results  --cache")
            
            # Copy the best model weights to the yolov5 directory
            os.system("cp yolov5/runs/train/yolov5s_results/weights/best.pt yolov5/")
            
            # Create the model trainer directory if it does not exist
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            
            # Copy the best model weights to the specified model trainer directory
            os.system(f"cp yolov5/runs/train/yolov5s_results/weights/best.pt {self.model_trainer_config.model_trainer_dir}/")
           
            # Clean up by removing training results and temporary files
            os.system("rm -rf yolov5/runs")
            os.system("rm -rf train")
            os.system("rm -rf valid")
            os.system("rm -rf data.yaml")

            # Create and return a ModelTrainerArtifact containing the path to the trained model
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path="yolov5/best.pt",
            )

            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")

            return model_trainer_artifact  # Return the artifact with the trained model path

        except Exception as e:
            raise AppException(e, sys)  # Handle exceptions and raise a custom application exception