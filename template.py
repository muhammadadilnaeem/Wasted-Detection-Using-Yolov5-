# import libraries
import os
import logging
from pathlib import Path

# set up logging string
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# set project name
project_name = "waste_detection"

# specify files and folders to be created
list_of_files = [
    ".github/workflows/.gitkeep",
    "data/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/constant/__init__.py",
    f"src/{project_name}/constant/training_pipeline/__init__.py",
    f"src/{project_name}/constant/application.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/entity/artifacts_entity.py",
    f"src/{project_name}/exception/__init__.py",
    f"src/{project_name}/logger/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/main_utils.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "app.py",
    "streamlit.py",

]

# Iterate over each file path in the list of files
for filepath in list_of_files:

    # Convert the filepath string to a Path object for easier manipulation
    filepath = Path(filepath)

    # Split the filepath into its directory and filename components
    filedir, filename = os.path.split(filepath)

    # Check if the directory part of the filepath is not empty
    if filedir !="":

        # Create the directory if it doesn't exist, without raising an error if it does
        os.makedirs(filedir, exist_ok=True)

        # log the meaasge that the file and directory created
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    # Check if the file does not exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        
        # Open the file in write mode (this creates the file if it doesn't exist)
        with open(filepath, "w") as f:

            # No content is written to the file
            pass

            # log the message that created an empty firl    
            logging.info(f"Creating empty file: {filepath}")

    else:

        # Log that the file already exists
        logging.info(f"{filename} is already exists")