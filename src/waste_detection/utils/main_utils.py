import os.path  # Import the os.path module for file path manipulations
import sys  # Import the sys module to access system-specific parameters and functions
import yaml  # Import the yaml module for reading and writing YAML files
import base64  # Import the base64 module for encoding and decoding base64 data

from waste_detection.logger import logging  # Import the logging module for logging messages
from waste_detection.exception import AppException  # Import the custom AppException class for error handling

def read_yaml_file(file_path: str) -> dict:
    """
    Read a YAML file and return its contents as a dictionary.

    :param file_path: Path to the YAML file
    :return: Contents of the file as a dictionary
    """
    try:
        # Open the YAML file in binary read mode
        with open(file_path, "rb") as yaml_file:
            logging.info("Read yaml file successfully")  # Log success message
            return yaml.safe_load(yaml_file)  # Load and return the YAML contents as a dictionary

    except Exception as e:
        # Raise a custom AppException if an error occurs during file reading
        raise AppException(e, sys) from e
    


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Write a given content to a YAML file.

    :param file_path: Path to the YAML file
    :param content: Content to write to the file
    :param replace: Boolean flag to determine if existing file should be replaced
    """
    try:
        # If replace is True and the file exists, remove it
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)

        # Create the directory for the file if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Open the file in write mode and dump the content as YAML
        with open(file_path, "w") as file:
            yaml.dump(content, file)  # Write the content to the YAML file
            logging.info("Successfully write_yaml_file")  # Log success message

    except Exception as e:
        # Raise a custom AppException if an error occurs during file writing
        raise AppException(e, sys)
    


def decodeImage(imgstring, fileName):
    """
    Decode a base64 image string and save it to a file.

    :param imgstring: Base64 encoded image string
    :param fileName: Name of the file to save the decoded image
    """
    imgdata = base64.b64decode(imgstring)  # Decode the base64 string into binary data
    with open("./data/" + fileName, 'wb') as f:  # Open a file in write-binary mode
        f.write(imgdata)  # Write the binary data to the file
        f.close()  # Close the file


def encodeImageIntoBase64(croppedImagePath):
    """
    Encode an image file into a base64 string.

    :param croppedImagePath: Path to the image file to encode
    :return: Base64 encoded string of the image
    """
    with open(croppedImagePath, "rb") as f:  # Open the image file in read-binary mode
        return base64.b64encode(f.read())  # Read the file and return the base64 encoded string