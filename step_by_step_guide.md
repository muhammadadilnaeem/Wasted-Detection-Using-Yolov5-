
------

# **`Step by Step Implementation of Kidney Disease Project`**

1. Create a **GitHub** repository.
    - Add **.gitignore**, **licence** and **README.md**.

2. Create a **template.py** file which will help us to create a Project template with single command **(Instead of making files and folders manually)**. here is how this file will look like:

    ```bash
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
        "research/trials.ipynb",
        "Dockerfile"
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
    ```
    - To implement this structure go to terminal and type:
    
    ```bash
    python template.py 
    ```

    - You will see automatically you folder structure will be created.

4. We need to add required libraries for this project in **requirements.txt**. for this prject we used the following:

    ```bash
    tensorflow
    pandas 
    dvc
    mlflow
    notebook
    ipykernel
    numpy
    matplotlib
    seaborn
    python-box
    pyYAML
    tqdm
    joblib
    types-PyYAML
    scipy
    Flask
    Flask-Cors
    streamlit
    gdown
    -e .
    ```

3. Now if we want to set up our folder as local package we need to write this code in **setup.py** which will look like this:

    ```bash
    # Import the setuptools library to facilitate packaging and distribution
    import setuptools

    # Open the README file to read its content for the long description, using UTF-8 encoding
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

    # Define the version of the package
    __version__ = "0.0.0"

    # Define metadata for the package
    REPO_NAME = "Waste-Detection-Using-Yolov5"  # GitHub repository name
    AUTHOR_USER_NAME = "muhammadadilnaeem"  # GitHub username
    SRC_REPO = "waste_detection"  # Name of the source directory/package
    AUTHOR_EMAIL = "madilnaeem0@gmail.com"   # Contact email for the author

    # Call the setup function to configure the package
    setuptools.setup(
        name=SRC_REPO,  # Name of the package, must match the directory in 'src'
        version=__version__,  # Version of the package
        author=AUTHOR_USER_NAME,   # Author name as it will appear in package metadata
        author_email=AUTHOR_EMAIL,  # Author's contact email
        description="A demo python package for Waste Detedtion Using Yolov5 Web Application.",  # Short package description
        long_description=long_description,  # Detailed description read from README.md
        long_description_content="text/markdown",  # Format of the long description (markdown)
        url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",  # URL of the package repository
        project_urls={   # Additional URLs related to the project, such as an issue tracker
            "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
        },
        package_dir={"": "src"}, # Specify that the root of packages is the 'src' directory
        packages=setuptools.find_packages(where="src")  # Automatically find all packages in 'src'
    )
    ```

4. Now if i want to set up **src** folder as my local package i need to add `-e .` at the end of my **requirements.txt** file. You do not need to run **setup.py** seperately.   

5. Now we need to create a **Virtual Environment**. This will help us to avoid libraries conflict. 

    - For creating virtual environment use this command:

        ```bash
        conda create -p venv python=3.7 -y
        ```

    - Now we need to **activate** the created Virtual Environment. For this use this command:

        ```bash
        conda activate /workspaces/Waste-Detection-Using-Yolov5/venv
        ```

        - In case if you see any error 
            
            - Close the current terminal or shell window and open a new one. This will allow the changes from conda init to take effect.

            - run this command again.

        - Now we have setup Virtual Environment we need to install libraries using **requirements.txt** file, For this go to terminal and type this command:

            ```bash
            pip install -r requirements.txt
            ```

6. Since we are working with **Object Detection** project we need to **Annotate** mean **label** our **Training** data.

    - Please **annotate** your data or get **annotated data**.
    - For this project I will use [Roboflow](https://universe.roboflow.com/material-identification/garbage-classification-3)

7. Next step would be to set up **logging and exception and utils** for better code readibilty and pracrice.

    - We need to set up **logging**.

        - Inside `src/waste_detection/logger` in `__init__.py` we will write custom logging which will look like this:

            ```bash
            import os  # Import the OS module for interacting with the operating system
            import logging  # Import the logging module for logging messages
            from datetime import datetime  # Import datetime for timestamping log files
            from from_root import from_root  # Import from_root to get the project's root directory

            # Create a log file name based on the current date and time
            LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

            # Define the path for the log file by joining the root directory with the log folder and log file name
            log_path = os.path.join(from_root(), "log", LOG_FILE)

            # Create the log directory if it doesn't already exist
            os.makedirs(log_path, exist_ok=True)

            # Combine the log directory path with the log file name to get the full log file path
            LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

            # Configure the logging settings
            logging.basicConfig(
                filename=LOG_FILE_PATH,  # Set the log file path
                format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",  # Define the log message format
                level=logging.INFO  # Set the logging level to INFO
            )
            ```
        - Now we need to write custom exception. for this Inside `src/waste_detection/exception` in `__init__.py` we will write custom exception which will look like this:

            ```bash
            import sys  # Import the sys module to access system-specific parameters and functions


            def error_message_detail(error, error_detail: sys):
                # Extract the exception traceback details
                _, _, exc_tb = error_detail.exc_info()

                # Get the name of the file where the exception occurred
                file_name = exc_tb.tb_frame.f_code.co_filename

                # Format the error message to include the file name, line number, and error message
                error_message = "Error occurred in Python script name [{0}] line number [{1}] error message [{2}]".format(
                    file_name, exc_tb.tb_lineno, str(error)
                )

                return error_message  # Return the formatted error message


            class AppException(Exception):
                # Custom exception class that inherits from the built-in Exception class
                def __init__(self, error_message, error_detail):
                    """
                    Initialize the AppException with a custom error message and details.

                    :param error_message: error message in string format
                    """
                    super().__init__(error_message)  # Call the base class constructor with the error message

                    # Generate a detailed error message using the provided error details
                    self.error_message = error_message_detail(
                        error_message, error_detail=error_detail
                    )

                def __str__(self):
                    # Override the string representation of the exception to return the custom error message
                    return self.error_message
            ```
        - Now we need to add utility functions. For this Inside `src/waste_detection/utils` in `main_utils.py` we will write some functions which will look like this:

            ```bash
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
            ```

8. Our project **workflow** will be in a sequence. We will update these files in order every time we write code.

    1. ***constants***
    2. ***entity***
    3. ***components***
    4. ***pipeline***
    5. ***app.py***


        
