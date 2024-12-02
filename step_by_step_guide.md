
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

7. Next step would be to set up **logging and exception** for better code readibilty and pracrice.

    - We need to set up **logging**.

        - Inside `src` folder we have `__init__.py`. We will write logging code in this file.It will help us direcly import `logger`. Here is what logging code will look like:

            ```bash
            import os  # Import the os module for interacting with the operating system
            import sys  # Import the sys module for system-specific parameters and functions
            import logging  # Import the logging module for logging messages

            # Define the logging format string
            logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

            # Specify the directory where log files will be stored
            log_dir = "logs"

            # Create the full path for the log file
            log_filepath = os.path.join(log_dir, "running_logs.log")

            # Create the log directory if it does not exist
            os.makedirs(log_dir, exist_ok=True)

            # Configure the logging settings
            logging.basicConfig(
                level=logging.INFO,  # Set the logging level to INFO
                format=logging_str,  # Use the defined format for log messages
                handlers=[
                    logging.FileHandler(log_filepath),  # Log messages to a file
                    logging.StreamHandler(sys.stdout)    # Also output log messages to the console
                ]
            )

            # Create a logger object with a specific name
            logger = logging.getLogger("kidney_disease_classifier_logger")  # This logger can be used throughout the application
            ```