

# Import the setuptools library to facilitate packaging and distribution
import setuptools

# Open the README file to read its content for the long description, using UTF-8 encoding
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Define the version of the package
__version__ = "0.0.0"

# Define metadata for the package
REPO_NAME = "Kidney-Disease-Classification-Using-MLFlow-And-DVC"  # GitHub repository name
AUTHOR_USER_NAME = "muhammadadilnaeem"  # GitHub username
SRC_REPO = "waste_detection"  # Name of the source directory/package
AUTHOR_EMAIL = "madilnaeem0@gmail.com"   # Contact email for the author

# Call the setup function to configure the package
setuptools.setup(
    name=SRC_REPO,  # Name of the package, must match the directory in 'src'
    version=__version__,  # Version of the package
    author=AUTHOR_USER_NAME,   # Author name as it will appear in package metadata
    author_email=AUTHOR_EMAIL,  # Author's contact email
    description="A demo python package for waste detection Web Application.",  # Short package description
    long_description=long_description,  # Detailed description read from README.md
    long_description_content="text/markdown",  # Format of the long description (markdown)
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",  # URL of the package repository
    project_urls={   # Additional URLs related to the project, such as an issue tracker
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"}, # Specify that the root of packages is the 'src' directory
    packages=setuptools.find_packages(where="src")  # Automatically find all packages in 'src'
)