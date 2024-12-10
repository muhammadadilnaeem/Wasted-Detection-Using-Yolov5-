
import streamlit as st
import base64
import os
import shutil
from pathlib import Path

# Define input and output directories
INPUT_DIR = Path("data/input/")
OUTPUT_DIR = Path("data/output/")
INPUT_IMAGE = INPUT_DIR / "inputImage.jpg"
OUTPUT_IMAGE = OUTPUT_DIR / "outputImage.jpg"

# Create directories if not exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Function to decode a base64 image string and save it to a file
def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)

# Function to encode an image file into a base64 string
def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

# Function to run YOLOv5 detection
def run_yolo_detection(input_image_path, output_image_path):
    YOLO_PATH = Path("yolov5/")
    MODEL_WEIGHTS = Path("model/best.pt")
    
    if not YOLO_PATH.exists():
        st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
        return False
    if not MODEL_WEIGHTS.exists():
        st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
        return False

    os.system(
        f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} --img 416 --conf 0.5 --source ../{input_image_path} --save-txt --save-conf"
    )
    
    detected_image_path = YOLO_PATH / "runs/detect/exp/inputImage.jpg"
    if detected_image_path.exists():
        shutil.move(str(detected_image_path), output_image_path)
        return True
    else:
        st.error("Output image not found. Detection may have failed.")
        return False

# Function to reset input and output directories
def reset_directories():
    shutil.rmtree(INPUT_DIR)
    shutil.rmtree(OUTPUT_DIR)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    st.success("Directories reset successfully! Ready for new predictions.")

# Streamlit UI
st.title("Waste Detection App")

uploaded_file = st.file_uploader("Upload an Image for Prediction", type=["jpg", "png"])
if uploaded_file:
    with open(INPUT_IMAGE, "wb") as f:
        f.write(uploaded_file.read())
    
    st.info("Running detection...")
    if run_yolo_detection(INPUT_IMAGE, OUTPUT_IMAGE):
        encoded_image = encodeImageIntoBase64(OUTPUT_IMAGE)
        decoded_image = base64.b64decode(encoded_image)
        st.image(decoded_image, caption="Detected Image", use_column_width=True)
        os.remove(INPUT_IMAGE)
        os.remove(OUTPUT_IMAGE)

if st.button("Reset"):
    reset_directories()



# import streamlit as st
# import base64
# import os
# import shutil

# # Define input and output directories
# INPUT_DIR = r"data/input/"
# OUTPUT_DIR = r"data/output/"
# INPUT_IMAGE = os.path.join(INPUT_DIR, "inputImage.jpg")
# OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "outputImage.jpg")

# # Create directories if not exist
# os.makedirs(INPUT_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# # Function to decode a base64 image string and save it to a file
# def decodeImage(imgstring, fileName):
#     """
#     Decode a base64 image string and save it to a file.

#     :param imgstring: Base64 encoded image string
#     :param fileName: Name of the file to save the decoded image
#     """
#     imgdata = base64.b64decode(imgstring)  # Decode the base64 string into binary data
#     with open(fileName, 'wb') as f:  # Open a file in write-binary mode
#         f.write(imgdata)  # Write the binary data to the file


# # Function to encode an image file into a base64 string
# def encodeImageIntoBase64(croppedImagePath):
#     """
#     Encode an image file into a base64 string.

#     :param croppedImagePath: Path to the image file to encode
#     :return: Base64 encoded string of the image
#     """
#     with open(croppedImagePath, "rb") as f:  # Open the image file in read-binary mode
#         return base64.b64encode(f.read())  # Read the file and return the base64 encoded string


# # Function to run YOLOv5 detection
# def run_yolo_detection(input_image_path, output_image_path):
#     # Ensure YOLOv5 directory exists
#     YOLO_PATH = r"yolov5/"
#     MODEL_WEIGHTS = "model/best.pt"
    
#     if not os.path.exists(YOLO_PATH):
#         st.error("YOLOv5 directory not found.")
#         return False
    
#     # Run YOLOv5 detection
#     os.system(
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} --img 416 --conf 0.5 --source ../{input_image_path} --save-txt --save-conf"
#     )
    
#     # Check for output and move it to the output directory
#     detected_image_path = os.path.join(YOLO_PATH, "runs/detect/exp/inputImage.jpg")
#     if os.path.exists(detected_image_path):
#         os.rename(detected_image_path, output_image_path)
#         return True
#     else:
#         st.error("Output image not found. Detection may have failed.")
#         return False


# # Function to reset input and output directories
# def reset_directories():
#     shutil.rmtree(INPUT_DIR)
#     shutil.rmtree(OUTPUT_DIR)
#     os.makedirs(INPUT_DIR, exist_ok=True)
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     st.success("Directories reset successfully! Ready for new predictions.")


# # Streamlit UI
# st.title("Waste Detection App")

# # Image Upload
# uploaded_file = st.file_uploader("Upload an Image for Prediction", type=["jpg", "png"])
# if uploaded_file:
#     try:
#         # Save uploaded image to input folder
#         with open(INPUT_IMAGE, "wb") as f:
#             f.write(uploaded_file.read())
        
#         # Run YOLOv5 detection
#         st.info("Running detection...")
#         detection_success = run_yolo_detection(INPUT_IMAGE, OUTPUT_IMAGE)
        
#         if detection_success:
#             # Encode the detected image
#             encoded_image = encodeImageIntoBase64(OUTPUT_IMAGE)
#             decoded_image = base64.b64decode(encoded_image)

#             # Display the results
#             st.image(decoded_image, caption="Detected Image", use_column_width=True)

#             # Cleanup individual files
#             os.remove(INPUT_IMAGE)
#             os.remove(OUTPUT_IMAGE)
        
#     except Exception as e:
#         st.error(f"Error: {str(e)}")

# # Reset Button
# if st.button("Reset"):
#     reset_directories()




# --------------------------------------

# import streamlit as st  # Import Streamlit for creating web applications
# import base64           # Import base64 for image encoding and decoding
# import os               # Import os for interacting with the operating system

# # Define paths for YOLOv5 and the model weights
# YOLO_PATH = r"yolov5/"  # Path to the YOLOv5 directory
# MODEL_WEIGHTS = r"model/best.pt"  # Path to the model weights file
# INPUT_IMAGE = r"data/input/inputImage.jpg"  # File path for the uploaded input image
# OUTPUT_IMAGE_DIR = r"yolov5/runs/detect/exp/"  # Directory for output images from detection
# OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_IMAGE_DIR, "inputImage.jpg")  # Full path for the output image

# # Function to decode a base64 image string and save it to a file
# def decodeImage(imgstring, fileName):
#     """
#     Decode a base64 image string and save it to a file.

#     :param imgstring: Base64 encoded image string
#     :param fileName: Name of the file to save the decoded image
#     """
#     imgdata = base64.b64decode(imgstring)  # Decode the base64 string into binary data
#     with open("./data/" + fileName, 'wb') as f:  # Open a file in write-binary mode
#         f.write(imgdata)  # Write the binary data to the file

# # Function to encode an image file into a base64 string
# def encodeImageIntoBase64(croppedImagePath):
#     """
#     Encode an image file into a base64 string.

#     :param croppedImagePath: Path to the image file to encode
#     :return: Base64 encoded string of the image
#     """
#     with open(croppedImagePath, "rb") as f:  # Open the image file in read-binary mode
#         return base64.b64encode(f.read())  # Read the file and return the base64 encoded string

# # Function to run YOLOv5 detection
# def run_yolo_detection(input_image_path):
#     # Ensure the YOLOv5 directory exists
#     if not os.path.exists(YOLO_PATH):
#         st.error("YOLOv5 directory not found.")  # Display an error if the directory is missing
#         return None  # Exit the function

#     # Execute the YOLOv5 detection command
#     os.system(
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} --img 416 --conf 0.5 --source ../{input_image_path}"
#     )
    
#     # Check if the output image was created
#     if not os.path.exists(OUTPUT_IMAGE_PATH):
#         st.error("Output image not found. Detection may have failed.")  # Show error if detection failed
#         return None  # Exit the function

#     return OUTPUT_IMAGE_PATH  # Return the path to the output image

# # Streamlit UI setup
# st.title("Waste Detection App")  # Set the title of the web app

# # Image upload interface
# uploaded_file = st.file_uploader("Upload an Image for Prediction", type=["jpg", "png"])  # Allow users to upload images
# if uploaded_file:  # Check if an image has been uploaded
#     try:
#         # Save the uploaded image to the specified path
#         with open(INPUT_IMAGE, "wb") as f:
#             f.write(uploaded_file.read())
        
#         # Run YOLOv5 detection on the uploaded image
#         st.info("Running detection...")  # Inform the user that detection is in progress
#         detected_image_path = run_yolo_detection(INPUT_IMAGE)  # Call the detection function
        
#         if detected_image_path:  # If detection was successful
#             # Encode the detected image to base64 format
#             encoded_image = encodeImageIntoBase64(detected_image_path)
#             decoded_image = base64.b64decode(encoded_image)  # Decode the base64 image

#             # Display the results in the Streamlit app
#             st.image(decoded_image, caption="Detected Image", use_column_width=True)  # Show the detected image

#             # Clean up the YOLOv5 runs directory to remove temporary files
#             os.system(f"rm -rf {YOLO_PATH}runs")
        
#     except Exception as e:  # Handle any exceptions that occur
#         st.error(f"Error: {str(e)}")  # Display the error message