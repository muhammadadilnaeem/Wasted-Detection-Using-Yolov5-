
import streamlit as st
import base64
import os
import shutil
from pathlib import Path
import glob
import pandas as pd

# -------------------
# Application Setup
# -------------------

# Set page configuration as the first Streamlit command
st.set_page_config(
    page_title="WasteWise AI 🌍", 
    page_icon="♻️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define core application paths and directories
# PROJECT_ROOT: Root directory of the application
# INPUT_DIR: Directory for storing uploaded images
# OUTPUT_DIR: Directory for storing processed images
# INPUT_IMAGE: Path for the current image being processed
# YOLO_PATH: Directory containing YOLOv5 implementation
# MODEL_WEIGHTS: Path to trained model weights

# Define project directories and paths
PROJECT_ROOT = Path(__file__).parent.resolve()
INPUT_DIR = PROJECT_ROOT / "data/input/"
OUTPUT_DIR = PROJECT_ROOT / "data/output/"
INPUT_IMAGE = INPUT_DIR / "inputImage.jpg"
YOLO_PATH = PROJECT_ROOT / "yolov5/"
MODEL_WEIGHTS = PROJECT_ROOT / "model/best.pt"

# Create directories if they don't exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------
# Styling Functions
# -------------------

# Custom CSS for enhanced styling
def local_css():

    """
    Apply custom CSS styling to the Streamlit application.
    Includes styles for:
    - Page background and typography
    - Headers and sidebar
    - Interactive elements (buttons, containers)
    - Image display areas
    - Metrics and statistics displays
    """

    st.markdown("""
    <style>
    /* Enhanced Page Background */
    .stApp {
        background-color: #e6f3f0;
        font-family: 'Roboto', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Stylish Headings */
    h1 {
        color: #2ecc71;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar Styling */
    .css-1aumxhk {
        background-color: #f1f8ff;
        border-right: 2px solid #3498db;
    }
    
    /* Instruction Box Styling */
    .instruction-box {
        background-color: #ecf0f1;
        border-left: 6px solid #2ecc71;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Enhanced Buttons */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Image Container Styling */
    .image-container {
        border: 3px dashed #3498db;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
        background-color: white;
        transition: all 0.3s ease;
    }
    
    .image-container:hover {
        border-color: #2ecc71;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Metrics and Stats Styling */
    .metric-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------
# Helper Functions
# -------------------


def decodeImage(imgstring, fileName):

    """
    Decode a base64 encoded image string and save it to a file.
    
    Args:
        imgstring (str): Base64 encoded image string
        fileName (str): Path where the decoded image will be saved
    """

    imgdata = base64.b64decode(imgstring)
    with open(fileName, "wb") as f:
        f.write(imgdata)

def encodeImageIntoBase64(image_path):

    """
    Encode an image file into base64 string.
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        bytes: Base64 encoded image data
    """

    with open(image_path, "rb") as f:
        return base64.b64encode(f.read())

def get_latest_output_dir():

    """
    Get the most recent YOLO detection output directory.
    
    Returns:
        Path: Path to the latest output directory or None if no directories exist
    """

    exp_dirs = sorted(glob.glob(f"{YOLO_PATH}/runs/detect/exp*"), key=os.path.getmtime)
    return Path(exp_dirs[-1]) if exp_dirs else None

def parse_detection_results(output_dir):
    
    """
    Parse YOLO detection results to count detected waste types.
    
    Args:
        output_dir (Path): Directory containing YOLO detection results
    
    Returns:
        dict: Dictionary mapping waste type IDs to their counts
    """


    label_files = list(output_dir.glob('*.txt'))
    waste_types = {}
    
    if not label_files:
        return waste_types
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                waste_type = int(line.split()[0])
                waste_types[waste_type] = waste_types.get(waste_type, 0) + 1
    
    return waste_types

def run_yolo_detection(input_image_path):
    
    
    """
    Run YOLOv5 object detection on the input image.
    
    Args:
        input_image_path (Path): Path to input image
    
    Returns:
        tuple: (output_image_path, waste_types) or (False, None) if detection fails
        - output_image_path: Path to the processed image with detections
        - waste_types: Dictionary of detected waste types and their counts
    """

     # Validate required paths exist
    if not YOLO_PATH.exists():
        st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
        return False, None
    
    # Validate required paths exist
    if not MODEL_WEIGHTS.exists():
        st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
        return False, None
    
    # Construct and execute YOLO detection command
    detect_command = (
        f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} "
        f"--img 416 --conf 0.5 --source {input_image_path} "
        f"--save-txt --save-conf"
    )
    
    try:
        import subprocess
        result = subprocess.run(detect_command, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            st.error(f"Detection command failed: {result.stderr}")
            return False, None

    except Exception as e:
        st.error(f"An error occurred during detection: {e}")
        return False, None
    
    # Process detection results
    latest_output_dir = get_latest_output_dir()
    if latest_output_dir:
        detected_image_path = latest_output_dir / "inputImage.jpg"
        if detected_image_path.exists():
            output_image_path = OUTPUT_DIR / detected_image_path.name
            shutil.copy(detected_image_path, output_image_path)
            
            # Parse detection results
            waste_types = parse_detection_results(latest_output_dir)
            
            return output_image_path, waste_types
    
    st.error("Detected image not found. Detection may have failed.")
    return False, None

def reset_directories():

    """
    Reset application directories by removing and recreating input/output folders.
    This ensures a clean state for new detections.
    """

    shutil.rmtree(INPUT_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def waste_type_mapping():
    
    """
    Provide mapping between numeric labels and waste type names.
    
    Returns:
        dict: Dictionary mapping numeric labels to waste type names
    """
       

    return {
        0: "Banana",
        1: "Chilli",
        2: "Drinkcan",
        3: "Drinkpack",
        4: "Foodcan",
        5: "Lettuce",
        6: "Paperbag",
        7: "Plasticbag",
        8: "Plasticbottle",
        9: "Sweetpotato",
        10: "Teabag",
        11: "tissueroll",
    }


# -------------------
# Main Application
# -------------------

def main():

    """
    Main application function that handles:
    1. UI setup and styling
    2. Sidebar information and controls
    3. Image upload and processing
    4. Results display
    5. Sample image selection and processing
    6. Application reset functionality
    """

    # Apply custom CSS
    local_css()

    # Sidebar with comprehensive instructions and app information
    with st.sidebar:
        st.image(r"research/result.jpg", caption="WasteWise AI")
        
        st.markdown("# `🌍 WasteWise AI`")
        st.markdown("""
        ## ***About the App***
        WasteWise AI is an intelligent waste detection and classification system 
        powered by advanced machine learning techniques. This is ***YOLOv5s*** for 
                    ***Object Detection*** of Waste.
        
        ## ***How to Use***
        1. 📤 **Upload an Image**
           - Click on the file uploader
           - Select a clear image of waste materials
           - Supported formats: JPG, PNG, JPEG
        
        2. 🤖 **AI Detection**
           - Our AI will automatically process the image
           - Detect and classify waste types
           - Provide visual and statistical insights
        
        3. 🔍 **Analyze Results**
           - View detected waste types
           - See classification confidence
           - Understand waste composition
        
        ### ***Best Practices***
        - Use clear, well-lit images
        - Ensure waste is clearly visible
        - Avoid cluttered backgrounds
        - One type of waste per image recommended
        """)
        
        # Waste Type Legend
        st.markdown("## ***🏷️ Waste Type Legend***")
        waste_types = waste_type_mapping()
        for idx, waste_type in waste_types.items():
            st.markdown(f"- **{idx}**: {waste_type}")

    # Main application layout and functionality
    st.markdown("<h1 style='text-align: center;'> 🌍 WasteWise: Intelligent Waste Detection 🤖</h1>", unsafe_allow_html=True)

    # Instruction Box
    st.markdown("""
    <div class='instruction-box'>
    📝 <strong>Quick Guide:</strong>
    Upload an image of waste materials, and our AI will help you understand its composition and classification!
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for reset functionality
    if "reset_triggered" not in st.session_state:
        st.session_state.reset_triggered = False

    # Display a success message if the application was reset
    if st.session_state.reset_triggered:
        st.success("✅ Application reset successfully! Upload a new image to start fresh.")
        st.session_state.reset_triggered = False

# Heading
    st.title("🔬 Upload An Image for Predicion 📤")

    # Provide a user-friendly interface for image upload
    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "🔍 Upload Waste Image", 
        type=["jpg", "png", "jpeg"], 
        help="Supported formats: JPG, PNG, JPEG",
        accept_multiple_files=False
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Process the uploaded image and display the results
    if uploaded_file:
        with open(INPUT_IMAGE, "wb") as f:
            f.write(uploaded_file.read())

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("📸 **Uploaded Image**")
            st.image(str(INPUT_IMAGE), width=660)

        with col2:
            output_image_path, waste_types = run_yolo_detection(INPUT_IMAGE)
            if output_image_path:
                st.markdown("🤖 **Predicted Image**")
                encoded_image = encodeImageIntoBase64(output_image_path)
                decoded_image = base64.b64decode(encoded_image)
                st.image(decoded_image, width=660)
                os.remove(INPUT_IMAGE)
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("🤖 **Predicted Image**")
                st.info("Prediction failed. Please try uploading a different image.")

    # Initialize session state for selected image
    if 'selected_image_path' not in st.session_state:
        st.session_state.selected_image_path = None

    
    # Sample images for prediction
    normal_images = [
        "sample_images/Chilli.jpg", "sample_images/DrinkCan.jpg", 
        "sample_images/DrinkPack.jpg", "sample_images/FoodCan.jpg", 
        "sample_images/Lettuce.jpg", "sample_images/PaperBag.jpg", 
        "sample_images/PlasticBag.jpg", "sample_images/PlasticContainer.jpg", 
        "sample_images/SweetPotato.jpg", "sample_images/TeaBag.jpg", 
        "sample_images/TissueRoll.jpg", "sample_images/Mixed.jpg", 
    ]

    st.title("Select Sample Images for Prediction 🖼️")

    # Display Normal Images
    with st.expander("📂 Normal Images For Prediction"):
        # Create a grid of columns for better layout
        cols = st.columns(4)  # 4 columns for better spacing

        for i, img_path in enumerate(normal_images):
            with cols[i % 4]:  # Cycle through columns
                # Display image
                st.image(img_path, caption=f"Sample {i+1}", use_container_width=True)

                # Add selection button
                if st.button(f"Select Image {i+1}", key=f"select_{i}"):

                    # Copy the selected sample image to input directory
                    shutil.copy(img_path, INPUT_IMAGE)
                    st.session_state.selected_image_path = img_path
                    st.success(f"📸 Selected: {os.path.basename(img_path)}")

    # Process the selected sample image
    if st.session_state.selected_image_path:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📸 ***Selected Sample Image***")
            st.image(st.session_state.selected_image_path, width=660)

        with col2:
            output_image_path, waste_types = run_yolo_detection(INPUT_IMAGE)

            if output_image_path:
                st.markdown("### 🤖 ***Predicted Image***")
                encoded_image = encodeImageIntoBase64(output_image_path)
                decoded_image = base64.b64decode(encoded_image)
                st.image(decoded_image, width=660)

                # Clean up input image after processing
                os.remove(INPUT_IMAGE)
                st.session_state.selected_image_path = None
            else:
                st.markdown("🤖 **Predicted Image**")
                st.info("Prediction failed. Please try selecting a different image.")
            

    # Provide a user-friendly reset button
    if st.button("🔄 Reset Application", help="Clear all uploaded images and reset the app"):
        try:
            reset_directories()
            st.session_state.reset_triggered = True
            st.success("✔ Application Reset Successfully!")
        except Exception as e:
            st.error(f"⚠️ Reset failed: {e}")

if __name__ == "__main__":
    main()


# (-----------------------------------------------------)

# import streamlit as st
# import base64
# import os
# import shutil
# from pathlib import Path
# import glob
# import pandas as pd

# # Set page configuration as the first Streamlit command
# st.set_page_config(
#     page_title="WasteWise AI 🌍", 
#     page_icon="♻️", 
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Define project directories and paths
# PROJECT_ROOT = Path(__file__).parent.resolve()
# INPUT_DIR = PROJECT_ROOT / "data/input/"
# OUTPUT_DIR = PROJECT_ROOT / "data/output/"
# INPUT_IMAGE = INPUT_DIR / "inputImage.jpg"
# YOLO_PATH = PROJECT_ROOT / "yolov5/"
# MODEL_WEIGHTS = PROJECT_ROOT / "model/best.pt"

# # Create directories if they don't exist
# INPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Custom CSS for enhanced styling
# def local_css():
#     st.markdown("""
#     <style>
#     /* Enhanced Page Background */
#     .stApp {
#         background-color: #e6f3f0;
#         font-family: 'Roboto', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#     }
    
#     /* Stylish Headings */
#     h1 {
#         color: #2ecc71;
#         text-align: center;
#         font-weight: bold;
#         margin-bottom: 20px;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
#     }
    
#     /* Sidebar Styling */
#     .css-1aumxhk {
#         background-color: #f1f8ff;
#         border-right: 2px solid #3498db;
#     }
    
#     /* Instruction Box Styling */
#     .instruction-box {
#         background-color: #ecf0f1;
#         border-left: 6px solid #2ecc71;
#         padding: 15px;
#         border-radius: 8px;
#         margin-bottom: 20px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
    
#     /* Enhanced Buttons */
#     .stButton>button {
#         background-color: #3498db;
#         color: white;
#         border: none;
#         padding: 10px 20px;
#         border-radius: 8px;
#         font-weight: bold;
#         transition: all 0.3s ease;
#         text-transform: uppercase;
#         letter-spacing: 1px;
#     }
    
#     .stButton>button:hover {
#         background-color: #2980b9;
#         transform: scale(1.05);
#         box-shadow: 0 4px 8px rgba(0,0,0,0.2);
#     }
    
#     /* Image Container Styling */
#     .image-container {
#         border: 3px dashed #3498db;
#         border-radius: 12px;
#         padding: 20px;
#         text-align: center;
#         margin-bottom: 20px;
#         background-color: white;
#         transition: all 0.3s ease;
#     }
    
#     .image-container:hover {
#         border-color: #2ecc71;
#         box-shadow: 0 4px 8px rgba(0,0,0,0.1);
#     }
    
#     /* Metrics and Stats Styling */
#     .metric-container {
#         background-color: white;
#         border-radius: 10px;
#         padding: 15px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#         margin-top: 20px;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Helper functions from previous implementation
# def decodeImage(imgstring, fileName):
#     imgdata = base64.b64decode(imgstring)
#     with open(fileName, "wb") as f:
#         f.write(imgdata)

# def encodeImageIntoBase64(image_path):
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read())

# def get_latest_output_dir():
#     exp_dirs = sorted(glob.glob(f"{YOLO_PATH}/runs/detect/exp*"), key=os.path.getmtime)
#     return Path(exp_dirs[-1]) if exp_dirs else None

# def parse_detection_results(output_dir):
#     """Parse YOLO detection results to extract waste type counts"""
#     label_files = list(output_dir.glob('*.txt'))
#     waste_types = {}
    
#     if not label_files:
#         return waste_types
    
#     for label_file in label_files:
#         with open(label_file, 'r') as f:
#             for line in f:
#                 waste_type = int(line.split()[0])
#                 waste_types[waste_type] = waste_types.get(waste_type, 0) + 1
    
#     return waste_types

# def run_yolo_detection(input_image_path):
#     # Error handling for missing directories/weights
#     if not YOLO_PATH.exists():
#         st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
#         return False, None

#     if not MODEL_WEIGHTS.exists():
#         st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
#         return False, None

#     detect_command = (
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} "
#         f"--img 416 --conf 0.5 --source {input_image_path} "
#         f"--save-txt --save-conf"
#     )
    
#     try:
#         import subprocess
#         result = subprocess.run(detect_command, shell=True, capture_output=True, text=True)
        
#         if result.returncode != 0:
#             st.error(f"Detection command failed: {result.stderr}")
#             return False, None

#     except Exception as e:
#         st.error(f"An error occurred during detection: {e}")
#         return False, None

#     latest_output_dir = get_latest_output_dir()
#     if latest_output_dir:
#         detected_image_path = latest_output_dir / "inputImage.jpg"
#         if detected_image_path.exists():
#             output_image_path = OUTPUT_DIR / detected_image_path.name
#             shutil.copy(detected_image_path, output_image_path)
            
#             # Parse detection results
#             waste_types = parse_detection_results(latest_output_dir)
            
#             return output_image_path, waste_types
    
#     st.error("Detected image not found. Detection may have failed.")
#     return False, None

# def reset_directories():
#     shutil.rmtree(INPUT_DIR, ignore_errors=True)
#     shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
#     INPUT_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# def waste_type_mapping():
#     """Map numeric labels to waste types"""
#     return {
#         0: "Banana",
#         1: "Chilli",
#         2: "Drinkcan",
#         3: "Drinkpack",
#         4: "Foodcan",
#         5: "Lettuce",
#         6: "Paperbag",
#         7: "Plasticbag",
#         8: "Plasticbottle",
#         9: "Sweetpotato",
#         10: "Teabag",
#         11: "tissueroll",
#     }

# def main():
#     # Apply custom CSS
#     local_css()

#     # Sidebar with comprehensive instructions and app information
#     with st.sidebar:
#         st.image(r"research/result.jpg", caption="WasteWise AI")
        
#         st.markdown("# `🌍 WasteWise AI`")
#         st.markdown("""
#         ## ***About the App***
#         WasteWise AI is an intelligent waste detection and classification system 
#         powered by advanced machine learning techniques. This is ***YOLOv5s*** for 
#                     ***Object Detection*** of Waste.
        
#         ## ***How to Use***
#         1. 📤 **Upload an Image**
#            - Click on the file uploader
#            - Select a clear image of waste materials
#            - Supported formats: JPG, PNG, JPEG
        
#         2. 🤖 **AI Detection**
#            - Our AI will automatically process the image
#            - Detect and classify waste types
#            - Provide visual and statistical insights
        
#         3. 🔍 **Analyze Results**
#            - View detected waste types
#            - See classification confidence
#            - Understand waste composition
        
#         ### ***Best Practices***
#         - Use clear, well-lit images
#         - Ensure waste is clearly visible
#         - Avoid cluttered backgrounds
#         - One type of waste per image recommended
#         """)
        
#         # Waste Type Legend
#         st.markdown("## ***🏷️ Waste Type Legend***")
#         waste_types = waste_type_mapping()
#         for idx, waste_type in waste_types.items():
#             st.markdown(f"- **{idx}**: {waste_type}")

#     # Main Application Title
#     st.markdown("<h1 style='text-align: center;'> 🌍 WasteWise: Intelligent Waste Detection 🤖</h1>", unsafe_allow_html=True)

#     # Instruction Box
#     st.markdown("""
#     <div class='instruction-box'>
#     📝 <strong>Quick Guide:</strong>
#     Upload an image of waste materials, and our AI will help you understand its composition and classification!
#     </div>
#     """, unsafe_allow_html=True)

#     # Initialize session state for reset functionality
#     if "reset_triggered" not in st.session_state:
#         st.session_state.reset_triggered = False

#     # Display a success message if the application was reset
#     if st.session_state.reset_triggered:
#         st.success("✅ Application reset successfully! Upload a new image to start fresh.")
#         st.session_state.reset_triggered = False

# # Heading
#     st.title("🔬 Upload An Image for Predicion 📤")

#     # Provide a user-friendly interface for image upload
#     st.markdown("<div class='image-container'>", unsafe_allow_html=True)
#     uploaded_file = st.file_uploader(
#         "🔍 Upload Waste Image", 
#         type=["jpg", "png", "jpeg"], 
#         help="Supported formats: JPG, PNG, JPEG",
#         accept_multiple_files=False
#     )
#     st.markdown("</div>", unsafe_allow_html=True)

#     # Process the uploaded image and display the results
#     if uploaded_file:
#         with open(INPUT_IMAGE, "wb") as f:
#             f.write(uploaded_file.read())

#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("📸 **Uploaded Image**")
#             st.image(str(INPUT_IMAGE), width=660)

#         with col2:
#             output_image_path, waste_types = run_yolo_detection(INPUT_IMAGE)
#             if output_image_path:
#                 st.markdown("🤖 **Predicted Image**")
#                 encoded_image = encodeImageIntoBase64(output_image_path)
#                 decoded_image = base64.b64decode(encoded_image)
#                 st.image(decoded_image, width=660)
#                 os.remove(INPUT_IMAGE)
                
#                 st.markdown("</div>", unsafe_allow_html=True)
#             else:
#                 st.markdown("🤖 **Predicted Image**")
#                 st.info("Prediction failed. Please try uploading a different image.")

#     # Initialize session state for selected image
#     if 'selected_image_path' not in st.session_state:
#         st.session_state.selected_image_path = None

    
#     # Sample images for prediction
#     normal_images = [
#         "sample_images/Chilli.jpg", "sample_images/DrinkCan.jpg", 
#         "sample_images/DrinkPack.jpg", "sample_images/FoodCan.jpg", 
#         "sample_images/Lettuce.jpg", "sample_images/PaperBag.jpg", 
#         "sample_images/PlasticBag.jpg", "sample_images/PlasticContainer.jpg", 
#         "sample_images/SweetPotato.jpg", "sample_images/TeaBag.jpg", 
#         "sample_images/TissueRoll.jpg", "sample_images/Mixed.jpg", 
#     ]

#     st.title("Select Sample Images for Prediction 🖼️")

#     # Display Normal Images
#     with st.expander("📂 Normal Images For Prediction"):
#         # Create a grid of columns for better layout
#         cols = st.columns(4)  # 4 columns for better spacing

#         for i, img_path in enumerate(normal_images):
#             with cols[i % 4]:  # Cycle through columns
#                 # Display image
#                 st.image(img_path, caption=f"Sample {i+1}", use_container_width=True)

#                 # Add selection button
#                 if st.button(f"Select Image {i+1}", key=f"select_{i}"):

#                     # Copy the selected sample image to input directory
#                     shutil.copy(img_path, INPUT_IMAGE)
#                     st.session_state.selected_image_path = img_path
#                     st.success(f"📸 Selected: {os.path.basename(img_path)}")

#     # Process the selected sample image
#     if st.session_state.selected_image_path:
#         col1, col2 = st.columns(2)

#         with col1:
#             st.markdown("### 📸 ***Selected Sample Image***")
#             st.image(st.session_state.selected_image_path, width=660)

#         with col2:
#             output_image_path, waste_types = run_yolo_detection(INPUT_IMAGE)

#             if output_image_path:
#                 st.markdown("### 🤖 ***Predicted Image***")
#                 encoded_image = encodeImageIntoBase64(output_image_path)
#                 decoded_image = base64.b64decode(encoded_image)
#                 st.image(decoded_image, width=660)

#                 # Clean up input image after processing
#                 os.remove(INPUT_IMAGE)
#                 st.session_state.selected_image_path = None
#             else:
#                 st.markdown("🤖 **Predicted Image**")
#                 st.info("Prediction failed. Please try selecting a different image.")
            

#     # Provide a user-friendly reset button
#     if st.button("🔄 Reset Application", help="Clear all uploaded images and reset the app"):
#         try:
#             reset_directories()
#             st.session_state.reset_triggered = True
#             st.success("✔ Application Reset Successfully!")
#         except Exception as e:
#             st.error(f"⚠️ Reset failed: {e}")

# if __name__ == "__main__":
#     main()