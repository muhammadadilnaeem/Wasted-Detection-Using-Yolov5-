

import streamlit as st
import base64
import os
import shutil
from pathlib import Path
import glob
import pandas as pd

# Set page configuration as the first Streamlit command
st.set_page_config(
    page_title="WasteWise AI 🌍", 
    page_icon="♻️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Custom CSS for enhanced styling
def local_css():
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

# Helper functions from previous implementation
def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, "wb") as f:
        f.write(imgdata)

def encodeImageIntoBase64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read())

def get_latest_output_dir():
    exp_dirs = sorted(glob.glob(f"{YOLO_PATH}/runs/detect/exp*"), key=os.path.getmtime)
    return Path(exp_dirs[-1]) if exp_dirs else None

def parse_detection_results(output_dir):
    """Parse YOLO detection results to extract waste type counts"""
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
    # Error handling for missing directories/weights
    if not YOLO_PATH.exists():
        st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
        return False, None

    if not MODEL_WEIGHTS.exists():
        st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
        return False, None

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
    shutil.rmtree(INPUT_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def waste_type_mapping():
    """Map numeric labels to waste types"""
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

def main():
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

    # Main Application Title
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
                
                # # Display waste type statistics
                # st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                # st.markdown("### 📊 Waste Composition")
                
                # if waste_types:
                #     waste_mapping = waste_type_mapping()
                    
                #     # Create DataFrame for waste types
                #     waste_df = pd.DataFrame.from_dict(
                #         {waste_mapping[k]: v for k, v in waste_types.items() if k in waste_mapping}, 
                #         orient='index', 
                #         columns=['Count']
                #     ).reset_index()
                #     waste_df.columns = ['Waste Type', 'Count']
                    
                #     # Display table
                #     st.table(waste_df)
                    
                #     # Pie chart of waste composition
                #     st.bar_chart(waste_df.set_index('Waste Type'))
                # else:
                #     st.info("No specific waste types detected.")
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("🤖 **Predicted Image**")
                st.info("Prediction failed. Please try uploading a different image.")

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

# import streamlit as st
# import base64
# import os
# import shutil
# from pathlib import Path
# import glob

# # Set page configuration as the first Streamlit command
# st.set_page_config(
#     page_title="Waste Detection AI 🚮", 
#     page_icon="🤖", 
#     layout="wide"
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
#         background-color: #f0f2f6;
#         font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#     }
    
#     /* Stylish Headings */
#     h1 {
#         color: #00b894;
#         text-align: center;
#         font-weight: bold;
#         margin-bottom: 20px;
#     }
    
#     /* Instruction Box Styling */
#     .instruction-box {
#         background-color: #ecf0f1;
#         border-left: 5px solid #3498db;
#         padding: 15px;
#         border-radius: 5px;
#         margin-bottom: 20px;
#     }
    
#     /* Enhanced Reset Button */
#     .stButton>button {
#         background-color: #e74c3c;
#         color: white;
#         border: none;
#         padding: 10px 20px;
#         border-radius: 5px;
#         font-weight: bold;
#         transition: all 0.3s ease;
#         display: block;
#         margin: 20px auto;
#     }
    
#     .stButton>button:hover {
#         background-color: #c0392b;
#         transform: scale(1.05);
#     }
    
#     /* Image Container Styling */
#     .image-container {
#         border: 2px dashed #95a5a6;
#         border-radius: 10px;
#         padding: 15px;
#         text-align: center;
#         margin-bottom: 20px;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Helper functions for image encoding/decoding and getting latest output directory
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

# # Run YOLOv5 detection on the input image
# def run_yolo_detection(input_image_path):
#     # Error handling for missing directories/weights
#     if not YOLO_PATH.exists():
#         st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
#         return False

#     if not MODEL_WEIGHTS.exists():
#         st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
#         return False

#     detect_command = (
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} "
#         f"--img 416 --conf 0.5 --source {input_image_path} "
#         f"--save-txt --save-conf"
#     )
    
#     try:
#         # Use subprocess for better error handling
#         import subprocess
#         result = subprocess.run(detect_command, shell=True, capture_output=True, text=True)
        
#         # Log command output for debugging
#         if result.returncode != 0:
#             st.error(f"Detection command failed: {result.stderr}")
#             return False

#     except Exception as e:
#         st.error(f"An error occurred during detection: {e}")
#         return False

#     latest_output_dir = get_latest_output_dir()
#     if latest_output_dir:
#         detected_image_path = latest_output_dir / "inputImage.jpg"
#         if detected_image_path.exists():
#             output_image_path = OUTPUT_DIR / detected_image_path.name
#             shutil.copy(detected_image_path, output_image_path)
#             return output_image_path
    
#     st.error("Detected image not found. Detection may have failed.")
#     return False

# # Reset the application by clearing input and output directories
# def reset_directories():
#     shutil.rmtree(INPUT_DIR, ignore_errors=True)
#     shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
#     INPUT_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# def main():
#     # Apply custom CSS
#     local_css()

#     # Display the title
#     st.markdown("<h1 style='text-align: center;'> 🚮 Intelligent Waste Detection System 🤖</h1>", unsafe_allow_html=True)

#     # Provide clear instructions for the user
#     st.markdown("""
#     <div class='instruction-box'>
#     📝 <strong>How to Use:</strong>
#     <ul>
#         <li>Upload an image of waste materials</li>
#         <li>Our AI will detect and classify waste types</li>
#         <li>View predictions side-by-side</li>
#         <li>Use the reset button to start over</li>
#     </ul>
#     </div>
#     """, unsafe_allow_html=True)

#     # Initialize session state for reset functionality
#     if "reset_triggered" not in st.session_state:
#         st.session_state.reset_triggered = False

#     # Display a success message if the application was reset
#     if st.session_state.reset_triggered:
#         st.success("✅ Application reset successfully! Upload a new image to start fresh.")
#         st.session_state.reset_triggered = False

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
#             output_image_path = run_yolo_detection(INPUT_IMAGE)
#             if output_image_path:
#                 st.markdown("🤖 **Predicted Image**")
#                 encoded_image = encodeImageIntoBase64(output_image_path)
#                 decoded_image = base64.b64decode(encoded_image)
#                 st.image(decoded_image, width=660)
#                 os.remove(INPUT_IMAGE)
#             else:
#                 st.markdown("🤖 **Predicted Image**")
#                 st.info("Prediction failed. Please try uploading a different image.")

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


# (Heading Adjustment needed)

# import streamlit as st
# import base64
# import os
# import shutil
# from pathlib import Path
# import glob

# # Set page configuration as the first Streamlit command
# st.set_page_config(
#     page_title="Waste Detection AI 🚮", 
#     page_icon="🤖", 
#     layout="wide"
# )

# # Rest of the imports remain the same
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
#         background-color: #f0f2f6;
#         font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#     }
    
#     /* Stylish Headings */
#     h1 {
#         color: #00b894;  /* Updated heading color to a vibrant teal */
#         text-align: center;
#         font-weight: bold;
#         background: linear-gradient(to right, #3498db, #2ecc71);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin-bottom: 20px;
#     }
    
#     /* Instruction Box Styling */
#     .instruction-box {
#         background-color: #ecf0f1;
#         border-left: 5px solid #3498db;
#         padding: 15px;
#         border-radius: 5px;
#         margin-bottom: 20px;
#     }
    
#     /* Enhanced Reset Button */
#     .stButton>button {
#         background-color: #e74c3c;
#         color: white;
#         border: none;
#         padding: 10px 20px;
#         border-radius: 5px;
#         font-weight: bold;
#         transition: all 0.3s ease;
#         display: block;
#         margin: 20px auto;
#     }
    
#     .stButton>button:hover {
#         background-color: #c0392b;
#         transform: scale(1.05);
#     }
    
#     /* Image Container Styling */
#     .image-container {
#         border: 2px dashed #95a5a6;
#         border-radius: 10px;
#         padding: 15px;
#         text-align: center;
#         margin-bottom: 20px;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Remaining functions unchanged from previous version
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

# def run_yolo_detection(input_image_path):
#     # Error handling for missing directories/weights
#     if not YOLO_PATH.exists():
#         st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
#         return False

#     if not MODEL_WEIGHTS.exists():
#         st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
#         return False

#     detect_command = (
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} "
#         f"--img 416 --conf 0.5 --source {input_image_path} "
#         f"--save-txt --save-conf"
#     )
    
#     try:
#         # Use subprocess for better error handling
#         import subprocess
#         result = subprocess.run(detect_command, shell=True, capture_output=True, text=True)
        
#         # Log command output for debugging
#         if result.returncode != 0:
#             st.error(f"Detection command failed: {result.stderr}")
#             return False

#     except Exception as e:
#         st.error(f"An error occurred during detection: {e}")
#         return False

#     latest_output_dir = get_latest_output_dir()
#     if latest_output_dir:
#         detected_image_path = latest_output_dir / "inputImage.jpg"
#         if detected_image_path.exists():
#             output_image_path = OUTPUT_DIR / detected_image_path.name
#             shutil.copy(detected_image_path, output_image_path)
#             return output_image_path
    
#     st.error("Detected image not found. Detection may have failed.")
#     return False

# def reset_directories():
#     shutil.rmtree(INPUT_DIR, ignore_errors=True)
#     shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
#     INPUT_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# def main():
#     # Apply custom CSS
#     local_css()

#     # Animated Title with Gradient
#     st.markdown("""
#     <h1 style='text-align: center;'>
#     🚮 Intelligent Waste Detection System 🤖
#     </h1>
#     """, unsafe_allow_html=True)

#     # Enhanced Instruction Box
#     st.markdown("""
#     <div class='instruction-box'>
#     📝 <strong>How to Use:</strong>
#     <ul>
#         <li>Upload an image of waste materials</li>
#         <li>Our AI will detect and classify waste types</li>
#         <li>View predictions side-by-side</li>
#         <li>Use the reset button to start over</li>
#     </ul>
#     </div>
#     """, unsafe_allow_html=True)

#     # Initialize session state for reset functionality
#     if "reset_triggered" not in st.session_state:
#         st.session_state.reset_triggered = False

#     # If reset was triggered, show success message
#     if st.session_state.reset_triggered:
#         st.success("✅ Application reset successfully! Upload a new image to start fresh.")
#         st.session_state.reset_triggered = False

#     # Image Upload with Enhanced Styling
#     st.markdown("<div class='image-container'>", unsafe_allow_html=True)
#     uploaded_file = st.file_uploader(
#         "🔍 Upload Waste Image", 
#         type=["jpg", "png", "jpeg"], 
#         help="Supported formats: JPG, PNG, JPEG",
#         accept_multiple_files=False
#     )
#     st.markdown("</div>", unsafe_allow_html=True)

#     if uploaded_file:
#         with open(INPUT_IMAGE, "wb") as f:
#             f.write(uploaded_file.read())

#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("📸 **Uploaded Image**")
#             st.image(str(INPUT_IMAGE), use_container_width=True)

#         with col2:
#             output_image_path = run_yolo_detection(INPUT_IMAGE)
#             if output_image_path:
#                 st.markdown("🤖 **Predicted Image**")
#                 encoded_image = encodeImageIntoBase64(output_image_path)
#                 decoded_image = base64.b64decode(encoded_image)
#                 st.image(decoded_image, use_container_width=True)
#                 os.remove(INPUT_IMAGE)
#             else:
#                 st.markdown("🤖 **Predicted Image**")
#                 st.info("Prediction failed. Please try uploading a different image.")

#     # Enhanced Reset Button
#     if st.button("🔄 Reset Application", help="Clear all uploaded images and reset the app"):
#         try:
#             reset_directories()
#             st.session_state.reset_triggered = True
#             st.success("✔ Application Reset Successfully!")
#         except Exception as e:
#             st.error(f"⚠️ Reset failed: {e}")

# if __name__ == "__main__":
#     main()


# (color update)


# import streamlit as st
# import base64
# import os
# import shutil
# from pathlib import Path
# import glob

# # Set page configuration IMMEDIATELY after importing streamlit
# st.set_page_config(
#     page_title="Waste Detection AI 🚮", 
#     page_icon="🤖", 
#     layout="wide"
# )

# # Rest of the imports and code remain the same as previous version
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
#         background-color: #f0f2f6;
#         font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#     }
    
#     /* Stylish Headings */
#     h1 {
#         color: #2c3e50;
#         text-align: center;
#         font-weight: bold;
#         background: linear-gradient(to right, #3498db, #2ecc71);
#         -webkit-background-clip: text;
#         -webkit-text-fill-color: transparent;
#         margin-bottom: 20px;
#     }
    
#     /* Instruction Box Styling */
#     .instruction-box {
#         background-color: #ecf0f1;
#         border-left: 5px solid #3498db;
#         padding: 15px;
#         border-radius: 5px;
#         margin-bottom: 20px;
#     }
    
#     /* Enhanced Reset Button */
#     .stButton>button {
#         background-color: #e74c3c;
#         color: white;
#         border: none;
#         padding: 10px 20px;
#         border-radius: 5px;
#         font-weight: bold;
#         transition: all 0.3s ease;
#         display: block;
#         margin: 20px auto;
#     }
    
#     .stButton>button:hover {
#         background-color: #c0392b;
#         transform: scale(1.05);
#     }
    
#     /* Image Container Styling */
#     .image-container {
#         border: 2px dashed #95a5a6;
#         border-radius: 10px;
#         padding: 15px;
#         text-align: center;
#         margin-bottom: 20px;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Rest of the functions remain the same as in the previous version
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

# def run_yolo_detection(input_image_path):
#     # Error handling for missing directories/weights
#     if not YOLO_PATH.exists():
#         st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
#         return False

#     if not MODEL_WEIGHTS.exists():
#         st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
#         return False

#     detect_command = (
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} "
#         f"--img 416 --conf 0.5 --source {input_image_path} "
#         f"--save-txt --save-conf"
#     )
    
#     try:
#         # Use subprocess for better error handling
#         import subprocess
#         result = subprocess.run(detect_command, shell=True, capture_output=True, text=True)
        
#         # Log command output for debugging
#         if result.returncode != 0:
#             st.error(f"Detection command failed: {result.stderr}")
#             return False

#     except Exception as e:
#         st.error(f"An error occurred during detection: {e}")
#         return False

#     latest_output_dir = get_latest_output_dir()
#     if latest_output_dir:
#         detected_image_path = latest_output_dir / "inputImage.jpg"
#         if detected_image_path.exists():
#             output_image_path = OUTPUT_DIR / detected_image_path.name
#             shutil.copy(detected_image_path, output_image_path)
#             return output_image_path
    
#     st.error("Detected image not found. Detection may have failed.")
#     return False

# def reset_directories():
#     shutil.rmtree(INPUT_DIR, ignore_errors=True)
#     shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
#     INPUT_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# def main():
#     # Apply custom CSS
#     local_css()

#     # Animated Title with Gradient
#     st.markdown("""
#     <h1 style='text-align: center; background-color: blue;'>
#     🚮 Intelligent Waste Detection System 🤖
#     </h1>
#     """, unsafe_allow_html=True)

#     # Enhanced Instruction Box
#     st.markdown("""
#     <div class='instruction-box'>
#     📝 <strong>How to Use:</strong>
#     <ul>
#         <li>Upload an image of waste materials</li>
#         <li>Our AI will detect and classify waste types</li>
#         <li>View predictions side-by-side</li>
#         <li>Use the reset button to start over</li>
#     </ul>
#     </div>
#     """, unsafe_allow_html=True)

#     # Initialize session state for reset functionality
#     if "reset_triggered" not in st.session_state:
#         st.session_state.reset_triggered = False

#     # If reset was triggered, show success message
#     if st.session_state.reset_triggered:
#         st.success("✅ Application reset successfully! Upload a new image to start fresh.")
#         st.session_state.reset_triggered = False

#     # Image Upload with Enhanced Styling
#     st.markdown("<div class='image-container'>", unsafe_allow_html=True)
#     uploaded_file = st.file_uploader(
#         "🔍 Upload Waste Image", 
#         type=["jpg", "png", "jpeg"], 
#         help="Supported formats: JPG, PNG, JPEG",
#         accept_multiple_files=False
#     )
#     st.markdown("</div>", unsafe_allow_html=True)

#     if uploaded_file:
#         with open(INPUT_IMAGE, "wb") as f:
#             f.write(uploaded_file.read())

#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("📸 **Uploaded Image**")
#             st.image(str(INPUT_IMAGE), use_container_width=True)

#         with col2:
#             output_image_path = run_yolo_detection(INPUT_IMAGE)
#             if output_image_path:
#                 st.markdown("🤖 **Predicted Image**")
#                 encoded_image = encodeImageIntoBase64(output_image_path)
#                 decoded_image = base64.b64decode(encoded_image)
#                 st.image(decoded_image, use_container_width=True)
#                 os.remove(INPUT_IMAGE)
#             else:
#                 st.markdown("🤖 **Predicted Image**")
#                 st.info("Prediction failed. Please try uploading a different image.")

#     # Enhanced Reset Button
#     if st.button("🔄 Reset Application", help="Clear all uploaded images and reset the app"):
#         try:
#             reset_directories()
#             st.session_state.reset_triggered = True
#             st.success("✔ Application Reset Successfully!")
#         except Exception as e:
#             st.error(f"⚠️ Reset failed: {e}")

# if __name__ == "__main__":
#     main()



# (wORKING gOOD)

# import streamlit as st
# import base64
# import os
# import shutil
# from pathlib import Path
# import glob

# # Define input and output directories
# PROJECT_ROOT = Path(__file__).parent.resolve()
# INPUT_DIR = PROJECT_ROOT / "data/input/"
# OUTPUT_DIR = PROJECT_ROOT / "data/output/"
# INPUT_IMAGE = INPUT_DIR / "inputImage.jpg"
# YOLO_PATH = PROJECT_ROOT / "yolov5/"
# MODEL_WEIGHTS = PROJECT_ROOT / "model/best.pt"

# # Create directories if they don't exist
# INPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Function to decode a base64 image string and save it to a file
# def decodeImage(imgstring, fileName):
#     imgdata = base64.b64decode(imgstring)
#     with open(fileName, "wb") as f:
#         f.write(imgdata)

# # Function to encode an image file into a base64 string
# def encodeImageIntoBase64(image_path):
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read())

# # Function to get the latest YOLOv5 output directory
# def get_latest_output_dir():
#     exp_dirs = sorted(glob.glob(f"{YOLO_PATH}/runs/detect/exp*"), key=os.path.getmtime)
#     return Path(exp_dirs[-1]) if exp_dirs else None

# # Function to run YOLOv5 detection
# def run_yolo_detection(input_image_path):
#     if not YOLO_PATH.exists():
#         st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
#         return False

#     if not MODEL_WEIGHTS.exists():
#         st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
#         return False

#     detect_command = (
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} "
#         f"--img 416 --conf 0.5 --source {input_image_path} "
#         f"--save-txt --save-conf"
#     )
#     os.system(detect_command)

#     latest_output_dir = get_latest_output_dir()
#     if latest_output_dir:
#         detected_image_path = latest_output_dir / "inputImage.jpg"
#         if detected_image_path.exists():
#             output_image_path = OUTPUT_DIR / detected_image_path.name
#             shutil.copy(detected_image_path, output_image_path)
#             return output_image_path
#     st.error("Detected image not found. Detection may have failed.")
#     return False

# # Function to reset input and output directories
# def reset_directories():
#     shutil.rmtree(INPUT_DIR, ignore_errors=True)
#     shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
#     INPUT_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# def main():
#     # Set page configuration
#     st.set_page_config(page_title="Waste Detection App", page_icon="🚮", layout="wide")

#     # App Title and Introduction
#     st.title("🚮 Waste Detection App 🚮")
#     st.subheader("Detect and Analyze Waste in Images Using AI 🤖")

#     # Initialize session state for reset functionality
#     if "reset_triggered" not in st.session_state:
#         st.session_state.reset_triggered = False

#     # If reset was triggered, show success message
#     if st.session_state.reset_triggered:
#         st.success("✅ Application reset successfully! Upload a new image to start fresh.")
#         st.session_state.reset_triggered = False  # Reset the state after displaying the message

#     # User Instructions
#     st.info("📝 Instructions: Upload an image of waste, and our AI will detect and classify waste types. View predictions side-by-side.")

#     # Image Upload and Display
#     uploaded_file = st.file_uploader("🔍 Upload Waste Image", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

#     if uploaded_file:
#         with open(INPUT_IMAGE, "wb") as f:
#             f.write(uploaded_file.read())

#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("📸 Uploaded Image")
#             st.image(str(INPUT_IMAGE), use_container_width=True)

#         with col2:
#             output_image_path = run_yolo_detection(INPUT_IMAGE)
#             if output_image_path:
#                 st.markdown("🤖 Predicted Image")
#                 encoded_image = encodeImageIntoBase64(output_image_path)
#                 decoded_image = base64.b64decode(encoded_image)
#                 st.image(decoded_image, use_container_width=True)
#                 os.remove(INPUT_IMAGE)
#             else:
#                 st.markdown("🤖 Predicted Image")
#                 st.info("Prediction failed. Please try uploading a different image.")

#     # Reset Button
#     if st.button("🔄 Reset Application", help="Clear all uploaded images and reset the app"):
#         try:
#             reset_directories()
#             st.session_state.reset_triggered = True
#             st.success("✔ Reset Successfull...")
#         except Exception as e:
#             st.error(f"⚠️ Reset failed: {e}")

# if __name__ == "__main__":
#     main()


# import streamlit as st
# import base64
# import os
# import shutil
# from pathlib import Path
# import glob

# # Define input and output directories
# PROJECT_ROOT = Path(__file__).parent.resolve()
# INPUT_DIR = PROJECT_ROOT / "data/input/"
# OUTPUT_DIR = PROJECT_ROOT / "data/output/"
# INPUT_IMAGE = INPUT_DIR / "inputImage.jpg"
# YOLO_PATH = PROJECT_ROOT / "yolov5/"
# MODEL_WEIGHTS = PROJECT_ROOT / "model/best.pt"

# # Create directories if they don't exist
# INPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Function to decode a base64 image string and save it to a file
# def decodeImage(imgstring, fileName):
#     imgdata = base64.b64decode(imgstring)
#     with open(fileName, "wb") as f:
#         f.write(imgdata)

# # Function to encode an image file into a base64 string
# def encodeImageIntoBase64(image_path):
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read())

# # Function to get the latest YOLOv5 output directory
# def get_latest_output_dir():
#     exp_dirs = sorted(glob.glob(f"{YOLO_PATH}/runs/detect/exp*"), key=os.path.getmtime)
#     return Path(exp_dirs[-1]) if exp_dirs else None

# # Function to run YOLOv5 detection
# def run_yolo_detection(input_image_path):
#     if not YOLO_PATH.exists():
#         st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
#         return False

#     if not MODEL_WEIGHTS.exists():
#         st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
#         return False

#     detect_command = (
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} "
#         f"--img 416 --conf 0.5 --source {input_image_path} "
#         f"--save-txt --save-conf"
#     )
#     os.system(detect_command)

#     latest_output_dir = get_latest_output_dir()
#     if latest_output_dir:
#         detected_image_path = latest_output_dir / "inputImage.jpg"
#         if detected_image_path.exists():
#             output_image_path = OUTPUT_DIR / detected_image_path.name
#             shutil.copy(detected_image_path, output_image_path)
#             return output_image_path
#     st.error("Detected image not found. Detection may have failed.")
#     return False

# # Function to reset input and output directories
# def reset_directories():
#     shutil.rmtree(INPUT_DIR, ignore_errors=True)
#     shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
#     INPUT_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# def main():
#     # Set page configuration
#     st.set_page_config(page_title="Waste Detection App", page_icon="🚮", layout="wide")

#     # App Title and Introduction
#     st.title("🚮 Waste Detection App 🚮")
#     st.subheader("Detect and Analyze Waste in Images Using AI 🤖")

#     # Initialize session state for reset functionality
#     if "reset" not in st.session_state:
#         st.session_state.reset = False

#     # Reset Application
#     if st.session_state.reset:
#         reset_directories()
#         st.session_state.reset = False

#     # User Instructions
#     st.info("📝 Instructions: Upload an image of waste, and our AI will detect and classify waste types. View predictions side-by-side.")

#     # Image Upload and Display
#     uploaded_file = st.file_uploader("🔍 Upload Waste Image", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

#     if uploaded_file:
#         with open(INPUT_IMAGE, "wb") as f:
#             f.write(uploaded_file.read())

#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("📸 Uploaded Image")
#             st.image(str(INPUT_IMAGE), use_container_width=True)

#         with col2:
#             output_image_path = run_yolo_detection(INPUT_IMAGE)
#             if output_image_path:
#                 st.markdown("🤖 Predicted Image")
#                 encoded_image = encodeImageIntoBase64(output_image_path)
#                 decoded_image = base64.b64decode(encoded_image)
#                 st.image(decoded_image, use_container_width=True)
#                 os.remove(INPUT_IMAGE)
#             else:
#                 st.markdown("🤖 Predicted Image")
#                 st.info("Prediction failed. Please try uploading a different image.")

#     # Reset Button
#     if st.button("🔄 Reset Application", help="Clear all uploaded images and reset the app"):
#         st.session_state.reset = True
#         st.experimental_rerun()

# if __name__ == "__main__":
#     main()



# import streamlit as st
# import base64
# import os
# import shutil
# from pathlib import Path
# import glob

# # Define input and output directories
# PROJECT_ROOT = Path(__file__).parent.resolve()
# INPUT_DIR = PROJECT_ROOT / "data/input/"
# OUTPUT_DIR = PROJECT_ROOT / "data/output/"
# INPUT_IMAGE = INPUT_DIR / "inputImage.jpg"
# YOLO_PATH = PROJECT_ROOT / "yolov5/"
# MODEL_WEIGHTS = PROJECT_ROOT / "model/best.pt"

# # Create directories if they don't exist
# INPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Function to decode a base64 image string and save it to a file
# def decodeImage(imgstring, fileName):
#     imgdata = base64.b64decode(imgstring)
#     with open(fileName, "wb") as f:
#         f.write(imgdata)

# # Function to encode an image file into a base64 string
# def encodeImageIntoBase64(image_path):
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read())

# # Function to get the latest YOLOv5 output directory
# def get_latest_output_dir():
#     exp_dirs = sorted(glob.glob(f"{YOLO_PATH}/runs/detect/exp*"), key=os.path.getmtime)
#     return Path(exp_dirs[-1]) if exp_dirs else None

# # Function to run YOLOv5 detection
# def run_yolo_detection(input_image_path):
#     if not YOLO_PATH.exists():
#         st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
#         return False

#     if not MODEL_WEIGHTS.exists():
#         st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
#         return False

#     detect_command = (
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} "
#         f"--img 416 --conf 0.5 --source {input_image_path} "
#         f"--save-txt --save-conf"
#     )
#     os.system(detect_command)

#     latest_output_dir = get_latest_output_dir()
#     if latest_output_dir:
#         detected_image_path = latest_output_dir / "inputImage.jpg"
#         if detected_image_path.exists():
#             output_image_path = OUTPUT_DIR / detected_image_path.name
#             shutil.copy(detected_image_path, output_image_path)
#             return output_image_path
#     st.error("Detected image not found. Detection may have failed.")
#     return False

# # Function to reset input and output directories
# def reset_directories():
#     shutil.rmtree(INPUT_DIR, ignore_errors=True)
#     shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
#     INPUT_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# def main():
#     # Set page configuration
#     st.set_page_config(page_title="Waste Detection App", page_icon="🚮", layout="wide")

#     # App Title and Introduction
#     st.title("🚮 Waste Detection App 🚮")
#     st.subheader("Detect and Analyze Waste in Images Using AI 🤖")

#     # User Instructions
#     st.info("📝 Instructions: Upload an image of waste, and our AI will detect and classify waste types. View predictions side-by-side.")

#     # Image Upload and Display
#     uploaded_file = st.file_uploader("🔍 Upload Waste Image", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

#     if uploaded_file:
#         reset_directories()
#         with open(INPUT_IMAGE, "wb") as f:
#             f.write(uploaded_file.read())

#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("📸 Uploaded Image")
#             st.image(str(INPUT_IMAGE), use_container_width=True)

#         with col2:
#             output_image_path = run_yolo_detection(INPUT_IMAGE)
#             if output_image_path:
#                 st.markdown("🤖 Predicted Image")
#                 encoded_image = encodeImageIntoBase64(output_image_path)
#                 decoded_image = base64.b64decode(encoded_image)
#                 st.image(decoded_image, use_container_width=True)
#                 os.remove(INPUT_IMAGE)
#             else:
#                 st.markdown("🤖 Predicted Image")
#                 st.info("Prediction failed. Please try uploading a different image.")

#     # Reset Button
#     if st.button("🔄 Reset Application", help="Clear all uploaded images and reset the app"):
#         reset_directories()
#         st.experimental_rerun()

# if __name__ == "__main__":
#     main()

# import os
# import glob
# import shutil
# from PIL import Image
# from pathlib import Path
# import streamlit as st

# # Define directories
# PROJECT_ROOT = Path(__file__).parent.resolve()
# INPUT_DIR = PROJECT_ROOT / "data/input/"
# OUTPUT_DIR = PROJECT_ROOT / "data/output/"
# INPUT_IMAGE = INPUT_DIR / "inputImage.jpg"
# YOLO_PATH = PROJECT_ROOT / "yolov5/"
# MODEL_WEIGHTS = PROJECT_ROOT / "model/best.pt"

# # Create directories if they don't exist
# INPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Function to run YOLOv5 detection
# def run_yolo_detection(input_image_path):
#     if not YOLO_PATH.exists():
#         st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
#         return False

#     if not MODEL_WEIGHTS.exists():
#         st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
#         return False

#     detect_command = (
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} "
#         f"--img 416 --conf 0.5 --source {input_image_path} "
#         f"--save-txt --save-conf"
#     )
#     os.system(detect_command)

#     latest_output_dir = get_latest_output_dir(YOLO_PATH)
#     if latest_output_dir:
#         detected_image_path = latest_output_dir / "inputImage.jpg"
#         if detected_image_path.exists():
#             output_image_path = OUTPUT_DIR / detected_image_path.name
#             shutil.copy(detected_image_path, output_image_path)
#             return output_image_path
#     st.error("Detected image not found. Detection may have failed.")
#     return False

# # Function to get the latest YOLOv5 output directory
# def get_latest_output_dir(yolo_path):
#     exp_dirs = sorted(glob.glob(f"{yolo_path}/runs/detect/exp*"), key=os.path.getmtime)
#     return Path(exp_dirs[-1]) if exp_dirs else None

# # Function to reset input and output directories
# def reset_directories():
#     shutil.rmtree(INPUT_DIR, ignore_errors=True)
#     shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
#     INPUT_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# def main():
#     # Set page configuration
#     st.set_page_config(
#         page_title="Waste Detection App",
#         page_icon="🚮",
#         layout="wide",
#         initial_sidebar_state="collapsed"
#     )

#     # App Title and Introduction
#     st.markdown(
#         """
#         <div style="background-color: #0072C6; color: white; padding: 2rem; text-align: center;">
#             <h1>🚮 Waste Detection App 🚮</h1>
#             <p>Detect and Analyze Waste in Images Using AI 🤖</p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     # Image Upload and Display
#     st.subheader("📸 Uploaded Image")
#     uploaded_file = st.file_uploader("🔍 Upload Waste Image", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

#     if "uploaded_image" not in st.session_state:
#         st.session_state["uploaded_image"] = None
#     if "predicted_image" not in st.session_state:
#         st.session_state["predicted_image"] = None

#     if uploaded_file:
#         reset_directories()
#         with open(INPUT_IMAGE, "wb") as f:
#             f.write(uploaded_file.read())

#         st.session_state["uploaded_image"] = INPUT_IMAGE
#         output_image_path = run_yolo_detection(INPUT_IMAGE)

#         if output_image_path:
#             st.session_state["predicted_image"] = output_image_path

#     # Display Uploaded Image
#     if st.session_state["uploaded_image"]:
#         st.image(str(st.session_state["uploaded_image"]), caption="Uploaded Image", use_column_width=True)

#     # Display Predicted Image
#     st.subheader("🤖 Predicted Image")
#     if st.session_state["predicted_image"]:
#         st.image(str(st.session_state["predicted_image"]), caption="Predicted Image", use_column_width=True)

#     # Reset Button
#     if st.button("🔄 Reset Application", help="Clear all uploaded images and reset the app"):
#         reset_directories()
#         st.session_state["uploaded_image"] = None
#         st.session_state["predicted_image"] = None
#         st.experimental_set_query_params()  # Refresh the app without `experimental_rerun`

# if __name__ == "__main__":
#     main()


# import os
# import glob
# import shutil
# import base64
# from PIL import Image
# from pathlib import Path
# import streamlit as st


# # Define directories
# PROJECT_ROOT = Path(__file__).parent.resolve()
# INPUT_DIR = PROJECT_ROOT / "data/input/"
# OUTPUT_DIR = PROJECT_ROOT / "data/output/"
# INPUT_IMAGE = INPUT_DIR / "inputImage.jpg"
# YOLO_PATH = PROJECT_ROOT / "yolov5/"
# MODEL_WEIGHTS = PROJECT_ROOT / "model/best.pt"

# # Create directories if they don't exist
# INPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Function to run YOLOv5 detection
# def run_yolo_detection(input_image_path):
#     if not YOLO_PATH.exists():
#         st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
#         return False

#     if not MODEL_WEIGHTS.exists():
#         st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
#         return False

#     detect_command = (
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} "
#         f"--img 416 --conf 0.5 --source {input_image_path} "
#         f"--save-txt --save-conf"
#     )
#     os.system(detect_command)

#     latest_output_dir = get_latest_output_dir(YOLO_PATH)
#     if latest_output_dir:
#         detected_image_path = latest_output_dir / "inputImage.jpg"
#         if detected_image_path.exists():
#             output_image_path = OUTPUT_DIR / detected_image_path.name
#             shutil.copy(detected_image_path, output_image_path)
#             return output_image_path
#     st.error("Detected image not found. Detection may have failed.")
#     return False

# # Function to get the latest YOLOv5 output directory
# def get_latest_output_dir(yolo_path):
#     exp_dirs = sorted(glob.glob(f"{yolo_path}/runs/detect/exp*"), key=os.path.getmtime)
#     return Path(exp_dirs[-1]) if exp_dirs else None

# # Function to reset input and output directories
# def reset_directories():
#     shutil.rmtree(INPUT_DIR, ignore_errors=True)
#     shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
#     INPUT_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# def main():
#     # Set page configuration
#     st.set_page_config(
#         page_title="Waste Detection App",
#         page_icon="🚮",
#         layout="wide",
#         initial_sidebar_state="collapsed"
#     )

#     # App Title and Introduction
#     st.markdown(
#         """
#         <div style="background-color: #0072C6; color: white; padding: 2rem; text-align: center;">
#             <h1>🚮 Waste Detection App 🚮</h1>
#             <p>Detect and Analyze Waste in Images Using AI 🤖</p>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     # User Instructions
#     st.markdown(
#         """
#         <div style="background-color: #F4F4F9; padding: 1rem; border-radius: 0.5rem; margin-top: 2rem;">
#             <h2 style="color: #2C3E50;">📝 Instructions:</h2>
#             <ul>
#                 <li>Upload an image of waste</li>
#                 <li>Our AI will detect and classify waste types</li>
#                 <li>View predictions side-by-side</li>
#             </ul>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     # Image Upload and Display
#     st.markdown(
#         """
#         <div style="display: flex; justify-content: center; margin-top: 2rem;">
#             <div style="width: 45%; background-color: #ECF0F1; padding: 1rem; border-radius: 0.5rem; margin-right: 1rem;">
#                 <h2 style="color: #2C3E50;">📸 Uploaded Image</h2>
#                 <div style="height: 400px; display: flex; justify-content: center; align-items: center;">
#                     <img id="uploaded-image" src="" style="max-width: 100%; max-height: 100%; display: none;">
#                 </div>
#             </div>
#             <div style="width: 45%; background-color: #ECF0F1; padding: 1rem; border-radius: 0.5rem; margin-left: 1rem;">
#                 <h2 style="color: #2C3E50;">🤖 Predicted Image</h2>
#                 <div style="height: 400px; display: flex; justify-content: center; align-items: center;">
#                     <img id="predicted-image" src="" style="max-width: 100%; max-height: 100%; display: none;">
#                 </div>
#             </div>
#         </div>
#         """,
#         unsafe_allow_html=True
#     )

#     uploaded_file = st.file_uploader("🔍 Upload Waste Image", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

#     if uploaded_file:
#         reset_directories()
#         with open(INPUT_IMAGE, "wb") as f:
#             f.write(uploaded_file.read())

#         output_image_path = run_yolo_detection(INPUT_IMAGE)
#         if output_image_path:
#             uploaded_image = Image.open(str(INPUT_IMAGE))
#             predicted_image = Image.open(str(output_image_path))

#             st.markdown(
#                 f"""
#                 <script>
#                 document.getElementById('uploaded-image').src = '{base64.b64encode(uploaded_image.tobytes()).decode("utf-8")}';
#                 document.getElementById('uploaded-image').style.display = 'block';
#                 document.getElementById('predicted-image').src = '{base64.b64encode(predicted_image.tobytes()).decode("utf-8")}';
#                 document.getElementById('predicted-image').style.display = 'block';
#                 </script>
#                 """,
#                 unsafe_allow_html=True
#             )

#             os.remove(INPUT_IMAGE)

#     # Reset Button
#     if st.button("🔄 Reset Application", help="Clear all uploaded images and reset the app"):
#         reset_directories()
#         st.experimental_rerun()

# if __name__ == "__main__":
#     main()

# import streamlit as st
# import base64
# import os
# import shutil
# from pathlib import Path
# import glob

# # Define input and output directories
# PROJECT_ROOT = Path(__file__).parent.resolve()
# INPUT_DIR = PROJECT_ROOT / "data/input/"
# OUTPUT_DIR = PROJECT_ROOT / "data/output/"
# INPUT_IMAGE = INPUT_DIR / "inputImage.jpg"
# YOLO_PATH = PROJECT_ROOT / "yolov5/"
# MODEL_WEIGHTS = PROJECT_ROOT / "model/best.pt"

# # Create directories if they don't exist
# INPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Function to decode a base64 image string and save it to a file
# def decodeImage(imgstring, fileName):
#     imgdata = base64.b64decode(imgstring)
#     with open(fileName, "wb") as f:
#         f.write(imgdata)

# # Function to encode an image file into a base64 string
# def encodeImageIntoBase64(image_path):
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read())

# # Function to get the latest YOLOv5 output directory
# def get_latest_output_dir():
#     exp_dirs = sorted(glob.glob(f"{YOLO_PATH}/runs/detect/exp*"), key=os.path.getmtime)
#     return Path(exp_dirs[-1]) if exp_dirs else None

# # Function to run YOLOv5 detection
# def run_yolo_detection(input_image_path):
#     if not YOLO_PATH.exists():
#         st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
#         return False

#     if not MODEL_WEIGHTS.exists():
#         st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
#         return False

#     detect_command = (
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} "
#         f"--img 416 --conf 0.5 --source {input_image_path} "
#         f"--save-txt --save-conf"
#     )
#     os.system(detect_command)

#     latest_output_dir = get_latest_output_dir()
#     if latest_output_dir:
#         detected_image_path = latest_output_dir / "inputImage.jpg"
#         if detected_image_path.exists():
#             output_image_path = OUTPUT_DIR / detected_image_path.name
#             shutil.copy(detected_image_path, output_image_path)
#             return output_image_path
#     st.error("Detected image not found. Detection may have failed.")
#     return False

# # Function to reset input and output directories
# def reset_directories():
#     shutil.rmtree(INPUT_DIR, ignore_errors=True)
#     shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
#     INPUT_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# def main():
#     # Set page configuration
#     st.set_page_config(page_title="Waste Detection App", page_icon="🚮", layout="wide")

#     # App Title and Introduction
#     st.title("🚮 Waste Detection App 🚮")
#     st.subheader("Detect and Analyze Waste in Images Using AI 🤖")

#     # User Instructions
#     st.info("📝 Instructions: Upload an image of waste, and our AI will detect and classify waste types. View predictions side-by-side.")

#     # Image Upload and Display
#     uploaded_file = st.file_uploader("🔍 Upload Waste Image", type=["jpg", "png", "jpeg"], accept_multiple_files=False)

#     if uploaded_file:
#         reset_directories()
#         with open(INPUT_IMAGE, "wb") as f:
#             f.write(uploaded_file.read())

#         col1, col2 = st.columns(2)
#         with col1:
#             st.markdown("📸 Uploaded Image")
#             st.image(str(INPUT_IMAGE), use_container_width=True)

#         with col2:
#             output_image_path = run_yolo_detection(INPUT_IMAGE)
#             if output_image_path:
#                 st.markdown("🤖 Predicted Image")
#                 encoded_image = encodeImageIntoBase64(output_image_path)
#                 decoded_image = base64.b64decode(encoded_image)
#                 st.image(decoded_image, use_container_width=True)
#                 os.remove(INPUT_IMAGE)

#     # Reset Button
#     if st.button("🔄 Reset Application", help="Clear all uploaded images and reset the app"):
#         reset_directories()
#         st.experimental_rerun()

# if __name__ == "__main__":
#     main()

# import streamlit as st
# import base64
# import os
# import shutil
# from pathlib import Path
# import glob

# # Define input and output directories
# PROJECT_ROOT = Path(__file__).parent.resolve()
# INPUT_DIR = PROJECT_ROOT / "data/input/"
# OUTPUT_DIR = PROJECT_ROOT / "data/output/"
# INPUT_IMAGE = INPUT_DIR / "inputImage.jpg"
# YOLO_PATH = PROJECT_ROOT / "yolov5/"
# MODEL_WEIGHTS = PROJECT_ROOT / "model/best.pt"

# # Create directories if they don't exist
# INPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Function to decode a base64 image string and save it to a file
# def decodeImage(imgstring, fileName):
#     imgdata = base64.b64decode(imgstring)
#     with open(fileName, "wb") as f:
#         f.write(imgdata)

# # Function to encode an image file into a base64 string
# def encodeImageIntoBase64(image_path):
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read())

# # Function to get the latest YOLOv5 output directory
# def get_latest_output_dir():
#     exp_dirs = sorted(glob.glob(f"{YOLO_PATH}/runs/detect/exp*"), key=os.path.getmtime)
#     return Path(exp_dirs[-1]) if exp_dirs else None

# # Function to run YOLOv5 detection
# def run_yolo_detection(input_image_path):
#     if not YOLO_PATH.exists():
#         st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
#         return False

#     if not MODEL_WEIGHTS.exists():
#         st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
#         return False

#     detect_command = (
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} "
#         f"--img 416 --conf 0.5 --source {input_image_path} "
#         f"--save-txt --save-conf"
#     )
#     os.system(detect_command)

#     latest_output_dir = get_latest_output_dir()
#     if latest_output_dir:
#         detected_image_path = latest_output_dir / "inputImage.jpg"
#         if detected_image_path.exists():
#             output_image_path = OUTPUT_DIR / detected_image_path.name
#             shutil.copy(detected_image_path, output_image_path)
#             return output_image_path
#     st.error("Detected image not found. Detection may have failed.")
#     return False

# # Function to reset input and output directories
# def reset_directories():
#     shutil.rmtree(INPUT_DIR, ignore_errors=True)
#     shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
#     INPUT_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Custom CSS for enhanced styling
# def local_css():
#     st.markdown("""
#     <style>
#     /* Global Styling */
#     body {
#         background-color: #f4f4f9;
#         font-family: 'Arial', sans-serif;
#     }

#     /* App Title Styling */
#     .title {
#         text-align: center;
#         color: #2c3e50;
#         font-size: 3em;
#         margin-bottom: 20px;
#         text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
#     }

#     /* Subtitle Styling */
#     .subtitle {
#         text-align: center;
#         color: #34495e;
#         font-size: 1.5em;
#         margin-bottom: 30px;
#     }

#     /* Image Upload Section Styling */
#     .upload-section {
#         background-color: #ecf0f1;
#         border-radius: 10px;
#         padding: 20px;
#         text-align: center;
#         transition: all 0.3s ease;
#     }

#     .upload-section:hover {
#         background-color: #d6dbdf;
#         transform: scale(1.02);
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }

#     /* Image Display Styling */
#     .image-container {
#         border: 2px dashed #3498db;
#         border-radius: 10px;
#         padding: 10px;
#         margin-top: 20px;
#         transition: all 0.3s ease;
#     }

#     .image-container:hover {
#         border-color: #2980b9;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }

#     /* Reset Button Styling */
#     .reset-button {
#         display: block;
#         width: 200px;
#         margin: 20px auto;
#         background-color: #e74c3c;
#         color: white;
#         border: none;
#         padding: 12px 24px;
#         font-size: 1.1em;
#         border-radius: 8px;
#         cursor: pointer;
#         transition: all 0.3s ease;
#         text-align: center;
#     }

#     .reset-button:hover {
#         background-color: #c0392b;
#         transform: scale(1.05);
#         box-shadow: 0 4px 6px rgba(0,0,0,0.2);
#     }

#     /* Instruction Text Styling */
#     .instructions {
#         background-color: #f1c40f;
#         color: #2c3e50;
#         padding: 15px;
#         border-radius: 8px;
#         margin-bottom: 20px;
#         text-align: center;
#         font-weight: bold;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Streamlit App Configuration
# def main():
#     # Set page configuration
#     st.set_page_config(page_title="Waste Detection App", page_icon="🚮", layout="wide")
    
#     # Apply custom CSS
#     local_css()

#     # App Title and Introduction
#     st.markdown("<div class='title'>🚮 Waste Detection App 🚮</div>", unsafe_allow_html=True)
#     st.markdown("<div class='subtitle'>Detect and Analyze Waste in Images Using AI 🤖</div>", unsafe_allow_html=True)

#     # User Instructions
#     st.markdown("""
#     <div class='instructions'>
#     📝 Instructions:
#     • Upload an image of waste
#     • Our AI will detect and classify waste types
#     • View predictions side-by-side
#     </div>
#     """, unsafe_allow_html=True)

#     # Create two columns for layout
#     col1, col2 = st.columns(2)

#     # Image Upload Section
#     with col1:
#         st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
#         uploaded_file = st.file_uploader("🔍 Upload Waste Image", type=["jpg", "png", "jpeg"])
#         st.markdown("</div>", unsafe_allow_html=True)

#         if uploaded_file:
#             reset_directories()
#             with open(INPUT_IMAGE, "wb") as f:
#                 f.write(uploaded_file.read())

#             st.markdown("<div class='image-container'>", unsafe_allow_html=True)
#             st.markdown("<h3 style='text-align:center;'>📸 Uploaded Image</h3>", unsafe_allow_html=True)
#             st.image(str(INPUT_IMAGE), caption="Your Selected Image", use_container_width=True)
#             st.markdown("</div>", unsafe_allow_html=True)

#     # Prediction Section
#     with col2:
#         if uploaded_file:
#             output_image_path = run_yolo_detection(INPUT_IMAGE)
#             if output_image_path:
#                 st.markdown("<div class='image-container'>", unsafe_allow_html=True)
#                 st.markdown("<h3 style='text-align:center;'>🤖 Predicted Image</h3>", unsafe_allow_html=True)
                
#                 encoded_image = encodeImageIntoBase64(output_image_path)
#                 decoded_image = base64.b64decode(encoded_image)
#                 st.image(decoded_image, caption="AI Detection Results", use_container_width=True)
#                 st.markdown("</div>", unsafe_allow_html=True)
                
#                 os.remove(INPUT_IMAGE)

#     # Reset Button with Enhanced Styling
#     if st.button("🔄 Reset Application", key="reset_button", help="Clear all uploaded images and reset the app"):
#         reset_directories()
#         st.experimental_rerun()

# if __name__ == "__main__":
#     main()


# import streamlit as st
# import base64
# import os
# import shutil
# from pathlib import Path
# import glob

# # Define input and output directories
# PROJECT_ROOT = Path(__file__).parent.resolve()
# INPUT_DIR = PROJECT_ROOT / "data/input/"
# OUTPUT_DIR = PROJECT_ROOT / "data/output/"
# INPUT_IMAGE = INPUT_DIR / "inputImage.jpg"
# YOLO_PATH = PROJECT_ROOT / "yolov5/"
# MODEL_WEIGHTS = PROJECT_ROOT / "model/best.pt"

# # Create directories if they don't exist
# INPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Function to decode a base64 image string and save it to a file
# def decodeImage(imgstring, fileName):
#     imgdata = base64.b64decode(imgstring)
#     with open(fileName, "wb") as f:
#         f.write(imgdata)

# # Function to encode an image file into a base64 string
# def encodeImageIntoBase64(image_path):
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read())

# # Function to get the latest YOLOv5 output directory
# def get_latest_output_dir():
#     exp_dirs = sorted(glob.glob(f"{YOLO_PATH}/runs/detect/exp*"), key=os.path.getmtime)
#     return Path(exp_dirs[-1]) if exp_dirs else None

# # Function to run YOLOv5 detection
# def run_yolo_detection(input_image_path):
#     if not YOLO_PATH.exists():
#         st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
#         return False

#     if not MODEL_WEIGHTS.exists():
#         st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
#         return False

#     detect_command = (
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} "
#         f"--img 416 --conf 0.5 --source {input_image_path} "
#         f"--save-txt --save-conf"
#     )
#     os.system(detect_command)

#     latest_output_dir = get_latest_output_dir()
#     if latest_output_dir:
#         detected_image_path = latest_output_dir / "inputImage.jpg"
#         if detected_image_path.exists():
#             output_image_path = OUTPUT_DIR / detected_image_path.name
#             shutil.copy(detected_image_path, output_image_path)
#             return output_image_path
#     st.error("Detected image not found. Detection may have failed.")
#     return False

# # Function to reset input and output directories
# def reset_directories():
#     shutil.rmtree(INPUT_DIR, ignore_errors=True)
#     shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
#     INPUT_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     st.success("Directories reset successfully! Ready for new predictions.")

# # Streamlit UI with enhanced design

# st.set_page_config("🚮 Waste Detection App 🚮")


# st.markdown(
#     """
#     <style>
#     .title {
#         text-align: center;
#         color: #4CAF50;
#         font-size: 2.5em;
#     }
#     .subtitle {
#         text-align: center;
#         color: #f39c12;
#         font-size: 1.5em;
#     }
#     .column {
#         padding: 10px;
#     }
#     .reset-button {
#         background-color: #e74c3c;
#         color: white;
#         border: none;
#         padding: 10px 20px;
#         font-size: 1em;
#         border-radius: 5px;
#         cursor: pointer;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )



# st.markdown("<div class='title'>🚮 Waste Detection App 🚮</div>", unsafe_allow_html=True)
# st.markdown("<div class='subtitle'>Upload an image to detect waste and view predictions side-by-side!</div>", unsafe_allow_html=True)

# # Split-screen layout
# col1, col2 = st.columns(2)

# # File upload section with image preview in the left column
# with col1:
#     uploaded_file = st.file_uploader("🔍 Upload an Image", type=["jpg", "png", "jpeg"])
#     if uploaded_file:
#         reset_directories()
#         with open(INPUT_IMAGE, "wb") as f:
#             f.write(uploaded_file.read())

#         st.image(str(INPUT_IMAGE), caption="Uploaded Image", use_container_width=True)

# # Prediction and results display in the right column
# if uploaded_file:
#     with col2:
#         output_image_path = run_yolo_detection(INPUT_IMAGE)
#         if output_image_path:
#             encoded_image = encodeImageIntoBase64(output_image_path)
#             decoded_image = base64.b64decode(encoded_image)
#             st.image(decoded_image, caption="Predicted Image", use_container_width=True)
#             os.remove(INPUT_IMAGE)

# # Reset button functionality
# if st.button("Reset", key="reset_button"):
#     reset_directories()
#     st.markdown("<div style='text-align: center; color: #e74c3c;'>🌀 Directories reset. Ready for new predictions!</div>", unsafe_allow_html=True)


# import streamlit as st
# import base64
# import os
# import shutil
# from pathlib import Path
# import glob

# # Define input and output directories
# PROJECT_ROOT = Path(__file__).parent.resolve()
# INPUT_DIR = PROJECT_ROOT / "data/input/"
# OUTPUT_DIR = PROJECT_ROOT / "data/output/"
# INPUT_IMAGE = INPUT_DIR / "inputImage.jpg"
# YOLO_PATH = PROJECT_ROOT / "yolov5/"
# MODEL_WEIGHTS = PROJECT_ROOT / "model/best.pt"

# # Create directories if they don't exist
# INPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Function to decode a base64 image string and save it to a file
# def decodeImage(imgstring, fileName):
#     imgdata = base64.b64decode(imgstring)
#     with open(fileName, "wb") as f:
#         f.write(imgdata)

# # Function to encode an image file into a base64 string
# def encodeImageIntoBase64(image_path):
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read())

# # Function to get the latest YOLOv5 output directory
# def get_latest_output_dir():
#     exp_dirs = sorted(glob.glob(f"{YOLO_PATH}/runs/detect/exp*"), key=os.path.getmtime)
#     return Path(exp_dirs[-1]) if exp_dirs else None

# # Function to run YOLOv5 detection
# def run_yolo_detection(input_image_path):
#     if not YOLO_PATH.exists():
#         st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
#         return False

#     if not MODEL_WEIGHTS.exists():
#         st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
#         return False

#     detect_command = (
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} "
#         f"--img 416 --conf 0.5 --source {input_image_path} "
#         f"--save-txt --save-conf"
#     )
#     st.info(f"Running detection command: {detect_command}")
#     os.system(detect_command)

#     latest_output_dir = get_latest_output_dir()
#     if latest_output_dir:
#         detected_image_path = latest_output_dir / "inputImage.jpg"
#         if detected_image_path.exists():
#             output_image_path = OUTPUT_DIR / detected_image_path.name
#             shutil.copy(detected_image_path, output_image_path)
#             return output_image_path
#     st.error("Detected image not found. Detection may have failed.")
#     return False

# # Function to reset input and output directories
# def reset_directories():
#     shutil.rmtree(INPUT_DIR, ignore_errors=True)
#     shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
#     INPUT_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     st.success("Directories reset successfully! Ready for new predictions.")

# # Streamlit UI
# st.title("Waste Detection App")

# # Automatically reset data if a new image is uploaded
# uploaded_file = st.file_uploader("Upload an Image for Prediction", type=["jpg", "png", "jpeg"])
# if uploaded_file:
#     reset_directories()  # Automatically reset directories before handling the new image
#     with open(INPUT_IMAGE, "wb") as f:
#         f.write(uploaded_file.read())

#     st.info("Running detection...")
#     output_image_path = run_yolo_detection(INPUT_IMAGE)
#     if output_image_path:
#         encoded_image = encodeImageIntoBase64(output_image_path)
#         decoded_image = base64.b64decode(encoded_image)
#         st.image(decoded_image, caption="Detected Image", use_container_width=True)
#         os.remove(INPUT_IMAGE)

# # Reset button functionality
# if st.button("Reset"):
#     reset_directories()
#     st.success("Uploaded image removed. Ready for new predictions!")



# import streamlit as st
# import base64
# import os
# import shutil
# from pathlib import Path
# import glob

# # Define input and output directories
# PROJECT_ROOT = Path(__file__).parent.resolve()
# INPUT_DIR = PROJECT_ROOT / "data/input/"
# OUTPUT_DIR = PROJECT_ROOT / "data/output/"
# INPUT_IMAGE = INPUT_DIR / "inputImage.jpg"
# YOLO_PATH = PROJECT_ROOT / "yolov5/"
# MODEL_WEIGHTS = PROJECT_ROOT / "model/best.pt"

# # Create directories if they don't exist
# INPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Function to decode a base64 image string and save it to a file
# def decodeImage(imgstring, fileName):
#     imgdata = base64.b64decode(imgstring)
#     with open(fileName, "wb") as f:
#         f.write(imgdata)

# # Function to encode an image file into a base64 string
# def encodeImageIntoBase64(image_path):
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read())

# # Function to get the latest YOLOv5 output directory
# def get_latest_output_dir():
#     exp_dirs = sorted(glob.glob(f"{YOLO_PATH}/runs/detect/exp*"), key=os.path.getmtime)
#     return Path(exp_dirs[-1]) if exp_dirs else None

# # Function to run YOLOv5 detection
# def run_yolo_detection(input_image_path):
#     if not YOLO_PATH.exists():
#         st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
#         return False

#     if not MODEL_WEIGHTS.exists():
#         st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
#         return False

#     detect_command = (
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} "
#         f"--img 416 --conf 0.5 --source {input_image_path} "
#         f"--save-txt --save-conf"
#     )
#     #st.info(f"Running detection command: {detect_command}")
#     os.system(detect_command)

#     latest_output_dir = get_latest_output_dir()
#     if latest_output_dir:
#         detected_image_path = latest_output_dir / "inputImage.jpg"
#         if detected_image_path.exists():
#             output_image_path = OUTPUT_DIR / detected_image_path.name
#             shutil.copy(detected_image_path, output_image_path)
#             return output_image_path
#     st.error("Detected image not found. Detection may have failed.")
#     return False

# # Function to reset input and output directories
# def reset_directories():
#     shutil.rmtree(INPUT_DIR, ignore_errors=True)
#     shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
#     INPUT_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     st.success("Directories reset successfully! Ready for new predictions.")

# # Streamlit UI
# st.title("Waste Detection App")

# uploaded_file = st.file_uploader("Upload an Image for Prediction", type=["jpg", "png", "jpeg"])
# if uploaded_file:
#     with open(INPUT_IMAGE, "wb") as f:
#         f.write(uploaded_file.read())

#     st.info("Running detection...")
#     output_image_path = run_yolo_detection(INPUT_IMAGE)
#     if output_image_path:
#         encoded_image = encodeImageIntoBase64(output_image_path)
#         decoded_image = base64.b64decode(encoded_image)
#         st.image(decoded_image, caption="Detected Image", use_container_width=True)
#         os.remove(INPUT_IMAGE)

# if st.button("Reset"):
#     reset_directories()


# import streamlit as st
# import base64
# import os
# import shutil
# from pathlib import Path

# # Define input and output directories
# PROJECT_ROOT = Path(__file__).parent.resolve()
# INPUT_DIR = PROJECT_ROOT / "data/input/"
# OUTPUT_DIR = PROJECT_ROOT / "data/output/"
# INPUT_IMAGE = INPUT_DIR / "inputImage.jpg"
# OUTPUT_IMAGE = OUTPUT_DIR / "outputImage.jpg"
# YOLO_PATH = PROJECT_ROOT / "yolov5/"
# MODEL_WEIGHTS = PROJECT_ROOT / "model/best.pt"

# # Create directories if they don't exist
# INPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Function to decode a base64 image string and save it to a file
# def decode_image(imgstring, file_name):
#     imgdata = base64.b64decode(imgstring)
#     with open(file_name, "wb") as f:
#         f.write(imgdata)

# # Function to encode an image file into a base64 string
# def encode_image_into_base64(image_path):
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read())

# # Function to run YOLOv5 detection
# def run_yolo_detection(input_image_path, output_image_path):
#     if not YOLO_PATH.exists():
#         st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
#         return False

#     if not MODEL_WEIGHTS.exists():
#         st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
#         return False

#     # Normalize paths for YOLOv5
#     input_image_path = input_image_path.resolve()
#     output_image_path = output_image_path.resolve()

#     # Construct and execute the YOLO detection command
#     detect_command = (
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS.resolve()} "
#         f"--img 416 --conf 0.5 --source {input_image_path} "
#         f"--save-txt --save-conf"
#     )

#     st.info(f"Running detection command:\n{detect_command}")
#     result = os.system(detect_command)

#     if result != 0:
#         st.error("YOLOv5 detection command failed. Please check your setup.")
#         return False

#     # Check if output image was generated
#     detected_image_path = YOLO_PATH / "runs/detect/exp/inputImage.jpg"
#     if detected_image_path.exists():
#         shutil.move(str(detected_image_path), str(output_image_path))
#         return True
#     else:
#         st.error("Detected image not found. Detection may have failed.")
#         return False

# # Function to reset input and output directories
# def reset_directories():
#     shutil.rmtree(INPUT_DIR)
#     shutil.rmtree(OUTPUT_DIR)
#     INPUT_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     st.success("Directories reset successfully! Ready for new predictions.")

# # Streamlit UI
# st.title("Waste Detection App")

# uploaded_file = st.file_uploader("Upload an Image for Prediction", type=["jpg", "png"])
# if uploaded_file:
#     with open(INPUT_IMAGE, "wb") as f:
#         f.write(uploaded_file.read())
    
#     st.info("Running detection...")
#     if run_yolo_detection(INPUT_IMAGE, OUTPUT_IMAGE):
#         encoded_image = encode_image_into_base64(OUTPUT_IMAGE)
#         decoded_image = base64.b64decode(encoded_image)
#         st.image(decoded_image, caption="Detected Image", use_column_width=True)
#         os.remove(INPUT_IMAGE)
#         os.remove(OUTPUT_IMAGE)

# if st.button("Reset"):
#     reset_directories()


# import streamlit as st
# import base64
# import os
# import shutil
# from pathlib import Path

# # Define input and output directories
# PROJECT_ROOT = Path(__file__).parent.resolve()
# INPUT_DIR = PROJECT_ROOT / "data/input/"
# OUTPUT_DIR = PROJECT_ROOT / "data/output/"
# INPUT_IMAGE = INPUT_DIR / "inputImage.jpg"
# OUTPUT_IMAGE = OUTPUT_DIR / "outputImage.jpg"
# YOLO_PATH = PROJECT_ROOT / "yolov5/"
# MODEL_WEIGHTS = PROJECT_ROOT / "model/best.pt"

# # Create directories if they don't exist
# INPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Function to decode a base64 image string and save it to a file
# def decodeImage(imgstring, fileName):
#     imgdata = base64.b64decode(imgstring)
#     with open(fileName, 'wb') as f:
#         f.write(imgdata)

# # Function to encode an image file into a base64 string
# def encodeImageIntoBase64(croppedImagePath):
#     with open(croppedImagePath, "rb") as f:
#         return base64.b64encode(f.read())

# # Function to run YOLOv5 detection
# def run_yolo_detection(input_image_path, output_image_path):
#     if not YOLO_PATH.exists():
#         st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
#         return False

#     if not MODEL_WEIGHTS.exists():
#         st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
#         return False

#     # Construct and execute the YOLO detection command
#     detect_command = (
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} "
#         f"--img 416 --conf 0.5 --source ../{input_image_path} "
#         f"--save-txt --save-conf"
#     )
#     os.system(detect_command)

#     detected_image_path = YOLO_PATH / "runs/detect/exp/inputImage.jpg"
#     if detected_image_path.exists():
#         shutil.move(str(detected_image_path), output_image_path)
#         return True
#     else:
#         st.error("Output image not found. Detection may have failed.")
#         return False

# # Function to reset input and output directories
# def reset_directories():
#     shutil.rmtree(INPUT_DIR)
#     shutil.rmtree(OUTPUT_DIR)
#     INPUT_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     st.success("Directories reset successfully! Ready for new predictions.")

# # Streamlit UI
# st.title("Waste Detection App")

# uploaded_file = st.file_uploader("Upload an Image for Prediction", type=["jpg", "png"])
# if uploaded_file:
#     with open(INPUT_IMAGE, "wb") as f:
#         f.write(uploaded_file.read())

#     st.info("Running detection...")
#     if run_yolo_detection(INPUT_IMAGE, OUTPUT_IMAGE):
#         encoded_image = encodeImageIntoBase64(OUTPUT_IMAGE)
#         decoded_image = base64.b64decode(encoded_image)
#         st.image(decoded_image, caption="Detected Image", use_column_width=True)
#         os.remove(INPUT_IMAGE)
#         os.remove(OUTPUT_IMAGE)

# if st.button("Reset"):
#     reset_directories()


# import streamlit as st
# import base64
# import os
# import shutil
# from pathlib import Path

# # Define input and output directories
# INPUT_DIR = Path("data/input/")
# OUTPUT_DIR = Path("data/output/")
# INPUT_IMAGE = INPUT_DIR / "inputImage.jpg"
# OUTPUT_IMAGE = OUTPUT_DIR / "outputImage.jpg"

# # Create directories if not exist
# INPUT_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# # Function to decode a base64 image string and save it to a file
# def decodeImage(imgstring, fileName):
#     imgdata = base64.b64decode(imgstring)
#     with open(fileName, 'wb') as f:
#         f.write(imgdata)

# # Function to encode an image file into a base64 string
# def encodeImageIntoBase64(croppedImagePath):
#     with open(croppedImagePath, "rb") as f:
#         return base64.b64encode(f.read())

# # Function to run YOLOv5 detection
# def run_yolo_detection(input_image_path, output_image_path):
#     YOLO_PATH = Path("yolov5/")
#     MODEL_WEIGHTS = Path("model/best.pt")
    
#     if not YOLO_PATH.exists():
#         st.error(f"YOLOv5 directory not found at {YOLO_PATH.resolve()}.")
#         return False
#     if not MODEL_WEIGHTS.exists():
#         st.error(f"Model weights not found at {MODEL_WEIGHTS.resolve()}.")
#         return False

#     os.system(
#         f"cd {YOLO_PATH} && python detect.py --weights {MODEL_WEIGHTS} --img 416 --conf 0.5 --source ../{input_image_path} --save-txt --save-conf"
#     )
    
#     detected_image_path = YOLO_PATH / "runs/detect/exp/inputImage.jpg"
#     if detected_image_path.exists():
#         shutil.move(str(detected_image_path), output_image_path)
#         return True
#     else:
#         st.error("Output image not found. Detection may have failed.")
#         return False

# # Function to reset input and output directories
# def reset_directories():
#     shutil.rmtree(INPUT_DIR)
#     shutil.rmtree(OUTPUT_DIR)
#     INPUT_DIR.mkdir(parents=True, exist_ok=True)
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     st.success("Directories reset successfully! Ready for new predictions.")

# # Streamlit UI
# st.title("Waste Detection App")

# uploaded_file = st.file_uploader("Upload an Image for Prediction", type=["jpg", "png"])
# if uploaded_file:
#     with open(INPUT_IMAGE, "wb") as f:
#         f.write(uploaded_file.read())
    
#     st.info("Running detection...")
#     if run_yolo_detection(INPUT_IMAGE, OUTPUT_IMAGE):
#         encoded_image = encodeImageIntoBase64(OUTPUT_IMAGE)
#         decoded_image = base64.b64decode(encoded_image)
#         st.image(decoded_image, caption="Detected Image", use_column_width=True)
#         os.remove(INPUT_IMAGE)
#         os.remove(OUTPUT_IMAGE)

# if st.button("Reset"):
#     reset_directories()



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