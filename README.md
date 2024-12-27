
----

# **Waste Detection Using YOLOv5 🚮🔍**

Welcome to the Waste Detection Using YOLOv5 project! This application identifies waste in images using the YOLOv5 object detection model and is deployed as a web app with Streamlit. 🌐

## **Features ✨**

- **Real-time Waste Detection**: Upload images to detect waste materials instantly. 🖼️⚡
- **User-Friendly Interface**: Interact with the model through a simple web application. 🖥️👍
- **Streamlit Deployment**: Seamless deployment using Streamlit Cloud. ☁️🚀

## **Getting Started 🛠️**

Follow these steps to set up and run the project locally:

### **Prerequisites 📋**

- **Python 3.8+**: Ensure Python is installed on your system. 🐍
- **Virtual Environment**: It's recommended to use a virtual environment to manage dependencies. 🌐

### **Installation 📥**

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/muhammadadilnaeem/Waste-Detection-Using-Yolov5.git
   cd Waste-Detection-Using-Yolov5
   ```

2. **Create and Activate Virtual Environment**:

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Unix or MacOS
   source venv/bin/activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Application ▶️

1. **Start the Streamlit App**:

   ```bash
   streamlit run streamlit.py
   ```

2. **Access the App**:

   Open your web browser and navigate to `http://localhost:8501` to interact with the application. 🌐

## **Model Training 🏋️‍♂️**

The YOLOv5 model is trained to detect various types of waste. To train the model:

1. **Prepare Dataset**: Collect and annotate images of waste materials.
2. **Configure Training**: Set up training parameters in the YOLOv5 configuration files.
3. **Train the Model**: Execute the training script to train the model on your dataset.

*Note*: Detailed training instructions are beyond the scope of this README. Please refer to the YOLOv5 documentation for comprehensive guidance. 📚

## **Deployment 🚀**

The application is deployed using Streamlit Cloud. To deploy:

1. **Sign Up**: Create an account on [Streamlit Cloud](https://streamlit.io/cloud).
2. **Deploy App**: Connect your GitHub repository and deploy the `streamlit.py` file.
3. **Access Live App**: Once deployed, access your application via the provided URL.

*Note*: This project does not utilize Docker or AWS services for deployment. 🛑🐳


## **License 📄**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 📜

## **Acknowledgements 🙏**

- **YOLOv5**: For the robust object detection model. 🤖
- **Streamlit**: For the intuitive web application framework. 🌟

Enjoy detecting waste with our application! 🗑️✨ 

----

