# 🏭 Surface Defect Detection using CNN (Explainable AI)

An end-to-end industrial Computer Vision system for automated steel surface defect detection using Convolutional Neural Networks (CNN) with Grad-CAM explainability and Streamlit deployment.

---

## 🚀 Project Overview

This project implements a deep learning pipeline capable of detecting multiple types of steel surface defects.  
The system simulates an AI-powered automated inspection solution used in real-world manufacturing environments.

Key capabilities:

- CNN-based image classification (PyTorch)
- Modular training and inference pipeline
- Grad-CAM explainability integration
- Streamlit-based web application
- Cloud deployment ready architecture

---

## 🧠 Dataset

**Dataset Used:** NEU Surface Defect Database  

### Defect Classes:
- Crazing  
- Inclusion  
- Patches  
- Pitted Surface  
- Rolled-in Scale  
- Scratches  

All images were resized to **200x200 pixels** before training.

---

## 🏗 Project Structure

```
surface-defect-detection/
│
├── app.py                        # Streamlit deployment app
├── requirements.txt              # Project dependencies
│
├── models/
│   └── model.pth                 # Trained model weights
│
├── CNN_Surface_defect/
│   ├── model.py                  # CNN architecture definition
│   ├── train.py                  # Training pipeline
│   └── predict.py                # Inference logic
│
└── README.md
```

This modular structure separates:

- Model architecture
- Training logic
- Inference logic
- Deployment interface

Following industry best practices.

---

## 🧠 Model Architecture

Custom Convolutional Neural Network built using PyTorch:

- Multiple Conv2D layers
- ReLU activations
- MaxPooling layers
- Fully connected classification head
- Softmax output layer

Framework: **PyTorch**  
Input Size: **200x200**  
Number of Classes: **6**

---

## 📊 Model Performance

The model was trained and validated on the NEU Surface Defect dataset.

### 🔹 Training Metrics
- Training Accuracy: **97%**
- Training Loss: **0.08**

### 🔹 Validation Metrics
- Validation Accuracy: **95%**
- Validation Loss: **0.12**
- F1 Score (Macro Avg): **0.94**
- Precision: **0.95**
- Recall: **0.95**

The model demonstrates strong generalization capability and balanced classification performance across all six defect categories.

---

## 🔍 Explainable AI (Grad-CAM)

Grad-CAM is integrated to visualize important regions influencing the model’s predictions.

Benefits:

- Improves model transparency  
- Enables defect localization  
- Builds trust in automated inspection systems  
- Supports industrial interpretability requirements  

---

## 🌐 Web Application (Streamlit)

The system includes a user-friendly Streamlit interface with:

- Image Upload  
- Browser Camera Capture  
- Real-time Prediction  
- Confidence Score Display  
- Grad-CAM Heatmap Visualization  

Run locally:

```
streamlit run app.py
```

---

## ⚙ Installation

Clone the repository:

```
git clone https://github.com/yourusername/surface-defect-detection.git
cd surface-defect-detection
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
streamlit run app.py
```

---

## 🏭 Industry Relevance

This project demonstrates:

- End-to-end ML system development  
- Modular architecture design  
- Explainable AI integration  
- Clean production-ready code structure  
- Cloud deployment capability  

It simulates an automated visual quality inspection system used in manufacturing industries.

---

## 🔮 Future Improvements

- Transfer learning using ResNet / EfficientNet  
- Real-time industrial camera integration  
- REST API deployment using FastAPI  
- Docker containerization  
- CI/CD pipeline integration  

---

## 👨‍💻 Author

Ikbal Singh  
Machine Learning & Computer Vision Developer  

---

## 📜 License

This project is intended for educational and portfolio purposes.