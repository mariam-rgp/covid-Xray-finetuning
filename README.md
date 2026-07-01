# 🫁 COVID-19 Chest X-Ray Classification using Deep Learning (VGG16 Fine-Tuning)

## 📖 Project Overview
This project focuses on classifying chest X-ray images into different categories (e.g., COVID-19, Normal, and other classes depending on dataset) using a **Transfer Learning approach with VGG16**.

The model is trained to assist in early detection support by analyzing medical images and predicting the likely class.

---

## 🧠 Model Architecture
- Base Model: **VGG16 (ImageNet pretrained)**
- Fine-tuning applied on top layers
- Added custom classification head:
  - GlobalAveragePooling2D
  - Dense layers with ReLU activation
  - Dropout for regularization
  - Softmax output layer

---

## 📂 Dataset
- COVID-19 Radiography Dataset
- Classes:
  - COVID-19
  - Normal
  - (Other classes depending on dataset)

Data split:
- Training set
- Validation set
- Test set

---

## ⚙️ Preprocessing
- Image resizing to (224, 224)
- Normalization (1./255)
- Data augmentation applied (if used)

---

## 🚀 Training Details
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Metrics: Accuracy
- Callbacks:
  - ModelCheckpoint (saves best model based on val_accuracy)
  - EarlyStopping (prevents overfitting)

---

## 📊 Evaluation Metrics
- Accuracy
- Validation Accuracy
- Loss curves
- Confusion Matrix
- Classification Report

---

## 📈 Results
- Best Validation Accuracy achieved: ~**0.98 (98%)**
- Model shows strong performance in distinguishing between classes.

---

## 📉 Visualizations
- Training vs Validation Accuracy/Loss plots
- Confusion Matrix heatmap
- Sample predictions with confidence scores

---



## 💾 Model Saving
Best model is saved using:

```python
ModelCheckpoint("models/vgg16_model.keras")

📌 Key Features
Transfer Learning with VGG16
Fine-tuning for medical imaging
Confusion Matrix analysis
Real image prediction
Highly accurate classification model