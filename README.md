
# 🖼️ AI Image Classifier Web App

This is a lightweight, real-time image classification web application built using **Streamlit** and a **pre-trained MobileNetV2** model from TensorFlow Keras. The app allows users to upload an image, receive top-3 class predictions with confidence scores, and optionally generate a downloadable report.

---

## 🚀 Features

- 📂 Upload `.jpg`, `.jpeg`, or `.png` image files
- ⚡ Real-time image classification using **MobileNetV2**
- 📊 Visual display of predictions with progress bars
- 📝 Downloadable prediction report as `.txt` file
- 🔊 Optional voice narration using Google Text-to-Speech (gTTS)
- 🎛️ Clean and interactive web UI powered by **Streamlit**

---

## 🧠 Model

- **MobileNetV2**: A lightweight deep CNN trained on the **ImageNet** dataset (1,000 classes).
- Utilizes softmax to output top-3 predictions with confidence scores.

---

## 🛠️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## ▶️ Running the App

After installing the dependencies, run the app using Streamlit:

💡 Make sure the file name is main.py and it’s in the same directory.

```bash
streamlit run main.py


