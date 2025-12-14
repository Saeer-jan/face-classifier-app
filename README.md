# 👤 Face View Classifier (Frontal vs. Profile)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/saeer-jan/face-classifier-app)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99.3%25-brightgreen.svg)

## 🚀 Live Demo
**Click here to use the app:** 👉 [Face View Classifier App](https://share.streamlit.io/saeer-jan/face-classifier-app)

*(Note: If the app is sleeping, click "Wake Up" and wait a minute!)*

---

## 📌 Project Overview
This project is an AI-powered Computer Vision application that classifies human faces into two categories:
* **Frontal View:** The person is looking directly at the camera.
* **Side Profile:** The person is looking to the left or right.

It solves the problem of filtering image datasets for facial recognition systems, ensuring only high-quality frontal or profile images are selected.

## 🧠 How It Works
The model utilizes **Transfer Learning** with the powerful **ResNet50** architecture.
1.  **Dataset:** Trained on 7,000+ images (balanced using Class Weights).
2.  **Preprocessing:** Images are resized to 224x224 and normalized.
3.  **Model:** A frozen ResNet50 base with a custom trainable top layer.
4.  **Performance:** Achieved **99.3% Validation Accuracy** after 10 epochs.

## 🛠️ Tech Stack
* **Language:** Python 3.10
* **Deep Learning:** TensorFlow / Keras
* **Computer Vision:** OpenCV, PIL
* **Web Framework:** Streamlit
* **Version Control:** Git & Git LFS (Large File Storage)

---

## 📸 Screenshots
*<img width="1913" height="966" alt="image" src="https://github.com/user-attachments/assets/036f4cad-0b5e-4218-9ec6-078a08c152a1" />
*

---

## 💻 How to Run Locally

If you want to run this project on your own machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Saeer-jan/face-classifier-app.git](https://github.com/Saeer-jan/face-classifier-app.git)
    cd face-classifier-app
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run app.py
    ```

---

## 👨‍💻 Author
**Saeer Jan**
* **Role:** AI & Computer Vision Engineer
* **LinkedIn:** [Your LinkedIn Profile](https://www.linkedin.com/in/umainazeer/)
* **GitHub:** [Saeer-jan](https://github.com/Saeer-jan)

---
*Built with ❤️ in Python.*
