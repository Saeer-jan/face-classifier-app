import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Saeer Jan | AI Portfolio",
    page_icon="🤖",
    layout="centered"
)

# ==========================================
# 2. SIDEBAR (Your Profile)
# ==========================================
with st.sidebar:
    st.title("👨‍💻 Developer Profile")

    # --- 1. PROFILE PICTURE ---
    # Tip: Put a file named 'profile_pic.jpg' in the same folder!
    if os.path.exists("profile_pic.jpg"):
        st.image("profile_pic.jpg", width=150)
    else:
        st.write("*(Add 'profile_pic.jpg' to see your photo here)*")

    # --- 2. YOUR NAME ---
    st.header("Saeer Jan")
    st.write("🚀 AI & Computer Vision Engineer")

    st.markdown("---")

    # --- 3. CONTACT INFO ---
    st.write("📧 **Email:**")
    st.code("saeerjansaeer@gmail.com")

    st.write("🔗 **LinkedIn:**")
    st.markdown("[Visit My Profile](https://www.linkedin.com/in/umainazeer/)")

    st.write("📂 **GitHub:**")
    st.markdown("[View My Code](https://github.com/Saeer-jan)")

    st.markdown("---")

    # --- 4. PROJECT DETAILS (Enhanced) ---
    st.info("""
        **Project Tech Stack:**
        \n🛠 **Framework:** TensorFlow & Keras
        \n🧠 **Model:** ResNet50 (Transfer Learning)
        \n📊 **Accuracy:** 99.3%
        \n📂 **Dataset:** 7,000 Face Images
    """)

# ==========================================
# 3. MAIN APP INTERFACE
# ==========================================

# Custom styling
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    h1 { color: #1E88E5; }
    </style>
    """, unsafe_allow_html=True)

st.title("👤 Face View Classifier")
st.markdown("### AI-Powered Frontal vs. Profile Detection")
st.write("Upload a face image, and my AI model will analyze whether it is a **Frontal View** or a **Side Profile**.")


# ==========================================
# 4. LOAD MODEL (Cached)
# ==========================================
@st.cache_resource
def load_model():
    # Ensure this matches your model filename exactly
    if not os.path.exists('best_face_model.keras'):
        st.error("❌ Model file 'best_face_model.keras' not found. Please upload it.")
        return None
    model = tf.keras.models.load_model('best_face_model.keras')
    return model


with st.spinner('Loading AI Brain...'):
    model = load_model()

# ==========================================
# 5. INPUT SECTION (Upload OR Sample)
# ==========================================
st.markdown("---")
col_input1, col_input2 = st.columns([1, 2])

with col_input1:
    use_sample = st.checkbox("Use Sample Image")

uploaded_file = None

if use_sample:
    if os.path.exists("sample_face.jpg"):
        uploaded_file = "sample_face.jpg"
    else:
        st.error("⚠️ 'sample_face.jpg' not found. Please add a test image to your folder.")
else:
    uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "png", "jpeg"])

# ==========================================
# 6. PREDICTION LOGIC
# ==========================================
if uploaded_file is not None and model is not None:
    # Layout: Image on Left, Result on Right
    col1, col2 = st.columns(2)

    # Open the image
    image = Image.open(uploaded_file)

    with col1:
        st.image(image, caption='Analyzed Image', use_column_width=True)

    # Preprocessing
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)

    # Fix PNG images with 4 channels (RGBA) -> RGB
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    # Predict
    prediction = model.predict(img_array)
    score = prediction[0][0]

    with col2:
        st.write("## 🔍 Analysis Result")
        st.markdown("<br>", unsafe_allow_html=True)  # Spacer

        # 0.5 Threshold Logic
        if score > 0.5:
            confidence = score * 100
            st.error(f"**SIDE PROFILE**")
            st.balloons()  # 🎉 Fun Effect
            st.progress(int(confidence))
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.caption("The model is very sure this is a side view.")
        else:
            confidence = (1 - score) * 100
            st.success(f"**FRONTAL FACE**")
            st.balloons()  # 🎉 Fun Effect
            st.progress(int(confidence))
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.caption("The model is very sure this is a front view.")