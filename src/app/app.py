import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# Configurare Căi - Detectează automat unde e proiectul
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "untrained_model_v0.h5"

# Titlu și Configurare
st.set_page_config(page_title="FDM Quality Control", page_icon="⚙️")
st.title("⚙️ Sistem AI - Control Calitate FDM")
st.subheader("Etapa 4: Demonstrație Arhitectură")

# Încărcare Model
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Eroare la încărcare: {e}")
        return None

model = load_model()

# Status Sistem
if model is None:
    st.error(f"❌ EROARE: Nu găsesc fișierul modelului la: {MODEL_PATH}")
    st.info("Rulează întâi scriptul 'src/neural_network/model.py'")
else:
    st.success("✅ Sistem ONLINE. Model arhitectural încărcat cu succes.")

    # Zona de Testare
    uploaded_file = st.file_uploader("Încarcă imagine strat (JPG/PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagine Capturată', width=300)
        
        if st.button("Simulare Analiză"):
            # Preprocesare simplă (Resize la 128x128)
            img_array = np.array(image)
            img_resized = cv2.resize(img_array, (128, 128))
            img_normalized = img_resized / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Predicție
            prediction = model.predict(img_batch)
            class_idx = np.argmax(prediction)
            confidence = np.max(prediction) * 100
            
            st.divider()
            st.info(f"🔍 Rezultat (Index Clasă): {class_idx}")
            st.metric("Încredere (Confidence)", f"{confidence:.2f}%")
            st.caption("Notă: Rezultatul este aleatoriu deoarece modelul nu este antrenat încă.")