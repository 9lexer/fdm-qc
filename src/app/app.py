import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import pandas as pd

# --- CONFIGURARE ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# IMPORTANT: Încărcăm modelul OPTIMIZAT (Etapa 6)
MODEL_PATH = BASE_DIR / "models" / "optimized_model.h5"
TRAIN_DIR = BASE_DIR / "data" / "train"

st.set_page_config(page_title="SIA FDM Quality Control", page_icon="✅", layout="wide")

st.title("🚀 SIA Control Calitate FDM (Etapa 6 - Final)")
st.markdown("### Modul de Inferență Optimizat")
st.markdown("---")

# --- 1. Detectare Automată a Claselor ---
def get_class_names():
    if not TRAIN_DIR.exists():
        st.error(f"❌ Nu găsesc folderul de antrenare la: {TRAIN_DIR}")
        return []
    return sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])

CLASS_NAMES = get_class_names()

# --- 2. Încărcare Model ---
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Eroare încărcare model: {e}")
        return None

model = load_model()

if model is None:
    st.error(f"❌ Nu găsesc modelul optimizat la: {MODEL_PATH}")
    st.warning("Verifică dacă ai copiat 'exp2_lr.h5' -> 'models/optimized_model.h5'")
else:
    # Sidebar cu informații
    with st.sidebar:
        st.success("✅ Model Optimizat Încărcat")
        st.info("**Statistici Model:**")
        st.markdown("- Acuratețe Test: **100%**")
        st.markdown("- F1-Score: **1.00**")
        st.markdown("- Loss: **0.14**")
        st.caption("Versiune: v1.0-final")

    # --- 3. Interfața de Upload ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Încarcă imaginea piesei...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Preprocesare
        img_array = np.array(image)
        img_resized = cv2.resize(img_array, (128, 128))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        # Predicție
        predictions = model.predict(img_batch)
        scores = predictions[0]
        
        class_idx = np.argmax(scores)
        confidence = np.max(scores) * 100
        
        if CLASS_NAMES:
            predicted_label = CLASS_NAMES[class_idx]
        else:
            predicted_label = f"Clasa {class_idx}"

        # --- 4. Afișare Rezultate ---
        with col1:
            st.image(image, caption='Imagine Analizată', use_container_width=True)

        with col2:
            st.subheader("Rezultat Diagnostic:")
            
            # Logică de afișare culori
            if predicted_label == "OK":
                st.success(f"✅ CONFORM: {predicted_label}")
            else:
                st.error(f"⚠️ DEFECT IDENTIFICAT: {predicted_label}")
            
            st.metric("Nivel de Încredere (Confidence)", f"{confidence:.2f}%")

            # Grafic Probabilități (Detaliat)
            st.markdown("#### Analiză Probabilități:")
            if CLASS_NAMES:
                chart_data = pd.DataFrame({
                    'Clasă': CLASS_NAMES,
                    'Probabilitate': scores * 100
                })
                st.bar_chart(chart_data.set_index('Clasă'))