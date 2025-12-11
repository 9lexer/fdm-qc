import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import pandas as pd
import os

# --- CONFIGURARE ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "trained_model.h5"
TRAIN_DIR = BASE_DIR / "data" / "train"

st.set_page_config(page_title="SIA FDM Quality Control", page_icon="🔍", layout="wide")

st.title("🔍 Detectare Defecte FDM (Etapa 5 - Final)")
st.markdown("### Modul de Inferență cu Debugging")

# --- 1. Detectare Automată a Claselor ---
def get_class_names():
    """Citește folderele din train pentru a stabili ordinea corectă a claselor."""
    if not TRAIN_DIR.exists():
        st.error(f"❌ Nu găsesc folderul de antrenare la: {TRAIN_DIR}")
        return []
    
    # Keras sortează alfabetic folderele. Facem la fel.
    classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    return classes

CLASS_NAMES = get_class_names()

# Afișăm clasele detectate pentru verificare
with st.expander("ℹ️ Vezi ordinea claselor detectată"):
    if CLASS_NAMES:
        st.write(f"Sistemul a detectat {len(CLASS_NAMES)} clase în această ordine:")
        st.code(CLASS_NAMES)
    else:
        st.warning("Nu am putut detecta clasele. Verifică folderul data/train.")

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
    st.error(f"❌ Nu găsesc modelul antrenat la: {MODEL_PATH}")
else:
    # --- 3. Interfața de Upload ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Încarcă o imagine...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Preprocesare
        img_array = np.array(image)
        img_resized = cv2.resize(img_array, (128, 128)) # Resize exact ca la antrenare
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        # Predicție
        predictions = model.predict(img_batch)
        scores = predictions[0] # Vectorul de probabilități
        
        class_idx = np.argmax(scores)
        confidence = np.max(scores) * 100
        
        # Siguranță în caz că nu avem clase detectate
        if CLASS_NAMES:
            predicted_label = CLASS_NAMES[class_idx]
        else:
            predicted_label = f"Clasa {class_idx} (Nume necunoscut)"

        # --- 4. Afișare Rezultate ---
        with col1:
            # AICI AM FĂCUT MODIFICAREA (use_container_width=True)
            st.image(image, caption='Imagine Originală', use_container_width=True)
            
            st.caption("Ce vede modelul (128x128px):")
            st.image(img_resized, width=128)

        with col2:
            st.subheader("Rezultat Predicție:")
            
            if predicted_label == "OK":
                st.success(f"✅ {predicted_label}")
            else:
                st.error(f"⚠️ {predicted_label}")
            
            st.metric("Încredere", f"{confidence:.2f}%")

            # --- 5. Grafic Detaliat (Debug) ---
            st.markdown("#### Detalii Probabilități:")
            
            if CLASS_NAMES:
                # Creăm un DataFrame pentru grafic
                chart_data = pd.DataFrame({
                    'Clasă': CLASS_NAMES,
                    'Probabilitate': scores * 100
                })
                
                st.bar_chart(chart_data.set_index('Clasă'))
                
                # Tabel cu valori exacte
                st.dataframe(chart_data.style.format({'Probabilitate': '{:.2f}%'}))