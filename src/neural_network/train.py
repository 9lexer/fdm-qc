import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
from pathlib import Path
import os

# --- CONFIGURARE ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_IN_PATH = BASE_DIR / "models" / "untrained_model_v0.h5"
MODEL_OUT_PATH = BASE_DIR / "models" / "trained_model.h5"
RESULTS_DIR = BASE_DIR / "results"
TRAIN_DIR = BASE_DIR / "data" / "train"
VAL_DIR = BASE_DIR / "data" / "validation"

# Parametri antrenare
EPOCHS = 30           # Încercăm 30 de epoci
BATCH_SIZE = 16       # Punem 16 pentru că ai 250 imagini (set mic)
IMAGE_SIZE = (128, 128)

def main():
    # 0. Pregătire foldere
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Încărcare Date
    print("🔄 Încărcăm datele de antrenare...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels='inferred',
        label_mode='int',       # Pentru sparse_categorical_crossentropy
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels='inferred',
        label_mode='int',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Normalizare (0-255 -> 0-1)
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # 2. Încărcare Model Schelet
    if not MODEL_IN_PATH.exists():
        print(f"❌ EROARE: Nu găsesc {MODEL_IN_PATH}. Rulează src/neural_network/model.py întâi!")
        return
    
    # Încărcăm modelul (ATENȚIE la calea din Etapa 4)
    model = tf.keras.models.load_model(MODEL_IN_PATH)
    print("✅ Model schelet încărcat.")
    
    # ⭐️ LINIA DE RECOMPILARE SALVATOARE ⭐️
    # Recompilăm modelul pentru a ne asigura că este în modul Eager/compatibil cu fit()
    model.compile(
        optimizer='adam', # Folosim optimizatorul din Etapa 4
        loss='sparse_categorical_crossentropy', # Folosim funcția de pierdere din Etapa 4
        metrics=['accuracy']
    )

    # 3. Configurare Callbacks (Salvare automată și Oprire timpurie)
    callbacks = [
        # Salvăm cel mai bun model (nu neapărat ultimul)
        ModelCheckpoint(filepath=MODEL_OUT_PATH, save_best_only=True, monitor='val_loss', verbose=1),
        # Oprim dacă nu mai învață timp de 8 epoci
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        # Scădem rata de învățare dacă se blochează
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    ]

    # 4. START ANTRENARE
    print(f"🚀 Începem antrenarea pentru {EPOCHS} epoci...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks
    )

    # 5. Salvare Istoric (pentru grafice)
    history_df = pd.DataFrame(history.history)
    history_file = RESULTS_DIR / "training_history.csv"
    history_df.to_csv(history_file, index=False)
    
    print("\n========================================")
    print(f"🎉 ANTRENARE COMPLETĂ!")
    print(f"📁 Model salvat în: {MODEL_OUT_PATH}")
    print(f"📊 Istoric salvat în: {history_file}")
    print("========================================")

if __name__ == "__main__":
    main()