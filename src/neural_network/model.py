import tensorflow as tf
from tensorflow.keras import layers, models
import os
from pathlib import Path

# Calea unde salvăm modelul "schelet"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
SAVE_PATH = MODELS_DIR / 'untrained_model_v0.h5'

def create_model():
    # Definim arhitectura CNN (input 128x128, 6 clase)
    model = models.Sequential([
        layers.Input(shape=(128, 128, 3)),
        
        # Strat 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Strat 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        # Output layer: 6 clase (OK + 5 defecte)
        layers.Dense(6, activation='softmax') 
    ])
    
    # Compilam modelul (obligatoriu ca sa fie functional)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Cream folderul models daca nu exista
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model = create_model()
    model.save(SAVE_PATH)
    print(f"✅ GATA: Modelul neantrenat a fost salvat în: {SAVE_PATH}")