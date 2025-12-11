import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from pathlib import Path

# --- CONFIGURARE ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
HISTORY_PATH = BASE_DIR / "results" / "training_history.csv"
MODEL_PATH = BASE_DIR / "models" / "trained_model.h5"
TEST_DIR = BASE_DIR / "data" / "test"
DOCS_DIR = BASE_DIR / "docs"
IMAGE_SIZE = (128, 128)

def plot_loss_curve():
    """Generare Grafic Loss (Cerință Nivel 2)"""
    if not HISTORY_PATH.exists():
        print("❌ Nu găsesc training_history.csv")
        return

    df = pd.read_csv(HISTORY_PATH)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['loss'], label='Training Loss', color='blue')
    plt.plot(df['val_loss'], label='Validation Loss', color='orange')
    plt.title('Curba de Învățare (Loss vs Epochs)')
    plt.xlabel('Epoci')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = DOCS_DIR / "loss_curve.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Grafic Loss salvat în: {save_path}")

def plot_confusion_matrix():
    """Generare Matrice de Confuzie (Cerință Nivel 3 - Bonus)"""
    if not MODEL_PATH.exists():
        print("❌ Nu găsesc modelul antrenat")
        return

    # Încărcare date test
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels='inferred',
        label_mode='int',
        image_size=IMAGE_SIZE,
        batch_size=32,
        shuffle=False 
    )
    class_names = test_ds.class_names
    
    # Normalizare
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    # Încărcare model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Predicții
    print("🔄 Generare predicții pentru Matricea de Confuzie...")
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    # Calcul Matrice
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matricea de Confuzie (Test Set)')
    plt.ylabel('Eticheta Reală')
    plt.xlabel('Eticheta Predisă')
    
    save_path = DOCS_DIR / "confusion_matrix.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Matrice de Confuzie salvată în: {save_path}")

if __name__ == "__main__":
    plot_loss_curve()
    plot_confusion_matrix()