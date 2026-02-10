import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import pandas as pd
import cv2

# --- CONFIGURARE ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "optimized_model.h5"
TEST_DIR = BASE_DIR / "data" / "test"
DOCS_DIR = BASE_DIR / "docs"
RESULTS_DIR = BASE_DIR / "results"

DOCS_DIR.mkdir(parents=True, exist_ok=True)

def analyze_performance():
    if not MODEL_PATH.exists():
        print("❌ EROARE: Nu găsesc models/optimized_model.h5!")
        return

    print("🔄 Încărcare model optimizat...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("🔄 Încărcare set de test...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels='inferred',
        label_mode='int',
        image_size=(128, 128),
        batch_size=32,
        shuffle=False 
    )
    
    class_names = test_ds.class_names
    # Creăm lista completă de indexi (0, 1, 2, 3, 4, 5)
    all_labels = np.arange(len(class_names))
    
    y_true = []
    y_pred_probs = []
    images = []
    
    for img_batch, label_batch in test_ds:
        img_norm = img_batch / 255.0
        preds = model.predict(img_norm, verbose=0)
        
        y_true.extend(label_batch.numpy())
        y_pred_probs.extend(preds)
        images.extend(img_batch.numpy().astype("uint8"))

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 1. Matrice de Confuzie (Forțăm dimensiunea 6x6 folosind labels=all_labels)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matricea de Confuzie (Model Optimizat)')
    plt.ylabel('Real')
    plt.xlabel('Predis')
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "confusion_matrix_optimized.png")
    print("✅ Matrice salvată în docs/confusion_matrix_optimized.png")

    # 2. Raport Metrici (Corecția este aici: labels=all_labels)
    print("\n📊 RAPORT FINAL (TEST SET):")
    # zero_division=0 ascunde avertismentele pentru clasa lipsă
    print(classification_report(y_true, y_pred, labels=all_labels, target_names=class_names, zero_division=0))

    # 3. Analiza Erorilor
    errors_indices = np.where(y_true != y_pred)[0]
    print(f"⚠️ Număr total erori pe test: {len(errors_indices)}")
    
    if len(errors_indices) > 0:
        plt.figure(figsize=(15, 5))
        num_show = min(5, len(errors_indices))
        
        for i, idx in enumerate(errors_indices[:num_show]):
            true_cls = class_names[y_true[idx]]
            pred_cls = class_names[y_pred[idx]]
            conf = np.max(y_pred_probs[idx]) * 100
            
            plt.subplot(1, 5, i+1)
            plt.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
            plt.title(f"Real: {true_cls}\nPred: {pred_cls}\nConf: {conf:.1f}%", color='red', fontsize=9)
            plt.axis('off')
            
        plt.tight_layout()
        plt.savefig(DOCS_DIR / "error_analysis_examples.png")
        print("✅ Exemple de erori salvate în docs/error_analysis_examples.png")
    else:
        print("🎉 Zero erori pe setul de test!")

if __name__ == "__main__":
    analyze_performance()