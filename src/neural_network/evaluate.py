import tensorflow as tf
from sklearn.metrics import f1_score
import numpy as np
import json
from pathlib import Path

# --- CONFIGURARE ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "trained_model.h5"
TEST_DIR = BASE_DIR / "data" / "test"
RESULTS_DIR = BASE_DIR / "results"
IMAGE_SIZE = (128, 128)

def evaluate_model():
    if not MODEL_PATH.exists():
        print(f"❌ Nu găsesc modelul antrenat: {MODEL_PATH}")
        return

    print("🔄 Încărcăm setul de test...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels='inferred',
        label_mode='int',
        image_size=IMAGE_SIZE,
        batch_size=32,
        shuffle=False # IMPORTANT: Fără shuffle pentru a alinia corect etichetele
    )

    # Normalizare
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    # Încărcare model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 1. Evaluare Standard (Loss & Accuracy)
    print("🚀 Evaluăm modelul...")
    loss, acc = model.evaluate(test_ds, verbose=1)

    # 2. Calcul F1-Score
    print("📊 Calculăm F1-score...")
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred_probs = model.predict(test_ds)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    f1_macro = f1_score(y_true, y_pred_classes, average='macro')

    # 3. Salvare Rezultate
    metrics = {
        "test_accuracy": round(acc, 4),
        "test_f1_macro": round(f1_macro, 4),
        "test_loss": round(loss, 4)
    }

    with open(RESULTS_DIR / "test_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)

    print("\n========================================")
    print(f"✅ REZULTATE FINALE PE TEST SET:")
    print(f"   Accuracy: {metrics['test_accuracy'] * 100:.2f}%")
    print(f"   F1-Score: {metrics['test_f1_macro']:.4f}")
    print("========================================")
    print(f"📁 Salvat în: {RESULTS_DIR / 'test_metrics.json'}")

if __name__ == "__main__":
    evaluate_model()