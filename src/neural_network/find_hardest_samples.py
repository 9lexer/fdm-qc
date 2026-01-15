import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

# --- CONFIGURARE ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "optimized_model.h5"
TEST_DIR = BASE_DIR / "data" / "test"
DOCS_DIR = BASE_DIR / "docs"

def find_hardest_samples():
    print("🔄 Căutăm cele mai dificile exemple (Low Confidence)...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels='inferred',
        label_mode='int',
        image_size=(128, 128),
        batch_size=32,
        shuffle=False 
    )
    class_names = test_ds.class_names
    
    # Colectăm toate predicțiile
    results = []
    
    for img_batch, label_batch in test_ds:
        img_norm = img_batch / 255.0
        preds = model.predict(img_norm, verbose=0)
        
        for i in range(len(preds)):
            prob = preds[i]
            pred_idx = np.argmax(prob)
            conf = np.max(prob) * 100
            true_idx = label_batch[i].numpy()
            
            # Salvăm detaliile
            results.append({
                "image": img_batch[i].numpy().astype("uint8"),
                "true_label": class_names[true_idx],
                "pred_label": class_names[pred_idx],
                "confidence": conf,
                "is_correct": pred_idx == true_idx
            })

    # Sortăm după încredere (de la mic la mare)
    # Ne interesează unde modelul a fost cel mai "nesigur", chiar dacă a răspuns corect
    results.sort(key=lambda x: x["confidence"])
    
    # Luăm primele 5 (cele mai nesigure)
    hardest = results[:5]
    
    print(f"✅ Am găsit {len(hardest)} exemple cu încredere scăzută.")

    # Desenăm Grid-ul
    plt.figure(figsize=(15, 5))
    for i, item in enumerate(hardest):
        plt.subplot(1, 5, i+1)
        plt.imshow(cv2.cvtColor(item['image'], cv2.COLOR_BGR2RGB))
        
        # Colorăm titlul: Verde dacă e corect, Roșu dacă e greșit (dar la tine toate vor fi verzi)
        color = 'green' if item['is_correct'] else 'red'
        
        plt.title(f"Real: {item['true_label']}\nPred: {item['pred_label']}\nConf: {item['confidence']:.1f}%", 
                  color=color, fontsize=10, fontweight='bold')
        plt.axis('off')
        
    plt.tight_layout()
    save_path = DOCS_DIR / "hardest_samples.png"
    plt.savefig(save_path)
    print(f"📁 Imagine salvată în: {save_path}")

if __name__ == "__main__":
    find_hardest_samples()