import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import pandas as pd

# --- CONFIGURARE ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_DIR = BASE_DIR / "data" / "train"
GEN_DIR = BASE_DIR / "data" / "generated"
DOCS_DIR = BASE_DIR / "docs"

# Asigură-te că există folderul docs
DOCS_DIR.mkdir(parents=True, exist_ok=True)

def create_distribution_chart(stats):
    """Generează un grafic cu bare: Original vs Sintetic."""
    classes = list(stats.keys())
    original_counts = [stats[c]['original'] for c in classes]
    synthetic_counts = [stats[c]['synthetic'] for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, original_counts, width, label='Originale (Kaggle/Raw)', color='skyblue')
    rects2 = ax.bar(x + width/2, synthetic_counts, width, label='Sintetice (Generate)', color='orange')

    ax.set_ylabel('Număr Imagini')
    ax.set_title('Distribuția Datelor: Contribuție Originală vs. Sursă')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()

    # Adaugă etichete pe bare
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    plt.tight_layout()
    save_path = DOCS_DIR / "data_distribution.png"
    plt.savefig(save_path)
    print(f"✅ Grafic salvat în: {save_path}")

def create_comparison_image():
    """Creează o imagine demonstrativă Before/After."""
    # Caută o clasă care are imagini în ambele locuri
    common_classes = [d.name for d in TRAIN_DIR.iterdir() if d.is_dir()]
    
    if not common_classes:
        return

    # Luăm prima clasă disponibilă
    demo_class = common_classes[0]
    
    # Luăm o imagine originală
    orig_imgs = list((TRAIN_DIR / demo_class).glob("*.jpg")) + list((TRAIN_DIR / demo_class).glob("*.png"))
    # Luăm o imagine sintetică
    synth_imgs = list((GEN_DIR / demo_class).glob("*.jpg")) + list((GEN_DIR / demo_class).glob("*.png"))

    if orig_imgs and synth_imgs:
        img_a = cv2.imread(str(orig_imgs[0]))
        img_b = cv2.imread(str(synth_imgs[0]))

        # Resize pentru a fi la fel la afișare
        img_a = cv2.resize(img_a, (300, 300))
        img_b = cv2.resize(img_b, (300, 300))

        # Le unim orizontal
        comparison = np.hstack((img_a, img_b))
        
        # Desenăm text pe ele
        cv2.putText(comparison, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "SINTETIC (Augmentat)", (310, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        save_path = DOCS_DIR / "generated_vs_real.png"
        cv2.imwrite(str(save_path), comparison)
        print(f"✅ Imagine comparativă salvată în: {save_path}")

def generate_log_file(stats):
    """Creează fișierul de log cerut."""
    log_path = DOCS_DIR / "data_statistics.csv"
    
    data = []
    total_orig = 0
    total_synth = 0
    
    for cls, counts in stats.items():
        data.append({
            "Clasa": cls,
            "Originale": counts['original'],
            "Sintetice": counts['synthetic'],
            "Total": counts['original'] + counts['synthetic']
        })
        total_orig += counts['original']
        total_synth += counts['synthetic']

    df = pd.DataFrame(data)
    df.to_csv(log_path, index=False)
    
    # Scriem și un sumar text pentru README
    with open(DOCS_DIR / "summary_log.txt", "w") as f:
        f.write(f"RAPORT GENERARE DATE - {pd.Timestamp.now()}\n")
        f.write("========================================\n")
        f.write(f"Total Imagini Originale: {total_orig}\n")
        f.write(f"Total Imagini Sintetice: {total_synth}\n")
        f.write(f"Procent Contributie Originala: {total_synth / (total_orig + total_synth) * 100:.2f}%\n")
        
    print(f"✅ Log-uri salvate în: {DOCS_DIR}")

def main():
    if not GEN_DIR.exists():
        print("❌ Nu am găsit folderul data/generated! Rulează întâi scriptul de generare.")
        return

    stats = {}
    classes = [d.name for d in TRAIN_DIR.iterdir() if d.is_dir()]

    print("Generare statistici...")
    for cls in classes:
        # Numără originale
        orig_path = TRAIN_DIR / cls
        n_orig = len(list(orig_path.glob("*.*")))
        
        # Numără sintetice
        gen_path = GEN_DIR / cls
        if gen_path.exists():
            n_gen = len(list(gen_path.glob("*.*")))
        else:
            n_gen = 0
            
        stats[cls] = {'original': n_orig, 'synthetic': n_gen}

    create_distribution_chart(stats)
    create_comparison_image()
    generate_log_file(stats)
    print("\n🎉 GATA! Dovezile sunt în folderul 'docs/'.")

if __name__ == "__main__":
    main()