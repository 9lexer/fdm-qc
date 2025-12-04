import os
import cv2
import numpy as np
import random
from pathlib import Path

# --- CONFIGURARE ---
# Scriptul va detecta automat folderul radacina 'fdm-qc'
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# AICI POȚI ALEGE SURSA:
# Varianta A: Folosești 'train' (Recomandat pentru ML corect)
INPUT_DIR = BASE_DIR / "data" / "train" 

# Varianta B: Dacă vrei să generezi din toate datele procesate, schimbă linia de sus cu:
# INPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR = BASE_DIR / "data" / "generated"
GENERATION_RATIO = 1.0 # 1.0 înseamnă că pentru fiecare poză veche generezi una nouă

def simulate_industrial_conditions(image):
    # 1. Simulare Iluminare (Brightness/Contrast)
    alpha = random.uniform(0.8, 1.2) # Contrast
    beta = random.randint(-40, 40)   # Brightness
    img_adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # 2. Simulare Zgomot (Gaussian Noise)
    row, col, ch = img_adjusted.shape
    mean = 0
    var = random.uniform(10, 50)
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy_img = img_adjusted + gauss
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    return noisy_img

def main():
    print(f"Director proiect detectat: {BASE_DIR}")
    print(f"Caut imagini în: {INPUT_DIR}")

    if not INPUT_DIR.exists():
        print(f"EROARE CRITICĂ: Folderul {INPUT_DIR} nu există!")
        print("Verifică dacă ai rulat pasul anterior (split_data) sau modifică INPUT_DIR în cod.")
        return

    # Pregătire output
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_generated = 0
    classes = [d.name for d in INPUT_DIR.iterdir() if d.is_dir()]
    
    for class_name in classes:
        input_class_path = INPUT_DIR / class_name
        output_class_path = OUTPUT_DIR / class_name
        output_class_path.mkdir(parents=True, exist_ok=True)
        
        # Căutăm extensii comune
        images = list(input_class_path.glob("*.jpg")) + list(input_class_path.glob("*.png")) + list(input_class_path.glob("*.jpeg"))
        
        target_count = int(len(images) * GENERATION_RATIO)
        
        if target_count == 0:
            print(f"⚠ Atenție: Nu am găsit imagini în clasa {class_name}")
            continue

        selected_images = random.choices(images, k=target_count)
        
        print(f"  > Clasa '{class_name}': Generăm {target_count} imagini sintetice...")
        
        for idx, img_path in enumerate(selected_images):
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            synthetic_img = simulate_industrial_conditions(img)
            
            save_name = f"synth_{idx}_{img_path.name}"
            cv2.imwrite(str(output_class_path / save_name), synthetic_img)
            total_generated += 1

    print(f"\n✅ SUCCES! Ai generat {total_generated} imagini în {OUTPUT_DIR}")

if __name__ == "__main__":
    main()