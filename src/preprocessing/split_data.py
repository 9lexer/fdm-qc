import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path

# --- CONFIGURARE ---
# Calea relativă către date (presupunând că scriptul e în src/preprocessing)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIRS = {
    "train": BASE_DIR / "data" / "train",
    "validation": BASE_DIR / "data" / "validation",
    "test": BASE_DIR / "data" / "test"
}

# Dimensiunea specificată în PPT (Arhitectura CNN)
IMG_SIZE = (128, 128) 

# Procentele de împărțire (70% antrenare, 15% validare, 15% testare)
SPLIT_RATIOS = (0.7, 0.15, 0.15)

def create_dir_structure():
    """Creează folderele goale, ștergând conținutul vechi dacă există."""
    for key, path in OUTPUT_DIRS.items():
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    
    if PROCESSED_DATA_DIR.exists():
        shutil.rmtree(PROCESSED_DATA_DIR)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

def process_and_split():
    if not RAW_DATA_DIR.exists():
        print(f"Eroare: Folderul {RAW_DATA_DIR} nu există!")
        return

    create_dir_structure()
    
    # Identificăm clasele (folderele din raw: Cracking, OK, Warping, etc.)
    classes = [d.name for d in RAW_DATA_DIR.iterdir() if d.is_dir()]
    print(f"Clase identificate: {classes}")

    total_images = 0

    for class_name in classes:
        class_path = RAW_DATA_DIR / class_name
        # Luăm toate imaginile jpg/png
        images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png")) + list(class_path.glob("*.jpeg"))
        
        # Amestecăm aleatoriu imaginile pentru a evita bias-ul
        random.shuffle(images)
        
        # Calculăm indecșii de tăiere
        n = len(images)
        train_end = int(n * SPLIT_RATIOS[0])
        val_end = train_end + int(n * SPLIT_RATIOS[1])
        
        splits = {
            "train": images[:train_end],
            "validation": images[train_end:val_end],
            "test": images[val_end:]
        }

        print(f"Procesare clasa '{class_name}': {n} imagini -> Train: {len(splits['train'])}, Val: {len(splits['validation'])}, Test: {len(splits['test'])}")

        for split_type, split_imgs in splits.items():
            # Creăm subfolderul pentru clasă (ex: data/train/Cracking)
            dest_dir = OUTPUT_DIRS[split_type] / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Creăm și folderul în 'processed' (pentru referință)
            processed_class_dir = PROCESSED_DATA_DIR / class_name
            processed_class_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_imgs:
                try:
                    # 1. Citire imagine
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # 2. Redimensionare (Preprocesare conform PPT)
                    img_resized = cv2.resize(img, IMG_SIZE)
                    
                    # 3. Salvare în destinația finală (Train/Val/Test)
                    cv2.imwrite(str(dest_dir / img_path.name), img_resized)
                    
                    # 4. Salvare copie în 'processed' (opțional, dar bun pentru verificare)
                    cv2.imwrite(str(processed_class_dir / img_path.name), img_resized)
                    
                    total_images += 1
                except Exception as e:
                    print(f"Eroare la procesarea {img_path.name}: {e}")

    print(f"\n✅ Gata! Au fost procesate și organizate {total_images} imagini.")

if __name__ == "__main__":
    process_and_split()