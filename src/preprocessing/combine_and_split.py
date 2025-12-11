import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURARE ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

# Sursele
SOURCES = [
    DATA_DIR / "train",
    DATA_DIR / "validation",
    DATA_DIR / "test",
    DATA_DIR / "generated"
]

# Destinațiile
DESTINATIONS = {
    "train": DATA_DIR / "train",
    "validation": DATA_DIR / "validation",
    "test": DATA_DIR / "test"
}

TEMP_DIR = DATA_DIR / "temp_combine_buffer"

# Split (70% / 15% / 15%)
SPLIT_RATIOS = (0.7, 0.15, 0.15)

def combine_and_resplit():
    print("🔄 Începem recombinarea sigură a dataset-ului...")

    # 1. Colectăm toate imaginile în memorie
    all_images = [] 
    
    # Identificăm clasele din folderul generated (care sigur există)
    classes = [d.name for d in (DATA_DIR / "generated").iterdir() if d.is_dir()]
    print(f"Clase identificate: {classes}")

    # 2. Copiem TOTUL într-un folder temporar (Safe Copy)
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    print("📦 Creăm o copie temporară a datelor (Safe Buffer)...")
    
    valid_images_count = 0
    
    for source_path in SOURCES:
        if not source_path.exists():
            continue
            
        for class_name in classes:
            src_class_path = source_path / class_name
            if not src_class_path.exists():
                continue
            
            # Creăm folder clasa în temp
            (TEMP_DIR / class_name).mkdir(exist_ok=True)
            
            images = list(src_class_path.glob("*.*"))
            for img_path in images:
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    # Copiem în temp cu nume unic pentru a evita duplicate
                    # ex: temp/Cracking/train_img1.jpg
                    unique_name = f"{source_path.name}_{img_path.name}"
                    dest_temp = TEMP_DIR / class_name / unique_name
                    shutil.copy2(img_path, dest_temp)
                    
                    # Salvăm calea din temp pentru split-ul ulterior
                    all_images.append((dest_temp, class_name, unique_name))
                    valid_images_count += 1

    print(f"✅ Copie completă. Total imagini securizate: {valid_images_count}")

    if valid_images_count == 0:
        print("❌ EROARE: Nu am găsit imagini! Rulează split_data.py întâi.")
        return

    # 3. Ștergem folderele destinație (Acum e sigur, datele sunt în TEMP)
    print("🧹 Curățăm folderele finale...")
    for key, path in DESTINATIONS.items():
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        for cls in classes:
            (path / cls).mkdir(exist_ok=True)

    # 4. Amestecăm și Împărțim
    random.shuffle(all_images)

    total = len(all_images)
    train_end = int(total * SPLIT_RATIOS[0])
    val_end = train_end + int(total * SPLIT_RATIOS[1])

    splits = {
        "train": all_images[:train_end],
        "validation": all_images[train_end:val_end],
        "test": all_images[val_end:]
    }

    # 5. Mutăm din TEMP în folderele finale
    print("🚀 Distribuim fișierele în train/val/test...")
    
    for split_name, files in splits.items():
        dest_root = DESTINATIONS[split_name]
        
        for src_temp_path, class_name, filename in tqdm(files, desc=split_name):
            dest_path = dest_root / class_name / filename
            shutil.move(src_temp_path, dest_path)

    # 6. Curățenie finală
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    print("\n✅ GATA! Dataset-ul a fost unificat și redistribuit corect.")
    print(f"   Train: {len(splits['train'])}")
    print(f"   Validation: {len(splits['validation'])}")
    print(f"   Test: {len(splits['test'])}")

if __name__ == "__main__":
    combine_and_resplit()