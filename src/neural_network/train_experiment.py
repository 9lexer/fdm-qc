import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
from pathlib import Path
import argparse
import os

# --- CONFIGURARE BASE ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_DIR = BASE_DIR / "data" / "train"
VAL_DIR = BASE_DIR / "data" / "validation"
RESULTS_DIR = BASE_DIR / "results" / "experiments"
MODELS_DIR = BASE_DIR / "models" / "experiments"

# Ne asigurăm că există folderele pentru experimente
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (128, 128)

def run_experiment(exp_name, learning_rate, batch_size, epochs, dropout_rate):
    print(f"\n🧪 START EXPERIMENT: {exp_name}")
    print(f"   LR={learning_rate}, Batch={batch_size}, Dropout={dropout_rate}")

    # 1. Încărcare Date (cu Batch Size variabil)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels='inferred',
        label_mode='int',
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels='inferred',
        label_mode='int',
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False
    )

    # Normalizare
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # 2. Definire Model (Reconstruim modelul aici pentru a putea schimba Dropout-ul)
    # Copiem arhitectura din model.py dar adăugăm Dropout variabil
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        tf.keras.layers.Flatten(),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),  # <--- AICI FOLOSIM PARAMETRUL
        
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Compilare cu Learning Rate variabil
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 3. Callbacks
    model_path = MODELS_DIR / f"{exp_name}.h5"
    
    callbacks = [
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]

    # 4. Antrenare
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Salvare Rezultate
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(RESULTS_DIR / f"{exp_name}_history.csv", index=False)
    
    # Returnăm ultima acuratețe de validare
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"✅ Experiment {exp_name} GATA! Val Accuracy: {final_val_acc:.4f}")
    return final_val_acc

if __name__ == "__main__":
    # Definim lista de experimente direct aici
    experiments = [
        # Nume       LR      Batch  Epochs  Dropout
        ("exp1_base", 0.001,  16,    30,     0.0), # Referința (ce ai avut până acum, aprox)
        ("exp2_lr",   0.0001, 16,    30,     0.0), # Learning Rate mai mic (fine tuning)
        ("exp3_batch",0.001,  32,    30,     0.0), # Batch size mai mare
        ("exp4_drop", 0.001,  16,    30,     0.3)  # Adăugăm Dropout (reduce overfitting)
    ]

    results = []
    
    print("🚀 ÎNCEPERE SESIUNE DE EXPERIMENTE...")
    
    for exp_params in experiments:
        name, lr, batch, eps, drop = exp_params
        acc = run_experiment(name, lr, batch, eps, drop)
        results.append({
            "Experiment": name,
            "Learning Rate": lr,
            "Batch Size": batch,
            "Dropout": drop,
            "Val Accuracy": acc
        })

    # Afișare tabel final
    print("\n📊 REZULTATE FINALE EXPERIMENTE:")
    df_results = pd.DataFrame(results)
    print(df_results)
    
    # Salvare tabel sinteză
    df_results.to_csv(BASE_DIR / "results" / "optimization_experiments.csv", index=False)
    print(f"\nRaport salvat în results/optimization_experiments.csv")