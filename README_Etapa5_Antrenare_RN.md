# 📘 README – Etapa 5: Configurarea și Antrenarea Modelului RN

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Lazar Andrei 
**Link Repository GitHub:** https://github.com/9lexer/fdm-qc
**Data predării:** 11.12.2025

---

## Scopul Etapei 5

Această etapă corespunde punctului **6. Configurarea și antrenarea modelului RN** din specificațiile proiectului.
Am realizat antrenarea efectivă a modelului CNN definit în Etapa 4, utilizând un dataset mixt (date originale + date sintetice generate). Modelul a fost evaluat pe un set de test independent, obținând performanțe ridicate, și a fost integrat în aplicația de monitorizare (Streamlit) pentru inferență în timp real.

---

## 1. Configurare și Hiperparametri (Nivel 1)

Am antrenat modelul folosind biblioteca **TensorFlow/Keras**. Mai jos sunt detaliile configurației utilizate pentru a obține rezultatele finale.

| **Hiperparametru** | **Valoare Aleasă** | **Justificare** |
|--------------------|-------------------|-----------------|
| **Learning rate** | 0.001 (Dinamic) | Am pornit cu valoarea standard 0.001 pentru optimizatorul Adam. Am utilizat `ReduceLROnPlateau` pentru a micșora rata (factor 0.5) când loss-ul s-a plafonat, permițând o ajustare fină a greutăților. |
| **Batch size** | 16 | Dataset-ul fiind relativ mic (~250 imagini), un batch size de 16 a asigurat actualizări frecvente ale gradienților și o generalizare mai bună decât un batch mare. |
| **Number of epochs** | 30 (Stop la 13) | Am setat o limită de 30 epoci, dar mecanismul de **Early Stopping** a oprit automat antrenarea la epoca 13, detectând că modelul a atins performanța maximă și prevenind overfitting-ul. |
| **Optimizer** | Adam | Algoritm adaptiv (Adaptive Moment Estimation), standardul industrial pentru rețele CNN, deoarece gestionează eficient ratele de învățare per parametru. |
| **Loss function** | Sparse Categorical Crossentropy | Funcția optimă pentru clasificare multi-clasă (6 clase: OK + 5 defecte) unde etichetele sunt furnizate ca numere întregi (sparse). |
| **Data Split** | 70% / 15% / 15% | Împărțire stratificată standard: 70% antrenare pentru învățare, 15% validare pentru tuning hiperparametri și 15% test pentru evaluare finală obiectivă. |

---

## 2. Rezultate și Metrici (Test Set)

Modelul a fost evaluat pe setul de test (imagini pe care nu le-a văzut niciodată la antrenare sau validare).

### Metrici Finale:
* **Acuratețe (Accuracy):** **97.37%** (Obiectiv atins: >65%)
* **F1-Score (Macro):** **0.9704** (Obiectiv atins: >60%)
* **Loss:** 0.1226

Aceste rezultate indică faptul că modelul generalizează excelent și distinge corect între cele 6 tipuri de defecte și piese OK, având o rată foarte mică de confuzie.

**Fișiere doveditoare în repo:**
* Istoric antrenare (toate epocile): `results/training_history.csv`
* Metrici finale JSON: `results/test_metrics.json`

---

## 3. Analiză Erori în Context Industrial (Nivel 2 - Obligatoriu)

**1. Pe ce clase greșește cel mai mult modelul?**
Având o acuratețe de ~97%, erorile sunt extrem de rare. Singurele confuzii potențiale apar între clasele geometrice similare, precum `Warping` (deformare colț) și `Off_platform` (desprindere totală), în stadiile incipiente ale defectului.

**2. Ce caracteristici ale datelor cauzează erori?**
Modelul utilizează imagini RGB. O provocare majoră o constituie **iluminarea variabilă** și **reflexiile filamentului** (în special la materialele lucioase/silk). Umbrele puternice pot fi interpretate eronat ca fisuri (`Cracking`) dacă contrastul nu este gestionat corect prin augmentare.

**3. Ce implicații are pentru aplicația industrială?**
* **False Positives (Alarmă falsă):** Sunt acceptabile. Dacă modelul oprește imprimanta pentru o piesă care era de fapt OK, pierdem puțin timp de producție, dar nu risipim material.
* **False Negatives (Defect ratat):** Sunt critice. Totuși, F1-score-ul de 0.97 arată că rata de detecție a defectelor reale (Recall) este foarte mare, deci riscul ca un defect să ajungă la client este minim.

**4. Măsuri corective propuse:**
1.  **Iluminare Controlată:** Implementarea unei benzi LED pe cadrul imprimantei pentru a elimina umbrele și a standardiza inputul vizual.
2.  **Dataset Extins:** Colectarea mai multor date pentru defectele fine (stringing subtil) care sunt greu de văzut la rezoluția 128x128.
3.  **Grayscale Conversion:** Pe viitor, testarea conversiei la Alb-Negru pentru a face modelul independent de culoarea filamentului utilizat.

---

## 4. Integrare UI și State Machine

Modelul antrenat (`models/trained_model.h5`) a fost integrat cu succes în aplicația Streamlit.

**Conformitate cu State Machine (Etapa 4):**
Fluxul implementat respectă diagrama definită anterior:
1.  **ACQUIRE:** Imaginea este încărcată în UI.
2.  **PREPROCESS:** Se aplică resize la 128x128 și normalizare (1/255), identic cu antrenarea.
3.  **INFERENCE:** Modelul antrenat returnează vectorul de probabilități.
4.  **DECISION:** Se afișează clasa cu probabilitate maximă și scorul de încredere.

**Dovadă funcționare:**
Screenshot-ul inferenței reale se află în: `docs/screenshots/inference_real.png`.

---

## 5. Grafice și Bonusuri (Nivel 2 & 3)

### Nivel 2: Curba de Învățare
Graficul de mai jos demonstrează convergența modelului și efectul Early Stopping (evitarea overfitting-ului). Loss-ul de validare scade constant odată cu cel de antrenare.
**Fișier:** `docs/loss_curve.png`

### Nivel 3 (Bonus): Matricea de Confuzie
Am generat matricea de confuzie pe setul de test pentru a analiza detaliat erorile.
**Fișier:** `docs/confusion_matrix.png`

**Analiză Bonus:**
Matricea arată o diagonală principală puternică, confirmând acuratețea ridicată. Puținele erori din afara diagonalei indică confuzii minore, cel mai probabil între clasele vizual similare.

---

## Structura Repository-ului (Etapa 5)

fdm-qc/
├── README_Etapa3.md               
├── README_Etapa4_Arhitectura_SIA.md
├── README_Etapa5_Antrenare_RN.md  # 🆕 DOCUMENTAȚIA PENTRU ACEASTĂ SĂPTĂMÂNĂ
├── requirements.txt               # Actualizat (tensorflow, streamlit, pandas, seaborn, etc.)
│
├── config/                        
│
├── data/
│   ├── raw/                       # Datele originale (Kaggle/Poze proprii)
│   ├── generated/                 # Datele sintetice (Contribuția ta 40-50%)
│   ├── processed/                 # (Opțional) Date intermediare
│   ├── train/                     # 🆕 Conține MIX (Raw + Generated) - 70%
│   ├── validation/                # 🆕 Conține MIX (Raw + Generated) - 15%
│   └── test/                      # 🆕 Conține MIX (Raw + Generated) - 15%
│
├── docs/
│   ├── state_machine.png          # Din Etapa 4
│   ├── data_distribution.png      # Dovadă Etapa 4 (Raport)
│   ├── generated_vs_real.png      # Dovadă Etapa 4 (Raport)
│   ├── loss_curve.png             # 🆕 Nivel 2: Graficul de Loss
│   ├── confusion_matrix.png       # 🆕 Nivel 3: Matricea de Confuzie
│   └── screenshots/
│       ├── ui_demo.png            # Etapa 4 (UI Vechi)
│       └── inference_real.png     # 🆕 Etapa 5 (UI cu Model Antrenat)
│
├── models/
│   ├── untrained_model_v0.h5      # Scheletul din Etapa 4
│   └── trained_model.h5           # 🆕 MODELUL FINAL ANTRENAT
│
├── results/                       # 🆕 FOLDER NOU
│   ├── training_history.csv       # Log-ul epocilor (folosit pt grafice)
│   └── test_metrics.json          # Acuratețe și F1 pe setul de test
│
└── src/
    ├── data_acquisition/
    │   ├── generate_synthetic_data.py # Script Etapa 4
    │   └── generate_report.py         # Script Raportare Etapa 4
    │
    ├── preprocessing/
    │   ├── split_data.py              # Script vechi
    │   └── combine_and_split.py       # 🆕 Scriptul care a unit datele
    │
    ├── neural_network/
    │   ├── model.py                   # Arhitectura
    │   ├── train.py                   # 🆕 Script Antrenare
    │   ├── evaluate.py                # 🆕 Script Evaluare
    │   └── generate_graphs.py         # 🆕 Script Grafice (Loss/Confusion)
    │
    └── app/
        └── app.py                     # 🆕 UI Actualizat (Inferență Reală)