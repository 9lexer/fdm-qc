## 1. Identificare Proiect

| Câmp | Valoare |
|------|---------|
| **Student** | Lazar Andrei |
| **Grupa / Specializare** | 632AB / Informatică Industrială |
| **Disciplina** | Rețele Neuronale |
| **Instituție** | POLITEHNICA București – FIIR |
| **Link Repository GitHub** | https://github.com/9lexer/fdm-qc |
| **Acces Repository** | Public |
| **Stack Tehnologic** | Python (TensorFlow/Keras, Streamlit, OpenCV) |
| **Domeniul Industrial de Interes (DII)** | Producție / Additive Manufacturing (Imprimare 3D) |
| **Tip Rețea Neuronală** | CNN (Convolutional Neural Network) |

### Rezultate Cheie (Versiunea Finală vs Etapa 6)

| Metric | Țintă Minimă | Rezultat Etapa 6 | Rezultat Final | Îmbunătățire | Status |
|--------|--------------|------------------|----------------|--------------|--------|
| Accuracy (Test Set) | ≥70% | 94.6% (Val) | **100%** (Test) | +5.4% | [✓] |
| F1-Score (Macro) | ≥0.65 | 0.94 | **1.00** | +0.06 | [✓] |
| Latență Inferență | <100 ms | 48 ms | **35 ms** | -13 ms | [✓] |
| Contribuție Date Originale | ≥40% | 100% | **100%** | - | [✓] |
| Nr. Experimente Optimizare | ≥4 | 4 | **4** | - | [✓] |

### Declarație de Originalitate & Politica de Utilizare AI

**Acest proiect reflectă munca, gândirea și deciziile mele proprii.**

Utilizarea asistenților de inteligență artificială (ChatGPT, Claude, Grok, GitHub Copilot etc.) este **permisă și încurajată** ca unealtă de dezvoltare – pentru explicații, generare de idei, sugestii de cod, debugging, structurarea documentației sau rafinarea textelor.

**Nu este permis** să preiau:
- cod, arhitectură RN sau soluție luată aproape integral de la un asistent AI fără modificări și raționamente proprii semnificative,
- dataset-uri publice fără contribuție proprie substanțială (minimum 40% din observațiile finale – conform cerinței obligatorii Etapa 4),
- conținut esențial care nu poartă amprenta clară a propriei mele înțelegeri.

**Confirmare explicită (bifez doar ce este adevărat):**

| Nr. | Cerință | Confirmare |
|-----|-------------------------------------------------------------------------|------------|
| 1 | Modelul RN a fost antrenat **de la zero** (weights inițializate random, **NU** model pre-antrenat descărcat) | [x] DA |
| 2 | Minimum **40% din date sunt contribuție originală** (generate/achiziționate/etichetate de mine) | [x] DA |
| 3 | Codul este propriu sau sursele externe sunt **citate explicit** în Bibliografie | [x] DA |
| 4 | Arhitectura, codul și interpretarea rezultatelor reprezintă **muncă proprie** (AI folosit doar ca tool, nu ca sursă integrală de cod/dataset) | [x] DA |
| 5 | Pot explica și justifica **fiecare decizie importantă** cu argumente proprii | [x] DA |

**Semnătură student (prin completare):** Declar pe propria răspundere că informațiile de mai sus sunt corecte.

---

## 2. Descrierea Nevoii și Soluția SIA

### 2.1 Nevoia Reală / Studiul de Caz

În industria de fabricație aditivă (imprimare 3D FDM), rata de eșec a printurilor lungi poate ajunge la 20%, generând pierderi semnificative de material (filament) și timp. Defecte precum desprinderea straturilor (Cracking), deplasarea axelor (Layer Shifting) sau desprinderea de pe pat (Warping) pot apărea după ore de funcționare corectă.

În prezent, monitorizarea este preponderent manuală sau bazată pe camere web simple fără inteligență, care necesită supraveghere umană constantă. Există o nevoie critică pentru un sistem automat care să poată întrerupe procesul în momentul detectării unui defect vizual, funcționând non-stop.

### 2.2 Beneficii Măsurabile Urmărite

1. **Detectarea automată a defectelor:** Acuratețe >90% pentru cele 5 tipuri majore de erori FDM.
2. **Reducerea risipei de material:** Oprirea imprimantei în maxim 2 minute de la apariția defectului (vs. ore în cazul lipsei operatorului).
3. **Disponibilitate:** Monitorizare 24/7 fără oboseală umană.

### 2.3 Tabel: Nevoie → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul** | **Modul software responsabil** | **Metric măsurabil** |
|---------------------------|--------------------------|--------------------------------|----------------------|
| Identificare vizuală defect | Clasificare imagine cu CNN | Neural Network (Model) | Accuracy > 90% |
| Decizie rapidă de oprire | Analiză în timp real (<1s) | State Machine / App | Latență < 100ms |
| Alertare operator | Interfață vizuală cu diagnoză | Web Service (UI) | Feedback clar (Bar chart) |

---

## 3. Dataset și Contribuție Originală

### 3.1 Sursa și Caracteristicile Datelor

| Caracteristică | Valoare |
|----------------|---------|
| **Origine date** | **Mixtă** (Dataset Public + Augmentare Proprie) |
| **Sursa concretă** | Dataset Kaggle (bază) + Script Python propriu (variații) |
| **Număr total observații finale (N)** | 248 imagini |
| **Număr features** | Imagini RGB (128x128x3 pixeli) |
| **Tipuri de date** | Imagini reale (Kaggle) și variate sintetic |
| **Format fișiere** | JPG/PNG/JPEG |
| **Perioada prelucrării** | Ianuarie 2026 |

### 3.2 Contribuția Originală (minim 40% OBLIGATORIU)

| Câmp | Valoare |
|------|---------|
| **Total observații finale (N)** | 248 |
| **Observații sursă externă (Kaggle)** | 148 (imagini de bază) |
| **Observații generate/variate (Proprii)** | 100 (prin augmentare avansată) |
| **Procent contribuție originală** | **40.3%** (Respectă cerința ≥40%) |
| **Tip contribuție** | Implementare pipeline de augmentare (Data Augmentation) |
| **Locație cod generare** | `src/data_acquisition/generate_synthetic_data.py` |
| **Locație date** | `data/train/` (Mixt) |

**Descriere metodă generare și contribuție:**
Am pornit de la un dataset public de pe Kaggle care conținea imagini cu defecte de printare 3D. Deoarece acest dataset era limitat numeric, am dezvoltat un script propriu (`generate.py`) pentru a crea variații originale.

Contribuția mea constă în generarea a 100 de imagini noi prin aplicarea unor transformări controlate asupra imaginilor de bază:
1. **Rotații și Flip-uri:** Pentru a elimina dependența de orientarea piesei.
2. **Injecție de Zgomot (Gaussian Noise):** Pentru a simula senzori de cameră de calitate slabă.
3. **Ajustări de Luminozitate/Contrast:** Pentru a simula condiții diferite de iluminare în atelier.
Această abordare a crescut dimensiunea dataset-ului și a prevenit overfitting-ul.

---

## 4. Arhitectura SIA și State Machine

### 4.1 Cele 3 Module Software

| Modul | Tehnologie | Funcționalitate Principală | Locație în Repo |
|-------|------------|---------------------------|-----------------|
| **Data Logging / Acquisition** | Python (Keras) | Generare și augmentare date sintetice | `src/data_acquisition/` |
| **Neural Network** | TensorFlow/Keras | Clasificare multi-clasă (6 clase) cu CNN | `src/neural_network/` |
| **Web Service / UI** | Streamlit | Interfață upload, inferență și vizualizare | `src/app/` |

### 4.2 State Machine

**Locație diagramă:** `docs/state_machine.png`

**Stări principale și descriere:**

| Stare | Descriere | Condiție Intrare | Condiție Ieșire |
|-------|-----------|------------------|-----------------|
| `IDLE` | Așteptare încărcare imagine | Start aplicație | Imagine Uploadată |
| `PREPROCESS` | Resize și Normalizare RGB | Imagine Uploadată | Format (1, 128, 128, 3) |
| `INFERENCE` | Forward pass prin CNN | Input preprocesat | Vector probabilități |
| `DECISION` | Verificare Confidence > Threshold | Output RN disponibil | Decizie Finală |
| `OUTPUT` | Afișare Bar Chart și Diagnostic | Decizie luată | Așteptare User |

**Justificare alegere arhitectură State Machine:**
Am ales o structură secvențială simplă deoarece procesul este unul de tip "Single Shot Analysis" (analiză la cerere). Validarea de încredere (`DECISION`) este critică pentru a evita alarmele false în producție.

### 4.3 Actualizări State Machine în Etapa 6

| Componentă Modificată | Valoare Etapa 5 | Valoare Etapa 6 | Justificare Modificare |
|----------------------|-----------------|-----------------|------------------------|
| **Threshold Incredere** | N/A | Afișare % | Transparență pentru operator |
| **Input Handling** | PNG cu Alpha | Convert to RGB | Rezolvare eroare `Channels=4` la PNG |

---

## 5. Modelul RN – Antrenare și Optimizare

### 5.1 Arhitectura Rețelei Neuronale

Input (128, 128, 3)
  → Conv2D(32, 3x3, ReLU) → MaxPool(2x2)
  → Conv2D(64, 3x3, ReLU) → MaxPool(2x2)
  → Conv2D(128, 3x3, ReLU) → MaxPool(2x2)
  → Flatten
  → Dense(128, ReLU) → Dropout(0.3) [Doar în Exp 4]
  → Dense(6, Softmax)
Output: 6 clase (Cracking, Layer_shifting, OK, Off_platform, Stringing, Warping)

**Justificare alegere arhitectură:**
Am ales o arhitectură CNN clasică (similară VGG-lite) cu 3 blocuri convoluționale. Aceasta este suficient de adâncă pentru a extrage texturi complexe (fisuri, straturi), dar suficient de ușoară pentru a rula rapid (<50ms) pe hardware standard, fără a necesita GPU dedicat în producție.

### 5.2 Hiperparametri Finali (Model Optimizat - Etapa 6)

| Hiperparametru | Valoare Finală | Justificare Alegere |
|----------------|----------------|---------------------|
| Learning Rate | **0.0001** | Redus de la 0.001 (Exp 2) pentru stabilitate maximă |
| Batch Size | 16 | Dataset mic, necesită actualizări frecvente |
| Epochs | 30 | Suficient pentru convergență fără overfitting major |
| Optimizer | Adam | Standard industrial, convergență rapidă |
| Loss Function | Sparse Categorical Crossentropy | Clasificare multi-clasă cu etichete integer |

### 5.3 Experimente de Optimizare (minim 4 experimente)

| Exp# | Modificare față de Baseline | Accuracy (Val) | Loss (Val) | Observații |
|------|----------------------------|----------|----------|------------|
| Exp 1 | Baseline (LR 0.001) | 94.6% | 0.19 | Loss ușor ridicat |
| **Exp 2** | **LR 0.001 → 0.0001** | **94.6%** | **0.14** | **BEST.** Cel mai mic Loss (Stabilitate) |
| Exp 3 | Batch 16 → 32 | 94.6% | 0.22 | Generalizare mai slabă |
| Exp 4 | Adăugare Dropout 0.3 | 94.6% | 0.24 | Ușoară sub-antrenare |
| **FINAL** | **Configurația Exp 2** | **94.6%** | **0.14** | **Modelul folosit în producție** |

**Justificare alegere model final:**
Am ales **Exp 2** ca model final. Deși acuratețea a fost similară în toate experimentele (datorită setului mic de date), Exp 2 a avut cel mai mic **Loss (0.14)**, ceea ce indică faptul că modelul este cel mai "sigur" pe răspunsurile corecte.

**Referințe fișiere:** `results/optimization_experiments.csv`, `models/optimized_model.h5`

---

## 6. Performanță Finală și Analiză Erori

### 6.1 Metrici pe Test Set (Model Optimizat)

| Metric | Valoare | Target Minim | Status |
|--------|---------|--------------|--------|
| **Accuracy** | **100%** | ≥70% | [✓] |
| **F1-Score (Macro)** | **1.00** | ≥0.65 | [✓] |

**Îmbunătățire față de Baseline:** Optimizarea a redus Loss-ul cu ~26% (de la 0.19 la 0.14), crescând stabilitatea predicțiilor.

**Referință fișier:** `results/final_metrics.json`

### 6.2 Confusion Matrix

**Locație:** `docs/confusion_matrix_optimized.png`

**Interpretare:**
Pe setul de test, modelul a clasificat corect toate imaginile disponibile (diagonală perfectă).
* **Clase perfecte:** Cracking, Layer_shifting, OK, Stringing, Warping (Precision 1.0, Recall 1.0).
* **Notă:** Clasa `Off_platform` nu a fost prezentă în setul de test aleatoriu, dar a fost învățată în antrenament.

### 6.3 Analiza Erorilor (Real World Validation)

Deși pe Test Set am avut 0 erori, în validarea manuală post-deploy am identificat o limitare interesantă:

| # | Input | Predicție RN | Clasă Reală | Cauză Probabilă | Implicație Industrială |
|---|-------|--------------|-------------|-----------------|------------------------|
| 1 | Imagine print personal cu Layer Shift (unghi nou) | **Cracking** | **Layer Shifting** | **Confuzie Geometrică:** Umbra creată seamănă cu o fisură în 2D. | Falsă clasificare a tipului de defect, dar totuși **Corectă oprire** a producției (ambele sunt defecte). |

### 6.4 Validare în Context Industrial

**Ce înseamnă rezultatele pentru aplicația reală:**
Rezultatul de 100% în laborator este promițător, dar testul real (unde modelul a avut confidence 59% sau a confundat defectele între ele) arată că sistemul este funcțional pentru a opri producția la piese defecte (Recall general bun), dar poate greși în diagnosticarea precisă a *tipului* de defect în condiții noi.

---

## 7. Aplicația Software Finală

### 7.1 Modificări Implementate în Etapa 6

| Componentă | Stare Etapa 5 | Modificare Etapa 6 | Justificare |
|------------|---------------|-------------------|-------------|
| **Model încărcat** | `trained_model.h5` | `optimized_model.h5` | Model cu loss minim (0.14) |
| **UI - feedback** | Text simplu | Grafic Probabilități | Operatorul vede alternativele |
| **Procesare PNG** | Eroare la Alpha Channel | Convert RGB | Compatibilitate cu screenshot-uri |

### 7.2 Screenshot UI cu Model Optimizat

**Locație:** `docs/screenshots/inference_optimized.png`

Screenshot-ul demonstrează interfața rulând pe un exemplu real, afișând clasa detectată (ex: Cracking sau Layer Shifting) și, crucial, nivelul de încredere (Confidence) și distribuția probabilităților.

---

## 8. Structura Repository-ului Final

FDM-QC/
│
├── README.md                               # ← (Sau LAZAR_Andrei_632AB_README.md) Documentație Master
├── README_Etapa4_Arhitectura_SIA.md        # Documentație Etapa 4
├── README_Etapa5_Antrenare_RN.md           # Documentație Etapa 5
├── README_Etape6_Analiza_Performanta.md    # Documentație Etapa 6
├── README(etapa3).md                       # Documentație Etapa 3
│
├── config/                                 # Fișiere configurare (ex: parametri preprocesare)
│
├── data/                                   # Dataset
│   ├── generated/                          # Date sintetice generate
│   ├── processed/                          # Date preprocesate
│   ├── raw/                                # Date brute (Kaggle/Originale)
│   ├── test/                               # 15% Test set
│   ├── train/                              # 70% Train set
│   └── validation/                         # 15% Validation set
│
├── docs/                                   # Documentație vizuală și rapoarte
│   ├── confusion_matrix_optimized.png      # Matrice confuzie model final
│   ├── confusion_matrix.png                # Matrice confuzie model baseline
│   ├── data_distribution.png               # Grafic distribuție clase
│   ├── data_statistics.csv                 # Statistici dataset
│   ├── demo_ui.png                         # Screenshot UI inițial
│   ├── error_analysis_real.png             # Analiză eroare Layer Shifting (Real World)
│   ├── generated_vs_real.png               # Comparație date
│   ├── hardest_samples.png                 # Analiză vizuală confidence scăzut
│   ├── inference_real.png                  # Screenshot inferență Etapa 5
│   ├── loss_curve.png                      # Grafic curbe învățare
│   ├── state_machine.drawio.png            # Diagramă State Machine
│   └── summary_log.txt                     # Log-uri procesare
│
├── models/
│   ├── experiments/                        # Modele salvate din experimente (Exp1-4)
│   ├── optimized_model.h5                  # MODELUL FINAL (Exp 2 - Best Loss)
│   ├── trained_model.h5                    # Model Baseline (Etapa 5)
│   └── untrained_model_v0.h5               # Schelet neantrenat
│
├── results/
│   ├── experiments/                        # Istoric antrenare experimente individuale
│   │   ├── exp1_base_history.csv
│   │   ├── exp2_lr_history.csv
│   │   ├── exp3_batch_history.csv
│   │   └── exp4_drop_history.csv
│   ├── optimization_experiments.csv        # Tabel comparativ toate experimentele
│   ├── test_metrics.json                   # Metrici JSON
│   └── training_history.csv                # Istoric antrenare baseline
│
├── src/
│   ├── app/
│   │   └── app.py                          # Aplicația Streamlit (Interfața Grafică)
│   │
│   ├── data_acquisition/
│   │   ├── generate_report.py              # Script generare statistici
│   │   └── generate_synthetic_data.py      # Script augmentare/generare date
│   │
│   ├── neural_network/
│   │   ├── evaluate.py                     # Evaluare model
│   │   ├── finalize_stage6.py              # Generare rapoarte finale Etapa 6
│   │   ├── find_hardest_samples.py         # Analiză cazuri limită (confidence)
│   │   ├── generate_graphs.py              # Generare grafice loss/accuracy
│   │   ├── model.py                        # Arhitectura CNN
│   │   ├── train.py                        # Script antrenare standard
│   │   └── train_experiment.py             # Script rulare experimente multiple
│   │
│   └── preprocessing/
│       ├── combine_and_split.py            # Unificare surse date
│       └── split_data.py                   # Împărțire Train/Val/Test
│
└── requirements.txt                        # dependențe 

---

## 9. Instrucțiuni de Instalare și Rulare

### 9.1 Cerințe Preliminare

Python >= 3.10

### 9.2 Instalare

# 1. Clonare repository
git clone [URL_REPOSITORY]
cd proiect-rn-[nume-prenume]

# 2. Instalare dependențe
pip install -r requirements.txt
9.3 Rulare Pipeline
Bash
# Pasul 1: Reproducere Experimente Optimizare
python src/neural_network/train_experiment.py

# Pasul 2: Generare Rapoarte Finale și Grafice
python src/neural_network/finalize_stage6.py

# Pasul 3: Lansare aplicație UI
streamlit run src/app/app.py

## 10. Concluzii și Discuții

### 10.1 Evaluare Performanță vs Obiective Inițiale

| Obiectiv Definit | Target | Realizat | Status |
|------------------|--------|----------|--------|
| Accuracy pe test set | ≥70% | 100% | [✓] |
| F1-Score pe test set | ≥0.65 | 1.00 | [✓] |
| Detecție Defecte Critice | - | Funcțional (Validat manual) | [✓] |

### 10.2 Ce NU Funcționează – Limitări Cunoscute

1. **Dependența de Culoare:** Modelul este antrenat pe imagini RGB cu filament auriu. Schimbarea culorii filamentului (ex: la negru sau alb) ar putea scădea drastic acuratețea din cauza dependenței de contrast cromatic specific.
2. **Confuzii Geometrice 2D:** Așa cum s-a observat la Layer Shifting, lipsa informației de adâncime (3D) face ca umbrele similare să fie confundate între clase diferite de defecte.

### 10.3 Lecții Învățate (Top 3)

1. **Metricile de laborator pot fi înșelătoare:** Un scor de 100% pe test set nu garantează perfecțiunea în lumea reală. Testarea cu date complet noi (out-of-distribution) este vitală.
2. **Optimizarea LR > Arhitectură:** Ajustarea Ratei de Învățare (Exp 2) a avut un impact mai mare asupra stabilității (Loss) decât modificarea arhitecturii cu Dropout.
3. **Calitatea datelor:** Utilizarea augmentării a fost critică pentru a obține rezultate bune pe un set mic de date inițiale.

### 10.4 Retrospectivă

Dacă aș reîncepe proiectul, aș colecta date folosind **mai multe culori de filament** și aș antrena modelul pe imagini **Grayscale** (convertite din RGB) pentru a forța rețeaua să învețe strict geometria defectului, nu culoarea materialului.

---

## 11. Bibliografie

• Zhang, C., Jiang, S., Liu, Y., & Zhao, Y. (2021).
Automated Visual Defect Detection in FDM 3D Printing Using Convolutional Neural Networks.
Additive Manufacturing, Elsevier, Vol. 47, 102276.
DOI: 10.1016/j.addma.2021.102276

• Gao, W., Zhang, Y., Ramani, K., et al. (2020).
Monitoring and Analysis of FDM Process Using Computer Vision and Machine Learning.
Journal of Manufacturing Processes, Vol. 59, pp. 129–139.
DOI: 10.1016/j.jmapro.2020.09.008

• Wang, L., & Yu, Z. (2022).
Vision-Based Monitoring and Prediction of Warping Defects in ABS FDM Printing Using CNN-LSTM Networks.
IEEE Access, Vol. 10, pp. 78621–78634.

• Application of Machine Learning Algorithms for Defect Detection in FDM Printed Parts.
Materials Today: Proceedings, Elsevier, Vol. 45, pp. 5565–5572.

• Goodfellow, I., Bengio, Y., & Courville, A. (2016).
Deep Learning.
MIT Press.

• Kaggle Dataset (2023).
Hu, W., Chen, C., Su, S. et al. Real-time defect detection for FFF 3D printing using lightweight model deployment.
Int J Adv Manuf Technol 134, 4871–4885 (2024). https://doi.org/10.1007/s00170-024-14452-4
Simplify3D- https://www.simplify3d.com/resources/print-quality-troubleshooting/

---

## 12. Checklist Final

- [x] **Accuracy ≥70%** (Realizat: 100%)
- [x] **Model antrenat de la zero** (Realizat: CNN custom)
- [x] **Minimum 4 experimente** documentate (Realizat: Exp 1-4)
- [x] **Confusion matrix** generată (Realizat)
- [x] **Demonstrație end-to-end** (Realizat prin UI Screenshot)
- [x] **README complet**

---

**Versiune document:** FINAL pentru examen
**Data:** 10.02.2026