# README – Etapa 6: Analiza Performanței, Optimizarea și Concluzii Finale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Lazar Andrei 
**Data predării:** 15.01.2026

---
## Scopul Etapei 6

Această etapă finalizează ciclul de dezvoltare al Sistemului cu Inteligență Artificială (SIA). 
Am realizat optimizarea modelului prin experimente sistematice, am analizat performanța finală (atingând 100% acuratețe pe setul de test disponibil) și am integrat modelul optimizat în aplicația software maturizată. De asemenea, am efectuat o validare manuală post-deploy care a evidențiat limitări subtile ale vederii computerizate 2D.

---

## 1. Optimizarea Parametrilor și Experimentare

Am realizat **4 experimente** distincte pentru a identifica cea mai bună configurație, variind rata de învățare (Learning Rate), dimensiunea lotului (Batch Size) și regularizarea (Dropout).

### Tabel Experimente de Optimizare

| **Exp#** | **Modificare față de Baseline** | **Accuracy** | **Loss** | **Observații** |
|----------|---------------------------------|--------------|----------|----------------|
| Exp 1 (Base)| Configurație standard (LR 0.001) | 94.6% | 0.19 | Convergență rapidă, dar loss ușor ridicat. |
| **Exp 2 (LR)** | **Learning rate 0.001 → 0.0001** | **94.6%** | **0.14** | **BEST.** Cea mai stabilă învățare și cea mai mică eroare (Loss). |
| Exp 3 (Batch)| Batch size 16 → 32 | 94.6% | 0.22 | Antrenare mai rapidă per epocă, dar generalizare ușor mai slabă. |
| Exp 4 (Drop)| Adăugare Dropout 0.3 | 94.6% | 0.24 | Dropout-ul a introdus o ușoară sub-antrenare pe acest dataset mic. |

**Justificare alegere configurație finală:**
Am selectat **Exp 2** ca model final (`optimized_model.h5`). Deși toate experimentele au atins o acuratețe similară (limitare statistică dată de dimensiunea mică a dataset-ului de validare), Exp 2 a obținut cel mai mic **Validation Loss (0.14)**. Un Loss mai mic indică faptul că modelul este mai "sigur" pe predicțiile sale corecte, fiind preferabil pentru stabilitate în producție.

---

## 2. Analiza Detaliată a Performanței

### 2.1 Confusion Matrix (Set de Test)

**Locație:** `docs/confusion_matrix_optimized.png`

**Interpretare:**
Pe setul de test rezervat, modelul a obținut o performanță ideală:
* **Acuratețe Globală:** 100%
* **Precision & Recall:** 1.00 pentru clasele prezente (Cracking, Layer_shifting, OK, Stringing, Warping).
* **Notă:** Acest rezultat indică faptul că modelul a învățat perfect trăsăturile din dataset-ul disponibil, dar necesită validare suplimentară pe cazuri "la limită" (vezi secțiunea 2.2).

### 2.2 Analiza Erorilor în Scenarii Reale (Post-Deploy)

Deși metricile de laborator au fost perfecte, testarea manuală ulterioară cu imagini noi a scos la iveală un caz de confuzie interesant, relevant pentru limitele tehnologiei.

**Imagine de Referință:** `docs/screenshots/error_analysis_real.png`

| **Clasă Reală** | **Predicție Model** | **Confidence** | **Analiză Cauzală** |
|-----------------|---------------------|----------------|---------------------|
| **Layer Shifting** | **Cracking** | 83.58% | **Ambiguitate 2D:** Atât deplasarea de strat (Layer Shift), cât și fisurarea (Cracking) apar vizual ca linii orizontale întunecate pe piesă. <br><br>În imaginea analizată, "treapta" specifică deplasării a creat o umbră puternică pe care modelul a interpretat-o ca o despărțitură între straturi (fisură). Lipsa informației de adâncime (3D) a dus la această confuzie geometrică (Overlap in Feature Space). |

**Măsură Corectivă Propusă:**
Pentru a elimina această confuzie în versiunea 2.0 a sistemului:
1. **Augmentare specifică:** Antrenarea cu imagini ce au iluminare venind din unghiuri diferite pentru a schimba comportamentul umbrelor.
2. **Senzori 3D:** Utilizarea unui profilometru laser care ar detecta imediat diferența de volum (treapta), invizibilă pentru o cameră 2D standard în anumite condiții de lumină.

---

## 3. Actualizarea Aplicației Software

Am maturizat aplicația pentru a reflecta stadiul final al proiectului și a oferi transparență totală operatorului.

### Tabel Modificări Aplicație Software

| **Componenta** | **Stare Etapa 5** | **Modificare Etapa 6** | **Justificare** |
|----------------|-------------------|------------------------|-----------------|
| **Model Încărcat** | `trained_model.h5` | `optimized_model.h5` | Loss redus cu ~26% (stabilitate crescută). |
| **Interfață UI** | Avertisment "Neantrenat" | Modul "Inferență Reală" | Aplicația este gata de producție (TRL 6). |
| **Validare Input** | Niciuna | Verificare structură date | Previne erorile de mapare a claselor întâlnite anterior. |
| **Vizualizare** | Text simplu | Grafic probabilități | Operatorul poate vedea alternativele (ex: Cracking vs Layer Shifting). |
| **Screenshot** | `inference_real.png` | `inference_optimized.png` | Dovada funcționării versiunii finale optimizate. |

---

## 4. Concluzii Finale și Lecții Învățate

### Evaluare sintetică
Proiectul a atins obiectivele tehnice propuse:
- ✅ **Acuratețe Finală:** 100% pe test set (Target > 70%).
- ✅ **F1-Score:** 1.00 (Target > 0.65).
- ✅ **Sistem Integrat:** Pipeline funcțional complet (Data Acquisition -> Preprocessing -> Model -> User Interface).

### Limitări Identificate
1. **Dependența de Iluminare/Perspective 2D:** Așa cum a demonstrat eroarea Layer Shifting vs Cracking, modelul este sensibil la umbrele care mimică alte defecte.
2. **Dimensiunea Dataset-ului:** Numărul redus de imagini (~250) a dus la o distribuție statistică inegală în test (clasa `Off_platform` nu a fost reprezentată în test set), deși a fost învățată corect la antrenare.

### Lecții Învățate
1. **Metricile nu sunt totul:** Un model poate avea acuratețe 100% pe test set și totuși să greșească în realitate. Testarea manuală "ad-hoc" este critică.
2. **Importanța Datelor Sintetice:** Generarea augmentată a fost esențială pentru a obține convergență pe un set de date atât de mic.
3. **Optimizarea Hiperparametrilor:** Reducerea Learning Rate-ului (Exp 2) a fost mult mai eficientă decât complicarea arhitecturii (Dropout) pentru acest caz specific.

---

## Livrabile și Structură
Fișierele esențiale care susțin această etapă:

* **Model Final:** `models/optimized_model.h5`
* **Raport Experimente:** `results/optimization_experiments.csv`
* **Confusion Matrix:** `docs/confusion_matrix_optimized.png`
* **Exemplu Eroare Reală:** `docs/screenshots/error_analysis_real.png`
* **Screenshot UI Final:** `docs/screenshots/inference_optimized.png`