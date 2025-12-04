# 📘 Documentație Set de Date - FDM Defect Detection

## 1. Prezentare Generală
Acest set de date este utilizat pentru antrenarea unei rețele neuronale convoluționale (CNN) capabile să detecteze defecte de imprimare 3D (FDM) în timp real. Structura datelor respectă arhitectura propusă în documentația de proiect.

## 2. Structura Datelor
Datele sunt organizate în trei subseturi principale, generate automat prin scriptul `src/preprocessing/split_data.py`:

* **data/raw/**: Imaginile originale, organizate pe foldere (clase).
* **data/processed/**: Imagini redimensionate la 128x128px (formatul de intrare CNN).
* **data/train/** (70%): Folosit pentru antrenarea modelului.
* **data/validation/** (15%): Folosit pentru evaluarea în timpul antrenării (tuning hiperparametri).
* **data/test/** (15%): Folosit pentru evaluarea finală a performanței.

## 3. Descrierea Claselor (Etichete)
Sistemul clasifică imaginile în următoarele categorii:

| Etichetă (Folder) | Descriere Defect |
| :--- | :--- |
| **OK** | Piese printate corect, fără defecte vizibile (Clasa de control). |
| **Cracking** | Delaminare sau fisuri între straturi cauzate de răcirea neuniformă. |
| **Layer_shifting** | Deplasarea straturilor pe axa X sau Y (pierderea pașilor la motoare). |
| **Off_platform** | Desprinderea totală a piesei de pe patul de printare. |
| **Stringing** | Apariția firelor subțiri de plastic între zonele printate (retrapere incorectă). |
| **Warping** | Deformarea colțurilor bazei din cauza aderenței slabe sau răcirii rapide. |

## 4. Preprocesare Aplicată
Conform specificațiilor proiectului [PPT Slide 11]:
1.  **Redimensionare:** Toate imaginile au fost convertite la rezoluția **128x128 px**.
2.  **Format:** RGB (3 canale).
3.  **Augmentare (Planificat pentru Etapa 4):** Se va aplica doar pe setul de *Train*.