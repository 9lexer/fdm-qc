# 🧠 Modul 2: Rețea Neuronală (Arhitectură CNN)

Acest modul este responsabil pentru definirea, compilarea și salvarea arhitecturii de Rețea Neuronală Convoluțională (CNN) utilizată în proiectul de detecție a defectelor FDM.

În **Etapa 4**, acest script generează un model "schelet" (neantrenat), cu greutăți inițializate aleatoriu, pentru a demonstra funcționalitatea pipeline-ului software.

---

## 🏗️ Detalii Arhitectură Curentă

Modelul este construit folosind biblioteca **TensorFlow / Keras** și are o arhitectură secvențială optimizată pentru clasificarea imaginilor de rezoluție mică (128x128px).

### Specificații Tehnice:
* **Input:** Imagini RGB de dimensiunea `(128, 128, 3)`.
* **Tip Model:** CNN Secvențial (Sequential API).
* **Număr Clase:** 6 (OK + 5 Tipuri de Defecte).
* **Funcție de Pierdere (Loss):** `sparse_categorical_crossentropy`.
* **Optimizator:** `adam`.

### Structura Straturilor (Layers):

1.  **Input Layer:** `(128, 128, 3)`
    * Preia imaginea brută normalizată.
2.  **Conv2D Block 1:**
    * 32 filtre, kernel 3x3, activare `ReLU`.
    * **MaxPooling2D:** 2x2 (reduce dimensiunea spațială la jumătate).
3.  **Conv2D Block 2:**
    * 64 filtre, kernel 3x3, activare `ReLU`.
    * **MaxPooling2D:** 2x2.
4.  **Flatten Layer:**
    * Transformă matricea 2D într-un vector 1D pentru straturile dense.
5.  **Dense Layer (Hidden):**
    * 64 neuroni, activare `ReLU` (pentru învățarea trăsăturilor complexe).
6.  **Output Layer:**
    * 6 neuroni, activare `Softmax`.
    * Returnează un vector de probabilități (suma = 100%) pentru fiecare clasă.

---

## 🚀 Utilizare

Pentru a genera (sau regenera) fișierul modelului, rulați scriptul `model.py` din rădăcina proiectului:

```bash
python src/neural_network/model.py