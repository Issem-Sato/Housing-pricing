# House Pricing (Machine Learning)

Progetto di Machine Learning per stimare il prezzo di immobili a partire da feature strutturate e semi-strutturate.  
L’obiettivo è costruire una pipeline di **data preprocessing** robusta e confrontare diversi modelli per individuare la soluzione con le migliori performance.

---

## 📌 Contenuto del progetto

- **Notebook principale**: contiene
  - preprocessing completo (pulizia + feature engineering)
  - creazione dei dataset `*_clean.csv`
  - training e confronto modelli (Random Forest, Linear Regression, Lasso, Gradient Boosting)
  - valutazione su holdout-test
- **Dataset**
  - `train.csv`, `test.csv` (raw)
  - `train_clean.csv`, `test_clean.csv` / oppure `X_clean.csv`, `y_clean.csv` (processed)
- **Output**
  - file di predizioni finali (es. `predicted_values_*.txt`)

---

## 🧠 Approccio

### 1) Data Preprocessing / Feature Engineering

Il dataset contiene variabili numeriche, categoriche e semi-strutturate (testuali).  
Il preprocessing è stato sviluppato per rendere la pipeline riproducibile e ridurre errori dovuti a formati non uniformi.

Principali trasformazioni:

**Parsing e normalizzazione feature semi-strutturate**
- `car_parking`: estrazione del numero di posti auto, distinguendo (es.) garage vs shared parking e flag di missing.
- `availability`: gestione di valori come `available` o `available from dd/mm/yyyy`, trasformati in indicatori e componenti temporali (anno/mese).
- `condominium_fees`: conversione stringhe in numerico, gestione casi speciali (es. *no fees*).
- `floor` / `total_floors_in_building`: parsing numerico e casi speciali (es. *ground floor*, *mezzanine*).
- `energy_efficiency_class`: mapping in scala ordinale e flag missing.
- `year_of_construction`: conversione numerica + feature derivata `building_age`.

**Feature engineering “semplice ma efficace”**
- `is_top_floor`, `floor_ratio`
- `num_balconies` estratto da `other_features` (es. `"2 balconies"`)

**Gestione delle feature multi-label (`other_features`)**
- split su separatore `|` + correzioni di concatenazioni note
- conversione in multi-hot encoding tramite `MultiLabelBinarizer`

**Encoding categorico**
- One-Hot Encoding per `conditions` e `zone`
- `handle_unknown="ignore"` per gestire categorie presenti solo nel test

**Imputazione valori mancanti**
- imputazione con statistiche calcolate sul train (es. mediane) per evitare leakage.

Risultato: creazione di dataset finali **puliti e numerici**:
- `train_clean.csv` / `test_clean.csv` (oppure `X_clean.csv` / `y_clean.csv`)

---

### 2) Training e confronto modelli

Dopo il preprocessing, i dati vengono splittati in:
- **Train**
- **Holdout Test** (per confronto equo tra modelli)

Modelli allenati e confrontati:
- **Linear Regression (OLS)** – baseline lineare
- **Lasso (LassoCV)** – regolarizzazione + feature selection implicita
- **Random Forest Regressor** – modello non lineare, robusto a feature eterogenee
- **Gradient Boosting Regressor** – boosting per migliorare accuratezza

Per i modelli lineari viene utilizzato **StandardScaler** (in pipeline) per garantire scaling coerente.

Metriche utilizzate sul test:
- **RMSE** (root mean squared error)
- **R²**

Alla fine viene selezionato il modello con RMSE più basso come candidato “migliore”.

---

## ✅ Risultati

Nel notebook viene generata una tabella con RMSE e R² sul holdout-test per ogni modello, oltre a grafici comparativi.  
Il modello migliore viene identificato automaticamente e (opzionalmente) usato per generare predizioni finali sul dataset di test.

---

## ▶️ Come eseguire

### Requisiti
Python 3.9+ consigliato.

Installa le dipendenze:

```bash
pip install -r requirements.txt
