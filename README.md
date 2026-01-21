# House Pricing (Machine Learning)

Machine Learning project to estimate real estate prices from structured and semi-structured features.  
The goal is to build a robust **data preprocessing** pipeline and compare different models to identify the solution with the best performance.

---

## 📌 Project contents

- **Main notebook**: includes
  - complete preprocessing (cleaning + feature engineering)
  - creation of `*_clean.csv` datasets
  - model training and comparison (Random Forest, Linear Regression, Lasso, Gradient Boosting)
  - evaluation on a holdout test set
- **Dataset**
  - `train.csv`, `test.csv` (raw)
  - `train_clean.csv`, `test_clean.csv` / or `X_clean.csv`, `y_clean.csv` (processed)
- **Output**
  - final prediction files (e.g., `predicted_values_*.txt`)

---

## 🧠 Approach

### 1) Data Preprocessing / Feature Engineering

The dataset contains numerical, categorical, and semi-structured (text-like) variables.  
Preprocessing was designed to make the pipeline reproducible and reduce errors due to non-uniform formats.

Main transformations:

**Parsing and normalization of semi-structured features**
- `car_parking`: extract the number of parking spots, distinguishing (e.g.) garage vs shared parking and adding a missing flag.
- `availability`: handle values such as `available` or `available from dd/mm/yyyy`, transforming them into indicators and time components (year/month).
- `condominium_fees`: convert strings to numeric values, handling special cases (e.g., *no fees*).
- `floor` / `total_floors_in_building`: numeric parsing and special cases (e.g., *ground floor*, *mezzanine*).
- `energy_efficiency_class`: map to an ordinal scale and add a missing flag.
- `year_of_construction`: numeric conversion + derived feature `building_age`.

**“Simple but effective” feature engineering**
- `is_top_floor`, `floor_ratio`
- `num_balconies` extracted from `other_features` (e.g., `"2 balconies"`)

**Handling multi-label features (`other_features`)**
- split on the `|` separator + fixes for known concatenation issues
- conversion to multi-hot encoding via `MultiLabelBinarizer`

**Categorical encoding**
- One-Hot Encoding for `conditions` and `zone`
- `handle_unknown="ignore"` to manage categories that appear only in the test set

**Missing value imputation**
- imputation using statistics computed on the training set (e.g., medians) to avoid leakage.

Result: creation of final **clean, numeric** datasets:
- `train_clean.csv` / `test_clean.csv` (or `X_clean.csv` / `y_clean.csv`)

---

### 2) Model training and comparison

After preprocessing, the data is split into:
- **Train**
- **Holdout test** (for a fair comparison across models)

Trained and compared models:
- **Linear Regression (OLS)** – linear baseline
- **Lasso (LassoCV)** – regularization + implicit feature selection
- **Random Forest Regressor** – non-linear model, robust to heterogeneous features
- **Gradient Boosting Regressor** – boosting to improve accuracy

For linear models, **StandardScaler** is used (within a pipeline) to ensure consistent scaling.

Metrics used on the test set:
- **RMSE** (root mean squared error)
- **R²**

Finally, the model with the lowest RMSE is selected as the “best” candidate.

---

## ✅ Results

The notebook generates a table with RMSE and R² on the holdout test set for each model, along with comparison plots.  
The best model is automatically identified and (optionally) used to generate final predictions on the test dataset.

---

## ▶️ How to run

### Requirements
Python 3.9+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
