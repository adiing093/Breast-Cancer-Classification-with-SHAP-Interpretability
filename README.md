# Breast-Cancer-Classification-with-SHAP-Interpretability
This repository contains a machine learning pipeline for classifying breast cancer diagnoses using Random Forest, with a focus on model explainability using SHAP values.

## 🔍 Dataset
- **Source**: UCI ML Breast Cancer Wisconsin Diagnostic Dataset
- **Shape**: 569 samples, 33 features (including target)
- **Target Variable**: Diagnosis (`0` = Benign, `1` = Malignant)

## ⚙️ Tools & Libraries Used
- `Python`
- `scikit-learn`
- `SHAP`
- `matplotlib`, `seaborn`
- `Pandas`, `NumPy`

## 🧠 Model
- Random Forest Classifier
- Feature scaling using `StandardScaler`
- Evaluation using accuracy, classification report

## 📊 Explainability with SHAP
- Used SHAP’s TreeExplainer and new `Explainer()` API
- Visualized global and local feature importances
- **Top Influential Features**: 
  - `area_worst`
  - `concave points_worst`
  - `radius_worst`
  - `perimeter_worst`
  - `concavity_worst`
- Positive SHAP values → model leaning towards predicting **Malignant**
- Negative SHAP values → model leaning towards predicting **Benign**

## ✅ Results
- Accuracy: ~97%
- SHAP analysis helped highlight how individual features like `area_worst` strongly influenced malignant predictions.
- Fully interpretable model pipeline.

## 📁 Project Structure
- `notebook.ipynb` – contains preprocessing, modeling, SHAP analysis
- `images/` – contains exported plots
