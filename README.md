# Project 2 - NLP: Fake News Detection

In this project, we were tasked with predicting fake news articles using natural language processing techniques.

We trained models on a pre-labeled dataset and made predictions on an unlabeled dataset. At the end, we evaluated our models by comparing predictions with the true labels.

---

## 🔍 Best Results

Our best-performing model used **Logistic Regression** with **TimeSeriesSplit**:

- **Accuracy:** 0.9864  
- **F1 Score:** 0.975  
- **Precision:** 0.996  
- **Recall:** 0.990  

---

## 📂 Repository Contents

| File Name | Description |
|-----------|-------------|
| `accuracy_calculation.ipynb` | Used to evaluate predictions against the real labels |
| `Model_training.ipynb` | Notebook with our model training results |
| `model_training.py` | Python script containing our initial basic models |
| `model:training_final.py` | Python script containing our final models |
| `modelLR.pkl` | Serialized version of our best-performing Logistic Regression model |
| `NLP Project 2 - Presentation Model.pptx` | Short presentation of our model and results |
| `predict.ipynb` | Notebook used to make predictions on the unlabeled dataset |
| `predict_trans_googlecollab.ipynb` | Notebook used for predictions with a DistilBERT model (Google Colab) |
| `predictionsLR.csv` | Predictions using Logistic Regression |
| `predictionsRF.csv` | Predictions using Random Forest |
| `predictionsLR_transformers_with_original_data.csv` | Predictions using Logistic Regression with transformer-transformed data |
| `Project_nlp_transformer.ipynb` | Notebook exploring a pretrained DistilBERT model |

---

**By:** Group 2 — *Alex & Daan*
