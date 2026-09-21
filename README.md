# IBM HR Analytics — Employee Attrition Analysis

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Flask](https://img.shields.io/badge/Flask-3.1-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.63-red)
![Accuracy](https://img.shields.io/badge/Accuracy-88.06%25-brightgreen)
![ROCAUC](https://img.shields.io/badge/ROC--AUC-95.33%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Project Overview

End-to-end machine learning project predicting employee attrition using **Random Forest**, **Logistic Regression**, and **XGBoost** classifiers trained on the IBM HR Analytics dataset.

## Student

- Name: Mohammed Sameer Khazi
- Internship: AICTE | IBM SkillsBuild 2026
- Email: mohammedsameerkhaji@gmail.com

### Model Comparison Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **XGBoost** ✅ **Best** | **90.08%** | **89.29%** | **91.09%** | **90.18%** | **96.86%** |
| Random Forest (deployed) | 88.06% | 87.90% | 88.26% | 88.08% | 95.33% |
| Logistic Regression | 72.27% | 71.83% | 73.28% | 72.55% | 80.08% |

> **XGBoost** achieves the highest scores across all metrics. **Random Forest** is deployed as `model.pkl` due to its interpretability and stable feature importances.

---

## Dataset

**IBM HR Analytics Employee Attrition & Performance**  
- Rows: **1,470 employees**  
- Features: **35** (32 after dropping 3 constant columns)  
- Target: `Attrition` (Yes / No) — 16.12% positive rate  
- Link: [Kaggle Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

## Technologies

| Category         | Libraries / Tools                                      |
|------------------|-------------------------------------------------------|
| Data Processing  | Python 3.13, Pandas, NumPy                            |
| Visualisation    | Matplotlib, Seaborn, Plotly                           |
| Machine Learning | Scikit-learn (RandomForest, LabelEncoder), XGBoost, imbalanced-learn (SMOTE) |
| Model Storage    | Joblib                                                |
| Backend API      | Flask 3.1                                             |
| Frontend UI      | Streamlit 1.63                                        |
| Report           | python-docx                                           |
| AI Assistant     | IBM Bob                                               |

---

## Project Structure

```
HRAnalysis/
├── WA_Fn-UseC_-HR-Employee-Attrition.csv   # IBM HR dataset (1,470 rows)
├── MohammedSameerKhazi_HRAnalysis.ipynb    # Jupyter notebook — EDA + training
├── train_model.py                          # Standalone model training script
├── app.py                                  # Flask REST API (10 endpoints, port 5000)
├── ui.py                                   # Streamlit dashboard (5 pages + About page)
├── requirements.txt                        # All Python dependencies
├── README.md                               # This file
├── model.pkl                               # Saved Random Forest model (~5 MB)
├── api_outputs.txt                         # All 10 API endpoint JSON responses
├── collect_api_outputs.py                  # API test + screenshot generator
├── MohammedSameerKhazi_ProjectReport.docx  # Full project report
└── report_images/                          # 21 core dashboard + EDA screenshots
    ├── streamlit_home.png
    ├── streamlit_eda1.png
    ├── streamlit_eda2.png
    ├── streamlit_eda3.png
    ├── streamlit_predict_form.png
    ├── streamlit_predict_high.png
    ├── streamlit_predict_low.png
    ├── streamlit_model_metrics.png
    ├── streamlit_model_comparison.png
    ├── streamlit_risk_table.png
    ├── streamlit_risk_filtered.png
    ├── eda_dept_attrition.png
    ├── eda_jobrole_attrition.png
    ├── eda_overtime.png
    ├── eda_worklife.png
    ├── eda_age.png
    ├── eda_income.png
    ├── eda_heatmap.png
    ├── eda_feature_importance.png
    ├── eda_confusion_matrix.png
    └── eda_model_comparison.png
```

---

## Run Instructions

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Train the model (generates model.pkl + 6 EDA charts)
python train_model.py

# 3. Start Flask API (port 5000)
python app.py

# 4. Start Streamlit UI (port 8501)
#    NOTE: Use python -m streamlit on Windows Microsoft Store Python
python -m streamlit run ui.py

# 5. Open browser
# Streamlit: http://localhost:8501
# Flask API: http://localhost:5000
```

---

## API Endpoints

Base URL: `http://localhost:5000`

| Method | Endpoint              | Description                                     |
|--------|-----------------------|-------------------------------------------------|
| GET    | `/`                   | API status, author, model & dataset loaded flags |
| GET    | `/health`             | System health check with timestamp              |
| GET    | `/model-info`         | Accuracy, F1, ROC-AUC, classification report    |
| GET    | `/dataset-info`       | Shape, columns, attrition rate, stats           |
| GET    | `/insights`           | Top 5 data-driven insights                      |
| GET    | `/department-stats`   | Attrition breakdown by department               |
| GET    | `/risk-factors`       | Top 10 attrition causes (model feature importance) |
| POST   | `/predict`            | Single employee attrition risk prediction       |
| POST   | `/batch-predict`      | Bulk predictions (up to 100 employees)          |
| GET    | `/sample-predictions` | Predictions for first 5 dataset employees       |

### Example — POST /predict

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"Age": 28, "MonthlyIncome": 2500, "OverTime": 1, "JobSatisfaction": 2}'
```

Response:
```json
{
  "prediction": "Low Risk",
  "attrition_probability": 0.2902,
  "confidence_%": 70.98,
  "risk_level": "LOW"
}
```

---

## Key Insights

1. **16.12% overall attrition rate** — exceeds typical industry benchmarks
2. **Sales department** has the highest attrition rate (~20.6%)
3. **OverTime** is a major driver — overtime workers leave at 3x the rate
4. **StockOptionLevel** is a top-5 model predictor — equity retains employees
5. **Employees aged 25–35** have the highest attrition concentration
6. **MonthlyIncome** is the single most important feature (14.2% importance)
7. **Work-Life Balance score=1** employees leave at ~31% vs ~17% average
8. **Random Forest outperforms Logistic Regression** by 16 percentage points in ROC-AUC
9. **247 out of 1,470 employees** are predicted as high-risk by the deployed model
10. **Sales department** has ~40% high-risk employees — the highest of any department

---

## Streamlit UI Pages

| Page                | Description                                                          |
|---------------------|----------------------------------------------------------------------|
| Home                | KPI cards, department chart, and CSV export buttons for all employees and high-risk employees |
| EDA                 | 6 interactive Plotly charts with department filter                   |
| Predict Risk        | Employee profile sliders → Risk prediction + gauge chart             |
| Model Performance   | Metric cards, gauges, confusion matrix, model comparison chart/table |
| Employee Risk Table | All 1,470 employees with Risk Level, filters, CSV download           |
| About               | Student profile, internship details, tech stack, project summary, and model performance |

---

## Submission Files

| File | Description |
|------|-------------|
| `MohammedSameerKhazi_HRAnalysis.ipynb` | Jupyter notebook with full EDA and model training |
| `requirements.txt` | All Python library dependencies |
| `MohammedSameerKhazi_ProjectReport.docx` | Full 13-section project report with charts |
| `README.md` | This documentation file |

