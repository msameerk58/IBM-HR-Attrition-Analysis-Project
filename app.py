"""
Flask Backend API — HR Employee Attrition Predictor (Enhanced)
Author : Mohammed Sameer Khazi
Course : AICTE IBM SkillsBuild Internship 2026
Port   : 5000

Endpoints:
  GET  /                    service status
  GET  /health              timestamp + model/data status
  GET  /model-info          metrics (accuracy, ROC-AUC, report)
  GET  /dataset-info        dataset statistics
  POST /predict             single-employee prediction
  POST /batch-predict       batch predictions (list of employees)
  GET  /sample-predictions  first 5 employees predictions
  GET  /insights            top 5 dataset insights
  GET  /department-stats    attrition breakdown by department
  GET  /risk-factors        top 10 attrition risk factors
"""

import os
import sys
import io
import logging
import traceback
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("hr-api")

app = Flask(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE, "model.pkl")
DATASET_PATH = os.path.join(BASE, "WA_Fn-UseC_-HR-Employee-Attrition.csv")
RAW_CSV_PATH = DATASET_PATH   # original CSV kept for analytics routes

model        = None
df_raw       = None           # label-encoded, used for ML
df_orig      = None           # raw string-label version, used for analytics
FEATURE_COLS = []
STARTUP_TIME = datetime.now(timezone.utc).isoformat()

# ── Request logging middleware ────────────────────────────────────────────────
@app.before_request
def _log_request():
    log.info("REQ  %s %s  body_bytes=%s",
             request.method, request.path,
             request.content_length or 0)

@app.after_request
def _log_response(response):
    log.info("RESP %s %s  status=%s",
             request.method, request.path, response.status_code)
    return response

# ── Load artefacts at startup ─────────────────────────────────────────────────
def load_artefacts():
    global model, df_raw, df_orig, FEATURE_COLS

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        log.info("model loaded -> %s  (%d estimators)", MODEL_PATH, model.n_estimators)
    else:
        log.warning("model.pkl not found at %s", MODEL_PATH)

    if os.path.exists(DATASET_PATH):
        df_orig = pd.read_csv(DATASET_PATH)          # keep original strings
        df_raw  = df_orig.copy()
        drop    = ["EmployeeCount", "StandardHours", "Over18"]
        df_raw.drop(columns=[c for c in drop if c in df_raw.columns], inplace=True)
        df_orig.drop(columns=[c for c in drop if c in df_orig.columns], inplace=True)

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in df_raw.select_dtypes(include="object").columns:
            if col != "Attrition":
                df_raw[col] = le.fit_transform(df_raw[col])
        df_raw["Attrition"] = (df_raw["Attrition"] == "Yes").astype(int)

        FEATURE_COLS = [c for c in df_raw.columns if c != "Attrition"]
        log.info("dataset loaded -> %d rows, %d features", df_raw.shape[0], len(FEATURE_COLS))
    else:
        log.warning("dataset not found at %s", DATASET_PATH)

load_artefacts()

# ── Guard helpers ─────────────────────────────────────────────────────────────
def _model_ready(): return model is not None
def _data_ready():  return df_raw is not None

def _require_model_and_data():
    if not _model_ready() or not _data_ready():
        return jsonify({"error": "Model or dataset not loaded — run train_model.py first"}), 503
    return None

def _ts():
    return datetime.now(timezone.utc).isoformat()


# ════════════════════════════════════════════════════════════════════════════════
# Route 1 — GET /
# ════════════════════════════════════════════════════════════════════════════════
@app.route("/", methods=["GET"])
def status():
    return jsonify({
        "status"        : "running",
        "service"       : "HR Attrition Prediction API",
        "author"        : "Mohammed Sameer Khazi",
        "course"        : "AICTE IBM SkillsBuild Internship 2026",
        "model_loaded"  : _model_ready(),
        "dataset_loaded": _data_ready(),
        "timestamp"     : _ts(),
        "endpoints"     : [
            "GET  /",
            "GET  /health",
            "GET  /model-info",
            "GET  /dataset-info",
            "POST /predict",
            "POST /batch-predict",
            "GET  /sample-predictions",
            "GET  /insights",
            "GET  /department-stats",
            "GET  /risk-factors",
        ]
    })


# ════════════════════════════════════════════════════════════════════════════════
# Route 2 — GET /health
# ════════════════════════════════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    try:
        model_status = "loaded" if _model_ready() else "not_loaded"
        data_status  = "loaded" if _data_ready()  else "not_loaded"
        overall      = "healthy" if (_model_ready() and _data_ready()) else "degraded"
        return jsonify({
            "status"          : overall,
            "timestamp"       : _ts(),
            "startup_time"    : STARTUP_TIME,
            "model_status"    : model_status,
            "dataset_status"  : data_status,
            "model_type"      : type(model).__name__ if _model_ready() else None,
            "n_estimators"    : int(model.n_estimators) if _model_ready() else None,
            "dataset_rows"    : int(df_raw.shape[0]) if _data_ready() else None,
            "feature_count"   : len(FEATURE_COLS) if FEATURE_COLS else None,
        }), 200
    except Exception as e:
        log.error("health error: %s", traceback.format_exc())
        return jsonify({"status": "error", "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════════════
# Route 3 — GET /model-info
# ════════════════════════════════════════════════════════════════════════════════
@app.route("/model-info", methods=["GET"])
def model_info():
    guard = _require_model_and_data()
    if guard: return guard

    try:
        from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
        from imblearn.over_sampling import SMOTE
        from sklearn.model_selection import train_test_split

        X = df_raw[FEATURE_COLS]
        y = df_raw["Attrition"]
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        _, X_test, _, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        report  = classification_report(y_test, y_pred,
                                        target_names=["No Attrition", "Attrition"],
                                        output_dict=True)
        return jsonify({
            "model_type"            : type(model).__name__,
            "n_estimators"          : int(model.n_estimators),
            "max_depth"             : model.max_depth,
            "n_features"            : len(FEATURE_COLS),
            "feature_names"         : FEATURE_COLS,
            "accuracy"              : round(accuracy_score(y_test, y_pred), 4),
            "roc_auc"               : round(roc_auc_score(y_test, y_proba), 4),
            "classification_report" : report,
            "timestamp"             : _ts(),
        })
    except Exception as e:
        log.error("model-info error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════════════
# Route 4 — GET /dataset-info
# ════════════════════════════════════════════════════════════════════════════════
@app.route("/dataset-info", methods=["GET"])
def dataset_info():
    guard = _require_model_and_data()
    if guard: return guard

    try:
        attrition_counts = df_raw["Attrition"].value_counts().to_dict()
        return jsonify({
            "total_rows"       : int(df_raw.shape[0]),
            "total_features"   : int(df_raw.shape[1]),
            "feature_names"    : FEATURE_COLS,
            "attrition_counts" : {str(k): int(v) for k, v in attrition_counts.items()},
            "attrition_rate_%" : round(df_raw["Attrition"].mean() * 100, 2),
            "numeric_features" : int(df_raw[FEATURE_COLS].select_dtypes(include="number").shape[1]),
            "missing_values"   : int(df_raw.isnull().sum().sum()),
            "age_stats"        : {
                "min"  : int(df_raw["Age"].min()),
                "max"  : int(df_raw["Age"].max()),
                "mean" : round(float(df_raw["Age"].mean()), 1)
            },
            "income_stats"     : {
                "min"  : int(df_raw["MonthlyIncome"].min()),
                "max"  : int(df_raw["MonthlyIncome"].max()),
                "mean" : round(float(df_raw["MonthlyIncome"].mean()), 1)
            },
            "timestamp"        : _ts(),
        })
    except Exception as e:
        log.error("dataset-info error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════════════
# Route 5 — POST /predict
# ════════════════════════════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    guard = _require_model_and_data()
    if guard: return guard

    data = request.get_json(force=True, silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object with employee features"}), 400

    try:
        row     = {col: float(data.get(col, df_raw[col].median())) for col in FEATURE_COLS}
        X_input = pd.DataFrame([row])[FEATURE_COLS]
        proba   = float(model.predict_proba(X_input)[0][1])
        label   = "High Risk" if proba >= 0.5 else "Low Risk"
        return jsonify({
            "prediction"            : label,
            "attrition_probability" : round(proba, 4),
            "confidence_%"          : round(proba * 100, 2) if proba >= 0.5
                                      else round((1 - proba) * 100, 2),
            "risk_level"            : "HIGH" if proba >= 0.65 else
                                      "MEDIUM" if proba >= 0.35 else "LOW",
            "input_features"        : row,
            "timestamp"             : _ts(),
        })
    except Exception as e:
        log.error("predict error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 400


# ════════════════════════════════════════════════════════════════════════════════
# Route 6 — POST /batch-predict
# ════════════════════════════════════════════════════════════════════════════════
@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    guard = _require_model_and_data()
    if guard: return guard

    data = request.get_json(force=True, silent=True)
    if not data or not isinstance(data, list):
        return jsonify({
            "error": "Request body must be a JSON array of employee objects",
            "example": [{"Age": 30, "MonthlyIncome": 3000, "OverTime": 1}]
        }), 400

    if len(data) > 100:
        return jsonify({"error": "Batch size limit is 100 employees per request"}), 400

    try:
        results = []
        for idx, emp in enumerate(data):
            if not isinstance(emp, dict):
                results.append({"employee_index": idx + 1,
                                 "error": "Employee entry must be a JSON object"})
                continue
            row     = {col: float(emp.get(col, df_raw[col].median())) for col in FEATURE_COLS}
            X_input = pd.DataFrame([row])[FEATURE_COLS]
            proba   = float(model.predict_proba(X_input)[0][1])
            label   = "High Risk" if proba >= 0.5 else "Low Risk"
            results.append({
                "employee_index"        : idx + 1,
                "prediction"            : label,
                "attrition_probability" : round(proba, 4),
                "risk_level"            : "HIGH" if proba >= 0.65 else
                                          "MEDIUM" if proba >= 0.35 else "LOW",
            })

        high_risk  = sum(1 for r in results if r.get("prediction") == "High Risk")
        low_risk   = len(results) - high_risk
        return jsonify({
            "batch_size"       : len(data),
            "high_risk_count"  : high_risk,
            "low_risk_count"   : low_risk,
            "high_risk_rate_%" : round(high_risk / len(data) * 100, 1),
            "predictions"      : results,
            "timestamp"        : _ts(),
        })
    except Exception as e:
        log.error("batch-predict error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 400


# ════════════════════════════════════════════════════════════════════════════════
# Route 7 — GET /sample-predictions
# ════════════════════════════════════════════════════════════════════════════════
@app.route("/sample-predictions", methods=["GET"])
def sample_predictions():
    guard = _require_model_and_data()
    if guard: return guard

    try:
        n      = min(int(request.args.get("n", 5)), 20)
        sample = df_raw[FEATURE_COLS].head(n)
        probas = model.predict_proba(sample)[:, 1]
        preds  = model.predict(sample)

        results = []
        for i, (prob, pred) in enumerate(zip(probas, preds)):
            results.append({
                "employee_index"        : i + 1,
                "prediction"            : "High Risk" if pred == 1 else "Low Risk",
                "attrition_probability" : round(float(prob), 4),
                "risk_level"            : "HIGH" if prob >= 0.65 else
                                          "MEDIUM" if prob >= 0.35 else "LOW",
                "actual_attrition"      : int(df_raw["Attrition"].iloc[i]),
                "correct_prediction"    : bool((pred == 1) == bool(df_raw["Attrition"].iloc[i])),
            })
        return jsonify({"sample_count": n, "sample_predictions": results, "timestamp": _ts()})
    except Exception as e:
        log.error("sample-predictions error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════════════
# Route 8 — GET /insights
# ════════════════════════════════════════════════════════════════════════════════
@app.route("/insights", methods=["GET"])
def insights():
    guard = _require_model_and_data()
    if guard: return guard

    try:
        d = df_orig   # use human-readable labels

        overall_rate  = round(d["Attrition"].eq("Yes").mean() * 100, 1)
        ot_yes_rate   = round(d[d["OverTime"]=="Yes"]["Attrition"].eq("Yes").mean() * 100, 1)
        ot_no_rate    = round(d[d["OverTime"]=="No"]["Attrition"].eq("Yes").mean() * 100, 1)
        ot_mult       = round(ot_yes_rate / ot_no_rate, 1) if ot_no_rate else None

        avg_inc_left  = round(d[d["Attrition"]=="Yes"]["MonthlyIncome"].mean(), 0)
        avg_inc_stay  = round(d[d["Attrition"]=="No"]["MonthlyIncome"].mean(), 0)

        wlb1_rate     = round(d[d["WorkLifeBalance"]==1]["Attrition"].eq("Yes").mean() * 100, 1)

        top_role      = (
            d.groupby("JobRole")["Attrition"]
            .apply(lambda x: round(x.eq("Yes").mean() * 100, 1))
            .idxmax()
        )
        top_role_rate = round(
            d[d["JobRole"]==top_role]["Attrition"].eq("Yes").mean() * 100, 1
        )

        single_rate   = round(d[d["MaritalStatus"]=="Single"]["Attrition"].eq("Yes").mean() * 100, 1)
        married_rate  = round(d[d["MaritalStatus"]=="Married"]["Attrition"].eq("Yes").mean() * 100, 1)

        return jsonify({
            "insights": [
                {
                    "rank"   : 1,
                    "title"  : "Overall Attrition Rate",
                    "value"  : f"{overall_rate}%",
                    "detail" : f"{overall_rate}% of employees left — significant HR concern requiring proactive retention strategies.",
                },
                {
                    "rank"   : 2,
                    "title"  : "OverTime is the Strongest Behavioural Driver",
                    "value"  : f"{ot_yes_rate}% vs {ot_no_rate}%",
                    "detail" : f"Overtime workers leave at {ot_yes_rate}% vs {ot_no_rate}% for non-overtime — {ot_mult}x higher risk.",
                },
                {
                    "rank"   : 3,
                    "title"  : "Income Gap Between Leavers and Stayers",
                    "value"  : f"${int(avg_inc_left):,} vs ${int(avg_inc_stay):,}",
                    "detail" : f"Employees who left earned ${int(avg_inc_left):,}/month on avg vs ${int(avg_inc_stay):,} for those who stayed.",
                },
                {
                    "rank"   : 4,
                    "title"  : "Poor Work-Life Balance Doubles Attrition",
                    "value"  : f"{wlb1_rate}% (score=1)",
                    "detail" : f"Employees with the lowest work-life balance score (1) have a {wlb1_rate}% attrition rate.",
                },
                {
                    "rank"   : 5,
                    "title"  : f"Highest-Risk Role: {top_role}",
                    "value"  : f"{top_role_rate}%",
                    "detail" : f"{top_role} has the highest attrition rate at {top_role_rate}%. Single employees leave at {single_rate}% vs {married_rate}% for married.",
                },
            ],
            "timestamp": _ts(),
        })
    except Exception as e:
        log.error("insights error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════════════
# Route 9 — GET /department-stats
# ════════════════════════════════════════════════════════════════════════════════
@app.route("/department-stats", methods=["GET"])
def department_stats():
    guard = _require_model_and_data()
    if guard: return guard

    try:
        d = df_orig
        stats = []
        for dept, grp in d.groupby("Department"):
            total        = len(grp)
            left         = grp["Attrition"].eq("Yes").sum()
            attr_rate    = round(left / total * 100, 1)
            avg_income   = round(grp["MonthlyIncome"].mean(), 0)
            avg_age      = round(grp["Age"].mean(), 1)
            ot_pct       = round(grp["OverTime"].eq("Yes").mean() * 100, 1)
            avg_sat      = round(grp["JobSatisfaction"].mean(), 2)
            stats.append({
                "department"         : dept,
                "total_employees"    : int(total),
                "employees_left"     : int(left),
                "employees_stayed"   : int(total - left),
                "attrition_rate_%"   : attr_rate,
                "avg_monthly_income" : int(avg_income),
                "avg_age"            : avg_age,
                "overtime_pct_%"     : ot_pct,
                "avg_job_satisfaction": avg_sat,
            })
        stats.sort(key=lambda x: x["attrition_rate_%"], reverse=True)

        return jsonify({
            "department_count" : len(stats),
            "highest_attrition": stats[0]["department"] if stats else None,
            "lowest_attrition" : stats[-1]["department"] if stats else None,
            "departments"      : stats,
            "timestamp"        : _ts(),
        })
    except Exception as e:
        log.error("department-stats error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════════════
# Route 10 — GET /risk-factors
# ════════════════════════════════════════════════════════════════════════════════
@app.route("/risk-factors", methods=["GET"])
def risk_factors():
    guard = _require_model_and_data()
    if guard: return guard

    try:
        # Use model feature importances as the primary ranking
        importances = pd.Series(
            model.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False)

        # Human-readable descriptions for each factor
        descriptions = {
            "MonthlyIncome"         : "Lower monthly income significantly increases likelihood of leaving.",
            "OverTime"              : "Working overtime is the strongest behavioural predictor of attrition.",
            "Age"                   : "Younger employees (18–30) show the highest attrition rates.",
            "TotalWorkingYears"     : "Less experienced employees with fewer total working years leave more.",
            "StockOptionLevel"      : "Employees with no stock options (level 0) leave at higher rates.",
            "YearsAtCompany"        : "Short-tenure employees (0–2 years) are the highest-risk group.",
            "JobLevel"              : "Entry-level (level 1) employees have significantly higher attrition.",
            "YearsInCurrentRole"    : "Employees stuck in the same role too long — or too short — tend to leave.",
            "DistanceFromHome"      : "Long commute distances correlate with increased attrition probability.",
            "WorkLifeBalance"       : "Poor work-life balance (score=1) doubles the attrition rate.",
            "JobSatisfaction"       : "Low job satisfaction (score=1) is a major early-warning indicator.",
            "EnvironmentSatisfaction": "Dissatisfaction with the work environment drives voluntary exits.",
            "JobInvolvement"        : "Low job involvement predicts disengagement and eventual departure.",
            "NumCompaniesWorked"    : "Job-hoppers (worked at many companies) are more prone to leaving again.",
            "MaritalStatus"         : "Single employees leave at nearly 2x the rate of married colleagues.",
        }

        top10 = importances.head(10)
        factors = []
        for rank, (feat, score) in enumerate(top10.items(), 1):
            factors.append({
                "rank"       : rank,
                "feature"    : feat,
                "importance" : round(float(score), 4),
                "importance_%": round(float(score) * 100, 2),
                "description": descriptions.get(feat, f"{feat} is a significant predictor of attrition."),
            })

        return jsonify({
            "top_risk_factors" : factors,
            "model_type"       : type(model).__name__,
            "total_features"   : len(FEATURE_COLS),
            "timestamp"        : _ts(),
        })
    except Exception as e:
        log.error("risk-factors error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ── 404 handler ───────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error"    : "Endpoint not found",
        "available": [
            "GET  /", "GET  /health", "GET  /model-info",
            "GET  /dataset-info", "POST /predict", "POST /batch-predict",
            "GET  /sample-predictions", "GET  /insights",
            "GET  /department-stats", "GET  /risk-factors",
        ]
    }), 404

# ── 405 handler ───────────────────────────────────────────────────────────────
@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": f"Method {request.method} not allowed on {request.path}"}), 405

# ── 500 handler ───────────────────────────────────────────────────────────────
@app.errorhandler(500)
def internal_error(e):
    log.error("500 on %s: %s", request.path, traceback.format_exc())
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
