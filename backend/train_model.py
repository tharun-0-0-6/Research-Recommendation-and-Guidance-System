"""
Research Recommendation Platform — ML Model Training
======================================================
This script:
  1. Loads basic_training_dataset.csv + roadmap_training_dataset.csv
  2. Builds TF-IDF + categorical feature matrix
  3. Trains KNN (nearest-neighbour retrieval) for similar project lookup
  4. Trains multi-label classifiers for equipment, methodology, and tool prediction
  5. Serialises all models and encoders to /models/ directory
  6. Runs a full test evaluation and prints metrics

Usage:
    pip install pandas scikit-learn numpy joblib
    python train_model.py
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix

import joblib
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
from pathlib import Path
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "datasets"

BASIC_CSV   = str(DATA_DIR / "basic_training_dataset.csv")
ROADMAP_CSV = str(DATA_DIR / "roadmap_training_dataset.csv")
MODEL_DIR   = str(ROOT.parent / "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TOP_K_EQUIPMENT   = 8
TOP_K_METHODOLOGY = 5
TOP_K_TOOLS       = 6
KNN_NEIGHBORS     = 10

# ──────────────────────────────────────────────
# 1. LOAD & AGGREGATE DATA
# ──────────────────────────────────────────────
print("\n📥 Loading datasets...")
basic_df   = pd.read_csv(BASIC_CSV)
roadmap_df = pd.read_csv(ROADMAP_CSV)
print(f"   Basic dataset : {basic_df.shape[0]:,} rows, {basic_df.shape[1]} columns")
print(f"   Roadmap dataset: {roadmap_df.shape[0]:,} rows, {roadmap_df.shape[1]} columns")

# Aggregate per project_id — collect lists of equipment, methodologies, tools
print("\n🔧 Aggregating per-project labels...")

def agg_str_list(series):
    return list(set([v.strip() for v in series.dropna() if v.strip()]))

project_meta = basic_df.groupby("project_id").first().reset_index()[[
    "project_id", "project_title", "project_description",
    "research_domain", "project_type", "budget_range",
    "duration_months", "team_expertise", "location"
]]

equipment_labels   = basic_df.groupby("project_id")["equipment_name"].apply(agg_str_list).reset_index()
equipment_labels.columns = ["project_id", "equipment_list"]

methodology_labels = basic_df.groupby("project_id")["methodology_name"].apply(agg_str_list).reset_index()
methodology_labels.columns = ["project_id", "methodology_list"]

tool_labels        = basic_df.groupby("project_id")["tool_name"].apply(agg_str_list).reset_index()
tool_labels.columns = ["project_id", "tool_list"]

df = project_meta \
    .merge(equipment_labels, on="project_id") \
    .merge(methodology_labels, on="project_id") \
    .merge(tool_labels, on="project_id")

# Remove empty label rows
df = df[df["equipment_list"].map(len) > 0].reset_index(drop=True)
print(f"   Aggregated: {len(df)} unique projects")

# ──────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────
print("\n⚙️  Building feature matrix...")

# Text features
text_combined = (
    df["project_title"].fillna("") + " " +
    df["project_description"].fillna("") + " " +
    df["team_expertise"].fillna("") + " " +
    df["research_domain"].fillna("")
)

tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True,
    min_df=2,
)
X_text = tfidf.fit_transform(text_combined)
print(f"   TF-IDF matrix: {X_text.shape}")

# Categorical features
cat_cols = ["research_domain", "project_type", "budget_range"]
ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
X_cat = ohe.fit_transform(df[cat_cols].fillna("Unknown"))
print(f"   Categorical OHE matrix: {X_cat.shape}")

# Numeric features
num_cols = ["duration_months"]
df["duration_months"] = pd.to_numeric(df["duration_months"], errors="coerce").fillna(6)
scaler = StandardScaler()
X_num = csr_matrix(scaler.fit_transform(df[num_cols]))
print(f"   Numeric matrix: {X_num.shape}")

# Combined
X = hstack([X_text, X_cat, X_num])
print(f"   Combined feature matrix: {X.shape}")

# ──────────────────────────────────────────────
# 3. LABEL BINARISERS
# ──────────────────────────────────────────────
mlb_equip  = MultiLabelBinarizer()
mlb_meth   = MultiLabelBinarizer()
mlb_tool   = MultiLabelBinarizer()

y_equip    = mlb_equip.fit_transform(df["equipment_list"])
y_meth     = mlb_meth.fit_transform(df["methodology_list"])
y_tool     = mlb_tool.fit_transform(df["tool_list"])

print(f"\n   Equipment  classes : {len(mlb_equip.classes_)}")
print(f"   Methodology classes: {len(mlb_meth.classes_)}")
print(f"   Tool classes       : {len(mlb_tool.classes_)}")

# ──────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)

X_train, X_test = X[train_idx], X[test_idx]
y_equip_train,  y_equip_test  = y_equip[train_idx],  y_equip[test_idx]
y_meth_train,   y_meth_test   = y_meth[train_idx],   y_meth[test_idx]
y_tool_train,   y_tool_test   = y_tool[train_idx],   y_tool[test_idx]

print(f"\n   Train size: {len(train_idx)}, Test size: {len(test_idx)}")

# ──────────────────────────────────────────────
# 5. TRAIN MULTI-LABEL CLASSIFIERS
# ──────────────────────────────────────────────

def train_multilabel(X_tr, y_tr, name):
    print(f"\n🏋️  Training {name} classifier...")
    clf = OneVsRestClassifier(
        LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", class_weight="balanced"),
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)
    print(f"   ✓ {name} classifier trained")
    return clf

clf_equip = train_multilabel(X_train, y_equip_train, "Equipment")
clf_meth  = train_multilabel(X_train, y_meth_train,  "Methodology")
clf_tool  = train_multilabel(X_train, y_tool_train,  "Tool")

# ──────────────────────────────────────────────
# 6. KNN FOR SIMILAR PROJECT RETRIEVAL
# ──────────────────────────────────────────────
print("\n🔍 Training KNN retrieval model...")
knn = NearestNeighbors(n_neighbors=KNN_NEIGHBORS + 1, metric="cosine", algorithm="brute", n_jobs=-1)
knn.fit(X)
print(f"   ✓ KNN trained on {X.shape[0]} projects (k={KNN_NEIGHBORS})")

# ──────────────────────────────────────────────
# 7. EVALUATION
# ──────────────────────────────────────────────

def evaluate(clf, X_te, y_te, mlb, name, top_k):
    proba = clf.predict_proba(X_te)
    # Top-K threshold
    preds = np.zeros_like(proba, dtype=int)
    for i, row in enumerate(proba):
        topk = np.argsort(row)[-top_k:]
        preds[i, topk] = 1

    f1_micro = f1_score(y_te, preds, average="micro", zero_division=0)
    f1_macro = f1_score(y_te, preds, average="macro", zero_division=0)
    print(f"\n   [{name}]  F1-micro={f1_micro:.3f}  F1-macro={f1_macro:.3f}")
    return f1_micro

print("\n📊 Evaluation on held-out test set:")
evaluate(clf_equip, X_test, y_equip_test, mlb_equip, "Equipment",   TOP_K_EQUIPMENT)
evaluate(clf_meth,  X_test, y_meth_test,  mlb_meth,  "Methodology", TOP_K_METHODOLOGY)
evaluate(clf_tool,  X_test, y_tool_test,  mlb_tool,  "Tool",        TOP_K_TOOLS)

# ──────────────────────────────────────────────
# 8. SAVE ALL ARTEFACTS
# ──────────────────────────────────────────────
print("\n💾 Saving model artefacts...")

joblib.dump(tfidf,        f"{MODEL_DIR}/tfidf.pkl")
joblib.dump(ohe,          f"{MODEL_DIR}/ohe.pkl")
joblib.dump(scaler,       f"{MODEL_DIR}/scaler.pkl")
joblib.dump(mlb_equip,    f"{MODEL_DIR}/mlb_equipment.pkl")
joblib.dump(mlb_meth,     f"{MODEL_DIR}/mlb_methodology.pkl")
joblib.dump(mlb_tool,     f"{MODEL_DIR}/mlb_tool.pkl")
joblib.dump(clf_equip,    f"{MODEL_DIR}/clf_equipment.pkl")
joblib.dump(clf_meth,     f"{MODEL_DIR}/clf_methodology.pkl")
joblib.dump(clf_tool,     f"{MODEL_DIR}/clf_tool.pkl")
joblib.dump(knn,          f"{MODEL_DIR}/knn.pkl")

# Save project index for KNN lookup
project_index = df[["project_id", "project_title", "research_domain", "location"]].to_dict(orient="records")
with open(f"{MODEL_DIR}/project_index.json", "w") as f:
    json.dump(project_index, f)

# Save KNN feature matrix (needed for inference)
joblib.dump(X, f"{MODEL_DIR}/X_full.pkl")

print(f"   ✓ All artefacts saved to ./{MODEL_DIR}/")

# ──────────────────────────────────────────────
# 9. LOAD ROADMAP INDEX
# ──────────────────────────────────────────────
print("\n🗺️  Building roadmap index...")
roadmap_index = defaultdict(list)
for _, row in roadmap_df.iterrows():
    roadmap_index[row["project_id"]].append({
        "phase_number": int(row["phase_number"]),
        "phase_name": row["phase_name"],
        "steps": row["steps"].split(" | ") if pd.notna(row["steps"]) else [],
    })

# Fallback: domain → project_type roadmap mapping
project_type_map = df.set_index("project_id")["project_type"].to_dict()

with open(f"{MODEL_DIR}/roadmap_index.json", "w") as f:
    json.dump(dict(roadmap_index), f)

with open(f"{MODEL_DIR}/project_type_map.json", "w") as f:
    json.dump(project_type_map, f)

print("   ✓ Roadmap index saved")

print("\n" + "=" * 55)
print("  ✅  TRAINING COMPLETE — All models saved to ./models/")
print("=" * 55)
print("\nModel files:")
for fname in sorted(os.listdir(MODEL_DIR)):
    size = os.path.getsize(f"{MODEL_DIR}/{fname}") / 1024
    print(f"   {fname:<35} {size:>8.1f} KB")