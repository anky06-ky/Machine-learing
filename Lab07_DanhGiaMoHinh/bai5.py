# -*- coding: utf-8 -*-
"""Bài 5: class_weight cho rừng và SVM; so độ thay đổi của recall."""

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "data" / "sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
changes = {}
print("Mo hinh   class_weight  precision  recall    F1")
for name in ["Rung", "SVM"]:
    results = []
    for weight in [None, "balanced"]:
        model = (RandomForestClassifier(n_estimators=200, random_state=0, class_weight=weight)
                 if name == "Rung" else make_pipeline(StandardScaler(), SVC(class_weight=weight)))
        du = model.fit(X_tr, y_tr).predict(X_te)
        scores = [precision_score(y_te, du, zero_division=0), recall_score(y_te, du), f1_score(y_te, du)]
        results.append(scores)
        print(f"{name:8s} {str(weight):12s}", "  ".join(f"{x:.4f}" for x in scores))
    delta = [b-a for a, b in zip(*results)]
    changes[name] = abs(delta[1])
    print("Thay doi precision, recall, F1:", [round(x, 4) for x in delta])
print("Thay doi recall nhieu nhat:", max(changes, key=changes.get))
print("Cac mo hinh phan ung khac nhau; can bang lop khong bao dam F1 tang.")
