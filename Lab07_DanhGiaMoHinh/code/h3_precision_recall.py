"""Mục 4: tính tay và đọc classification_report."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import split_data
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_tr, X_te, y_tr, y_te = split_data()
du = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, y_tr).predict(X_te)
tn, fp, fn, tp = confusion_matrix(y_te, du, labels=[0, 1]).ravel()
p = tp / (tp + fp) if tp + fp else 0.0
r = tp / (tp + fn) if tp + fn else 0.0
f1 = 2*p*r/(p+r) if p+r else 0.0
thu_vien = [precision_score(y_te, du, zero_division=0),
            recall_score(y_te, du, zero_division=0), f1_score(y_te, du, zero_division=0)]
print("                 precision  recall    F1")
print("Tinh tay:       ", "  ".join(f"{x:.4f}" for x in [p, r, f1]))
print("Scikit-learn:   ", "  ".join(f"{x:.4f}" for x in thu_vien))
np.testing.assert_allclose([p, r, f1], thu_vien)
print(classification_report(y_te, du, target_names=["khong bo", "bo hoc"], digits=4, zero_division=0))
print("Macro: moi lop nang nhu nhau. Weighted: trong so theo support.")
