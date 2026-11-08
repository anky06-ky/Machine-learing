# -*- coding: utf-8 -*-
"""Bài 4: tự vẽ hai đường ROC và nhận xét từ kết quả thực tế."""

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "data" / "sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
curves, aucs = {}, {}
for name, model in [("Logistic", make_pipeline(StandardScaler(), LogisticRegression())),
                    ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=0))]:
    prob = model.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, prob)
    curves[name] = (fpr, tpr)
    aucs[name] = roc_auc_score(y_te, prob)
    plt.plot(fpr, tpr, label=f"{name}: AUC={aucs[name]:.4f}")
    print(f"{name}: AUC={aucs[name]:.4f}")
plt.plot([0, 1], [0, 1], "k--", label="Ngau nhien")
plt.xlabel("FPR"); plt.ylabel("TPR / Recall"); plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "roc.png", dpi=150, bbox_inches="tight")
plt.close()
print("Da luu: roc.png")
# So hai đường ROC dạng bậc thang tại các FPR chung, xử lý cả đoạn đứng.
grid = np.unique(np.concatenate([v[0] for v in curves.values()]))
heights = [np.array([tpr[fpr <= x].max() for x in grid]) for fpr, tpr in curves.values()]
diff = heights[0] - heights[1]
print("AUC lon hon (tong the):", max(aucs, key=aucs.get))
print("Hai duong doi thu tu cao/thap:", bool(np.any(diff > 1e-12) and np.any(diff < -1e-12)))
print("AUC lon hon khong dong nghia tot hon tai moi nguong; xem cac doan giao tren hinh.")
