# -*- coding: utf-8 -*-
"""Bài tập 4: tự vẽ đường ROC của hồi quy logistic và rừng ngẫu nhiên."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THU_MUC = Path(__file__).resolve().parent
DATA_DIR = THU_MUC.parent / "data"

df = pd.read_csv(DATA_DIR / "sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = {
    "Hoi quy logistic": make_pipeline(StandardScaler(), LogisticRegression()),
    "Rung 200 cay": RandomForestClassifier(200, random_state=0),
}

duong = {}
plt.figure(figsize=(6, 6))
for ten, mh in mo_hinh.items():
    p = mh.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, p)
    auc = roc_auc_score(y_te, p)
    duong[ten] = (fpr, tpr)
    plt.plot(fpr, tpr, label=f"{ten}, AUC = {auc:.4f}")
    print(f"{ten:17s} AUC = {auc:.4f}, duong ROC co {len(fpr)} diem")

plt.plot([0, 1], [0, 1], "k--", label="Doan ngau nhien, AUC = 0.5")
plt.xlabel("Ty le bao dong nham (FPR)")
plt.ylabel("Recall (TPR)")
plt.title("Duong ROC tren tap kiem tra")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(THU_MUC / "roc.png", dpi=150)
plt.close()
print("Da luu hinh:", (THU_MUC / "roc.png").relative_to(THU_MUC.parent))
print()

# So hai đường đúng như trên hình: plt.plot nối các điểm ROC bằng đoạn
# thẳng, np.interp tính recall trên đoạn nối đó tại 10001 mốc FPR.
moc = np.linspace(0, 1, 10001)
recall_lr = np.interp(moc, *duong["Hoi quy logistic"])
recall_rf = np.interp(moc, *duong["Rung 200 cay"])
rung_cao_hon = recall_rf > recall_lr + 1e-9
print("So recall o cung ty le bao dong nham (tren 10001 moc FPR):")
print(f"  logistic cao hon: {np.mean(recall_lr > recall_rf + 1e-9):.1%} so moc,"
      f" rung cao hon: {rung_cao_hon.mean():.1%} so moc")
# Tìm các đoạn liền nhau mà rừng nằm trên logistic
bien = np.flatnonzero(np.diff(np.r_[False, rung_cao_hon, False].astype(int)))
for dau, cuoi in zip(bien[::2], bien[1::2] - 1):
    i = dau + np.argmax(recall_rf[dau:cuoi + 1] - recall_lr[dau:cuoi + 1])
    print(f"  rung nam tren logistic khi FPR tu {moc[dau]:.4f} toi"
          f" {moc[cuoi]:.4f}; cach xa nhat tai FPR {moc[i]:.4f}:"
          f" rung {recall_rf[i]:.4f}, logistic {recall_lr[i]:.4f}")
print("Hai duong cat nhau:", "co" if rung_cao_hon.any() else "khong")
