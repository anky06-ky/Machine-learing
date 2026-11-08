"""Mục 5: minh hoạ đổi ngưỡng; chọn ngưỡng bằng OOF trên tập học."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import split_data, savefig, plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_tr, X_te, y_tr, y_te = split_data()
lr = make_pipeline(StandardScaler(), LogisticRegression())
# Mỗi xác suất OOF đến từ mô hình chưa học mẫu tương ứng.
oof = cross_val_predict(lr, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
scores = [f1_score(y_tr, oof >= t, zero_division=0) for t in thresholds]
best = thresholds[np.argmax(scores)]
print(f"Nguong chon tren tap hoc: {best:.2f}; F1 OOF={max(scores):.4f}")
prob = lr.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
print("Minh hoa tren test; khong dung bang nay de chon nguong.")
print("nguong  so bao   TP  FP  FN   precision  recall    F1")
for t in [0.5, 0.4, 0.3, 0.2, 0.1, best]:
    du = prob >= t
    tn, fp, fn, tp = confusion_matrix(y_te, du, labels=[0, 1]).ravel()
    print(f"{t:6.2f} {du.sum():7d} {tp:4d} {fp:3d} {fn:3d} "
          f"{precision_score(y_te, du, zero_division=0):9.4f} "
          f"{recall_score(y_te, du):7.4f} {f1_score(y_te, du):7.4f}")
plt.plot(thresholds, scores, "o-")
plt.axvline(best, color="red", linestyle="--", label=f"Nguong {best:.2f}")
plt.xlabel("Nguong"); plt.ylabel("F1 OOF tren tap hoc"); plt.legend()
savefig("chon_nguong_oof.png")
print("Ha nguong thuong tang recall, dong thoi tang so bao dong nham.")
