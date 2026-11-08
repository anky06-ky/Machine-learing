"""Mục 7: phân tầng, CV nhiều thước đo và class_weight."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import read_data, split_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = read_data("sinh_vien_bo_hoc.csv")
counts = [int(train_test_split(df.bo_hoc, test_size=0.25, random_state=rs)[1].sum()) for rs in range(10)]
print("Khong stratify, so ban bo hoc tren test:", counts)
X_tr, X_te, y_tr, y_te = split_data()
lr = make_pipeline(StandardScaler(), LogisticRegression())
scores = cross_validate(lr, X_tr, y_tr, cv=5, scoring=["accuracy", "precision", "recall", "f1"])
for name in ["accuracy", "precision", "recall", "f1"]:
    print(f"CV {name:10s}: {scores['test_'+name].mean():.4f}")
print("Tap kiem tra   precision  recall    F1")
for weight in [None, "balanced"]:
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight=weight))
    du = model.fit(X_tr, y_tr).predict(X_te)
    print(f"{str(weight):13s} {precision_score(y_te, du, zero_division=0):.4f}    {recall_score(y_te, du):.4f}  {f1_score(y_te, du):.4f}")
print("Can bang lop phat nang hon khi doan sai lop hiem; van phai xem ca precision.")
