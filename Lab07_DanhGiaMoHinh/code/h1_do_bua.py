"""Mục 2: so accuracy với mô hình luôn đoán lớp đông nhất."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import read_data, split_data
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = read_data("sinh_vien_bo_hoc.csv")
print("Kich thuoc bang:", df.shape)
print(df.head().to_string(index=False))
print("So sinh vien moi lop:\n", df.bo_hoc.value_counts().to_string())
print(f"Ty le bo hoc: {df.bo_hoc.mean():.3f}")
X_tr, X_te, y_tr, y_te = split_data()
print(f"Tap kiem tra: {len(y_te)} ban, {y_te.sum()} ban bo hoc")
for ten, model in [("Mo hinh luoi", DummyClassifier(strategy="most_frequent")),
                   ("Hoi quy logistic", make_pipeline(StandardScaler(), LogisticRegression()))]:
    du = model.fit(X_tr, y_tr).predict(X_te)
    print(f"{ten:17s} accuracy {accuracy_score(y_te, du):.4f}, "
          f"tim ra {((du == 1) & (y_te == 1)).sum()} tren {y_te.sum()} ban bo hoc")
print("Nhan xet: accuracy cao chua du; can xem so ban bo hoc duoc tim ra.")
