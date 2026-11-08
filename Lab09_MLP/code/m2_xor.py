"""Mục 3: logistic và MLP 1, 2, 4, 8 nơ-ron ẩn trên XOR."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import split_data
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X_tr, X_te, y_tr, y_te = split_data("xor.csv", "nhan")
lr = LogisticRegression().fit(X_tr, y_tr)
print(f"Logistic: {accuracy_score(y_te, lr.predict(X_te)):.4f}")
print("No-ron an  so tham so   tap hoc    tap kiem tra")
for width in [1, 2, 4, 8]:
    model = MLPClassifier(hidden_layer_sizes=(width,), solver="lbfgs", max_iter=5000, random_state=0).fit(X_tr, y_tr)
    n_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
    print(f"{width:9d} {n_params:11d} {model.score(X_tr, y_tr):10.4f} {model.score(X_te, y_te):14.4f}")
print("XOR khong tach duoc bang mot duong thang; lop an phi tuyen tao duoc ranh gioi gap khuc.")
