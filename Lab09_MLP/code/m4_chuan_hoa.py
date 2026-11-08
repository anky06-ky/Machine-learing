"""Mục 5: giữ nguyên mạng và cách chia, chỉ thay chuẩn hoá."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import split_data
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_tr, X_te, y_tr, y_te = split_data("sinh_vien_mo_rong.csv", "qua_mon")
raw = MLPClassifier(hidden_layer_sizes=(20,), max_iter=3000, random_state=0).fit(X_tr, y_tr)
scaled = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(20,), max_iter=3000, random_state=0)).fit(X_tr, y_tr)
print(f"Doan ca lop qua: {y_te.mean():.4f}")
print(f"Khong chuan hoa: {raw.score(X_te, y_te):.4f}")
print(f"Co chuan hoa: {scaled.score(X_te, y_te):.4f}")
print(f"Khong chuan hoa, so ban doan qua: {raw.predict(X_te).sum()} tren {len(y_te)}")
print("Cot chi tieu thang do lon chi phoi toi uu; scaler chi fit tren tap hoc trong pipeline.")
