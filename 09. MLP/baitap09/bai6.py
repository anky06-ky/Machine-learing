# -*- coding: utf-8 -*-
"""Bài tập 6: mạng nơ-ron cho hồi quy giá nhà, và chuyện ngoại suy."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THU_MUC = Path(__file__).resolve().parent
DATA_DIR = THU_MUC.parent / "data"

# Giống bài 1: giá theo diện tích, chia 80/20 với random_state=42
df = pd.read_csv(DATA_DIR / "gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42)

mo_hinh = {
    "MLP (20,)": make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(20,), max_iter=5000,
                     random_state=0)),
    "Hoi quy tuyen tinh": LinearRegression(),
}
can_250 = pd.DataFrame({"dien_tich": [250.0]})
print(f"Dien tich trong tap hoc: tu {X_tr.dien_tich.min()} toi"
      f" {X_tr.dien_tich.max()} m2")
print("  Mo hinh               MAE tap kiem tra   gia can 250 m2")
for ten, mh in mo_hinh.items():
    mh.fit(X_tr, y_tr)
    mae = mean_absolute_error(y_te, mh.predict(X_te))
    print(f"  {ten:20s}  {mae:9.4f} ty        {mh.predict(can_250)[0]:8.4f} ty")
print()

# Độ dốc của mỗi mô hình ở rìa dữ liệu và ở xa ngoài miền học
print("Gia tang them khi dien tich tang 10 m2:")
for ten, mh in mo_hinh.items():
    doc = [mh.predict(pd.DataFrame({"dien_tich": [a + 10.0]}))[0]
           - mh.predict(pd.DataFrame({"dien_tich": [a]}))[0]
           for a in [60.0, 110.0, 240.0]]
    print(f"  {ten:20s}  tu 60 m2: {doc[0]:.3f}, tu 110 m2: {doc[1]:.3f},"
          f" tu 240 m2: {doc[2]:.3f} ty")

# Vẽ hai đường dự đoán, kéo dài tới 260 m2
luoi = pd.DataFrame({"dien_tich": np.linspace(20, 260, 300)})
plt.figure(figsize=(8, 5))
plt.scatter(X_tr.dien_tich, y_tr, s=18, label="Tap hoc")
plt.scatter(X_te.dien_tich, y_te, s=18, label="Tap kiem tra")
for ten, mh in mo_hinh.items():
    plt.plot(luoi.dien_tich, mh.predict(luoi), label=ten)
plt.axvspan(X_tr.dien_tich.max(), 260, color="gray", alpha=0.15,
            label="Ngoai mien hoc")
plt.xlabel("Dien tich (m2)")
plt.ylabel("Gia (ty dong)")
plt.title("MLP va hoi quy tuyen tinh khi ngoai suy")
plt.legend()
plt.tight_layout()
plt.savefig(THU_MUC / "mlp_hoi_quy_ngoai_suy.png", dpi=150)
plt.close()
print("Da luu hinh:", (THU_MUC / "mlp_hoi_quy_ngoai_suy.png")
      .relative_to(THU_MUC.parent))
