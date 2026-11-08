# -*- coding: utf-8 -*-
"""Bài 6: MLPRegressor và hồi quy tuyến tính, cùng một đầu vào diện tích."""

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "data" / "gia_nha.csv")
X_tr, X_te, y_tr, y_te = train_test_split(df[["dien_tich"]], df.gia, test_size=0.2, random_state=42)
new = pd.DataFrame({"dien_tich": [250.]})
grid = pd.DataFrame({"dien_tich": np.linspace(20, 270, 250)})
print(f"Mien dien tich tap hoc: [{X_tr.dien_tich.min()}, {X_tr.dien_tich.max()}] m2")
plt.scatter(X_tr.dien_tich, y_tr, s=20, label="Tap hoc")
for name, model in [("MLP", make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(20,), max_iter=5000, random_state=0))),
                    ("Tuyen tinh", LinearRegression())]:
    model.fit(X_tr, y_tr)
    print(f"{name}: MAE={mean_absolute_error(y_te, model.predict(X_te)):.4f} ty; gia 250m2={model.predict(new)[0]:.4f} ty")
    plt.plot(grid.dien_tich, model.predict(grid), label=name)
plt.axvline(250, color="gray", linestyle="--")
plt.xlabel("Dien tich (m2)"); plt.ylabel("Gia (ty dong)"); plt.legend()
plt.tight_layout()
plt.savefig(BASE_DIR / "mlp_hoi_quy_ngoai_suy.png", dpi=150, bbox_inches="tight")
plt.close()
print("Da luu: mlp_hoi_quy_ngoai_suy.png")
print("250m2 nam ngoai mien hoc: day la ngoai suy, khong co gia that de khang dinh mo hinh nao dung.")
print("MLP ReLU cho duong gap khuc; khong bi rang buoc tiep tuc xu huong nhu mo hinh tuyen tinh.")
