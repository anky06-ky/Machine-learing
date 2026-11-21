# -*- coding: utf-8 -*-
"""Bài tập 4: thêm số phòng vào mô hình, so R2 với mô hình một biến."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Đường dẫn tính theo vị trí tệp mã, nên chạy từ thư mục nào cũng được.
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "gia_nha.csv"
df = pd.read_csv(DATA_PATH)
y = df["gia"]


def tinh_r2(cac_cot):
    """R2 trên tập kiểm tra, chia 80/20 với random_state=42 như trong tài liệu."""
    X = df[cac_cot]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    mo_hinh = LinearRegression()
    mo_hinh.fit(X_train, y_train)
    return r2_score(y_test, mo_hinh.predict(X_test))


r2_mot_bien = tinh_r2(["dien_tich"])
r2_hai_bien = tinh_r2(["dien_tich", "so_phong"])

print(f"R2 mo hinh mot bien (dien_tich)           : {r2_mot_bien:.4f}")
print(f"R2 mo hinh hai bien (dien_tich + so_phong): {r2_hai_bien:.4f}")
print(f"Chenh lech: {r2_hai_bien - r2_mot_bien:+.4f}")
print()
print(f"Nhan xet: them so_phong lam R2 tang tu {r2_mot_bien:.4f} len "
      f"{r2_hai_bien:.4f}. Muc tang co that nhung nho, vi dien_tich da "
      "giai thich gan het gia va so phong di kem chat voi dien_tich. "
      "Them cot nay co loi nhe, dang dung nhung khong tao khac biet lon.")
