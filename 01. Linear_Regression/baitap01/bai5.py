# -*- coding: utf-8 -*-
"""Bài tập 5: thử các tốc độ học khác nhau cho gradient descent.

Thân vòng lặp giống b6_gradient_descent.py, chỉ đổi toc_do_hoc, giữ 200 vòng.
"""

from pathlib import Path

import pandas as pd

# Đường dẫn tính theo vị trí tệp mã, nên chạy từ thư mục nào cũng được.
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "gia_nha.csv"
df = pd.read_csv(DATA_PATH)
x_goc = df["dien_tich"].to_numpy()
y = df["gia"].to_numpy()

# Chuẩn hoá diện tích như ở b6
x_tb = x_goc.mean()
x_do_lech = x_goc.std()
x = (x_goc - x_tb) / x_do_lech
n = len(x)
so_vong = 200


def chay_gradient_descent(toc_do_hoc):
    """Chạy 200 vòng, trả về MSE sau vòng cuối."""
    w, b = 0.0, 0.0
    for vong in range(1, so_vong + 1):
        chenh_lech = (w * x + b) - y
        grad_w = (2 / n) * (chenh_lech * x).sum()
        grad_b = (2 / n) * chenh_lech.sum()
        w -= toc_do_hoc * grad_w
        b -= toc_do_hoc * grad_b
    return (((w * x + b) - y) ** 2).mean()


print("Hai toc do hoc theo de bai (200 vong):")
for toc_do_hoc in (0.001, 1.02):
    mse = chay_gradient_descent(toc_do_hoc)
    print(f"  toc do hoc = {toc_do_hoc:<6}  MSE o vong 200 = {mse:.4f}")
print()

print("Lam them: cac toc do hoc khac (200 vong):")
for toc_do_hoc in (0.1, 0.5, 0.9, 0.99, 1.0, 1.01):
    mse = chay_gradient_descent(toc_do_hoc)
    print(f"  toc do hoc = {toc_do_hoc:<6}  MSE o vong 200 = {mse:.4f}")
print()

print("Giai thich:")
print("- 0.001 qua nho: moi buoc chi nhich rat it nen sau 200 vong van chua "
      "toi day, MSE van con cao.")
print("- 1.02 qua lon: moi buoc nhay vuot qua day sang suon ben kia va dung "
      "cao hon cho cu, nen MSE tang len rat lon.")
print("- Sau khi chuan hoa, ranh gioi nam quanh toc do hoc = 1.0. Voi 0.1, 0.5, "
      "0.9 van ve dung 0.1790, 0.99 ve cham hon, tu 1.0 tro len MSE khong "
      "giam nua va 1.01, 1.02 thi bung ra.")
