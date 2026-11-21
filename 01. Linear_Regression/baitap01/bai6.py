# -*- coding: utf-8 -*-
"""Bài tập 6: hàm dự đoán giá có cảnh báo khi diện tích ngoài vùng dữ liệu."""

# Hai hệ số đã tìm ở Mục 5, không cần huấn luyện lại.
w = 0.078367
b = 0.401752


def du_doan_gia(dien_tich):
    """Dự đoán giá (tỷ đồng) theo mô hình một biến, cảnh báo nếu ngoại suy."""
    if dien_tich < 35.5 or dien_tich > 117.5:
        print(f"  Canh bao: {dien_tich} m2 nam ngoai khoang 35.5 - 117.5 m2 "
              "cua du lieu, ket qua chi la suy doan keo dai.")
    return w * dien_tich + b


for dien_tich in (60, 80, 200):
    gia_du_doan = du_doan_gia(dien_tich)
    print(f"Can {dien_tich} m2 -> du doan {gia_du_doan:.3f} ty dong")
