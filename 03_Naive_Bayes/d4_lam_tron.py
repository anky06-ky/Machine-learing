# -*- coding: utf-8 -*-
"""Bước 4: tự cài Naive Bayes trên một kho nhỏ để thấy vấn đề số không.

Đoạn này cố ý KHÔNG dùng scikit-learn, vì thư viện đã làm trơn sẵn nên
sinh viên không nhìn thấy chỗ hỏng.
"""
import sys
from collections import Counter

# Đảm bảo hiển thị tiếng Việt trên terminal Windows không bị lỗi encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Kho tin nhắn tí hon, viết thẳng vào mã cho dễ theo dõi
kho = [
    ("trung thuong ngay", 1),
    ("khuyen mai giam gia", 1),
    ("trung thuong click", 1),
    ("hop nhom chieu mai", 0),
    ("bai tap nhom nop mai", 0),
    ("thay gui tai lieu", 0),
]

tu_dien = sorted({t for cau, nhan in kho for t in cau.split()})
V = len(tu_dien)
print(f"Tu dien co {V} tu:")
# In thành nhiều dòng, mỗi dòng 9 từ, để không tràn ra ngoài khung
for i in range(0, V, 9):
    print(" " + " ".join(tu_dien[i:i+9]))
print()

# Đếm số lần mỗi từ xuất hiện trong từng Lớp
dem = {0: Counter(), 1: Counter()}
tong = {0: 0, 1: 0}
for cau, nhan in kho:
    for t in cau.split():
        dem[nhan][t] += 1
        tong[nhan] += 1

print("So lan xuat hien cua tung tu")
print(f" {'tu':12s} {'rac':>5s} {'thuong':>7s}")
for t in tu_dien:
    print(f" {t:12s} {dem[1][t]:5d} {dem[0][t]:7d}")
print()

def xac_suat(tu, nhan, alpha):
    """P(tu|nhan) co lam tron Laplace voi he so alpha."""
    return (dem[nhan][tu] + alpha) / (tong[nhan] + alpha * V)

def diem(cau, nhan, alpha):
    """Tich cua xac suat tien nghiem va cac xac suat tu."""
    p = 0.5  # hai lop can bang nen tien nghiem bang nhau
    for t in cau.split():
        if t in tu_dien:
            p *= xac_suat(t, nhan, alpha)
    return p

cau_thu = "trung thuong bai tap"
print(f"Tin nhan thu: \"{cau_thu}\"")
print("Tu 'bai' chi xuat hien o lop thuong, chua bao gio o lop rac.")
print()

for alpha in [0.0, 1.0]:
    d_rac = diem(cau_thu, 1, alpha)
    d_thuong = diem(cau_thu, 0, alpha)
    ket = "rac" if d_rac > d_thuong else "thuong"
    
    print(f"alpha = {alpha}")
    print(f"  diem lop rac    = {d_rac:.10f}")
    print(f"  diem lop thuong = {d_thuong:.10f}")
    print(f"  ket luan        = {ket}")
    print()

print("Khong lam tron, CA HAI diem deu bang 0 nen mo hinh te liet.")
print("Lop rac bang 0 vi tu 'bai' chua tung xuat hien o lop rac.")
print("Lop thuong bang 0 vi tu 'trung' chua tung xuat hien o lop thuong.")
print("Co lam tron, diem lop rac lon hon va mo hinh ket luan dung.")