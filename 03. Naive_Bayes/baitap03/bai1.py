# -*- coding: utf-8 -*-
"""Bài tập 1: Bayes với ba nguyên nhân (ba nhóm sinh viên)."""


def bayes_nhieu_nhanh(tien_nghiem, kha_nang):
    """Tra ve hau nghiem cua tung nhanh.

    tien_nghiem : danh sach P(nhom), cong lai bang 1
    kha_nang    : danh sach P(qua mon | nhom)
    """
    tu_so = [p * k for p, k in zip(tien_nghiem, kha_nang)]
    mau_so = sum(tu_so)  # cong thuc xac suat toan phan P(qua mon)
    return [t / mau_so for t in tu_so]


ten_nhom = ["hoc deu", "hoc don", "it hoc"]
ty_le_nhom = [0.50, 0.30, 0.20]
ty_le_qua = [0.95, 0.70, 0.30]

hau_nghiem = bayes_nhieu_nhanh(ty_le_nhom, ty_le_qua)

print("Gap mot ban da qua mon. Xac suat ban do thuoc tung nhom:")
for ten, p in zip(ten_nhom, hau_nghiem):
    print(f"  {ten:8s}: {p:.4f}")
print()
print(f"Tong ba xac suat = {sum(hau_nghiem):.4f} (phai bang 1)")
