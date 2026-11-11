# -*- coding: utf-8 -*-
"""Bài tập 4: lưu ảnh đã nén 16 màu và so dung lượng tệp.

Chạy mặc định với ảnh china.jpg của scikit-learn. Thêm ảnh của chính em:
    python baitap08/bai4.py --anh "D:/Anh/anh_cua_em.jpg"
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

THU_MUC = Path(__file__).resolve().parent


def nen_16_mau(anh):
    """Dựa vào k5_nen_anh.py: học 16 tâm trên 5000 điểm ảnh rồi tô lại ảnh."""
    cao, rong, _ = anh.shape
    diem = anh.reshape(-1, 3)
    rng = np.random.default_rng(0)
    so_mau_hoc = min(5000, len(diem))
    mau = diem[rng.choice(len(diem), so_mau_hoc, replace=False)]
    km = KMeans(n_clusters=16, n_init=4, random_state=0).fit(mau)
    anh_moi = km.cluster_centers_[km.predict(diem)].reshape(cao, rong, 3)
    sai = np.abs(anh_moi - anh).mean() * 255
    return anh_moi, sai


def dua_ve_rgb_0_1(anh):
    """Ảnh về dạng (cao, rộng, 3), giá trị thực từ 0 tới 1."""
    anh = np.asarray(anh)
    if anh.ndim == 2:  # ảnh xám, nhân ba kênh
        anh = np.stack([anh] * 3, axis=-1)
    if anh.shape[2] == 4:  # PNG có kênh trong suốt, bỏ kênh thứ tư
        print("  Anh co 4 kenh (RGBA), bo kenh trong suot truoc khi phan cum")
        anh = anh[:, :, :3]
    if anh.dtype == np.uint8:  # JPG đọc ra số nguyên 0..255
        anh = anh / 255.0
    return anh.astype(float)


def xu_ly(anh, ten):
    anh = dua_ve_rgb_0_1(anh)
    anh_moi, sai = nen_16_mau(anh)
    tep_goc = THU_MUC / f"{ten}_goc.png"
    tep_16 = THU_MUC / f"{ten}_16_mau.png"
    plt.imsave(tep_goc, anh)
    plt.imsave(tep_16, anh_moi)
    kb_goc = tep_goc.stat().st_size / 1024
    kb_16 = tep_16.stat().st_size / 1024
    so_mau = len(np.unique(np.round(anh_moi.reshape(-1, 3) * 255), axis=0))
    print(f"  anh_moi co dang {anh_moi.shape}, con {so_mau} mau,"
          f" sai lech trung binh {sai:.2f} muc sang")
    print(f"  {tep_goc.name:22s} {kb_goc:8.1f} KB")
    print(f"  {tep_16.name:22s} {kb_16:8.1f} KB")
    print(f"  Tep 16 mau nho hon {kb_goc / kb_16:.2f} lan")


ap = argparse.ArgumentParser(description="Nen anh con 16 mau bang K-Means")
ap.add_argument("--anh", help="duong dan toi anh cua em (jpg hoac png)")
tham_so = ap.parse_args()

print("Anh china.jpg cua scikit-learn:")
xu_ly(load_sample_image("china.jpg"), "anh")
print()

if tham_so.anh:
    print(f"Anh cua em: {tham_so.anh}")
    xu_ly(plt.imread(tham_so.anh), "anh_cua_em")
else:
    print("Chua chi anh cua em. Chay lai voi --anh DUONG_DAN de lam phan hai.")
print()
print("Ly thuyet: 16 mau can 4 bit moi diem thay vi 24 bit, nho hon 6 lan.")
print("Tep PNG con nen them, nen ty le dung luong that khong dung bang 6.")
