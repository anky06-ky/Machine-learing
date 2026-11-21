# -*- coding: utf-8 -*-
"""Bài tập 1: tự tính silhouette của từng điểm chỉ bằng numpy."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

THU_MUC = Path(__file__).resolve().parent
DATA_DIR = THU_MUC.parent / "data"


def silhouette_mot_diem(X, nhan, i):
    """Tính a, b, s của điểm thứ i theo đúng định nghĩa ở mục 4.

    a: khoảng cách trung bình tới các điểm khác cùng cụm (không tính chính nó).
    b: khoảng cách trung bình tới các điểm của cụm gần nhất khác.
    s = (b - a) / max(a, b).
    """
    X = np.asarray(X, dtype=float)
    nhan = np.asarray(nhan)
    kc = np.sqrt(((X - X[i]) ** 2).sum(axis=1))  # khoảng cách tới mọi điểm

    cung_cum = nhan == nhan[i]
    cung_cum[i] = False
    if not cung_cum.any():  # cụm chỉ có một điểm, quy ước s = 0
        return 0.0, 0.0, 0.0
    a = kc[cung_cum].mean()
    b = min(kc[nhan == c].mean() for c in np.unique(nhan) if c != nhan[i])
    s = (b - a) / max(a, b)
    return a, b, s


df = pd.read_csv(DATA_DIR / "khach_hang.csv")
X = df[["thu_nhap", "diem_chi_tieu"]]
Xs = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Xs)
nhan = km.labels_

ket_qua = np.array([silhouette_mot_diem(Xs, nhan, i) for i in range(len(Xs))])
s = ket_qua[:, 2]
print(f"Silhouette trung binh tu tinh : {s.mean():.6f}")
print(f"silhouette_score cua sklearn  : {silhouette_score(Xs, nhan):.6f}")
print()

thap = np.argsort(s)[:5]
print("Nam khach co s nho nhat:")
print("  dong  thu_nhap  chi_tieu  tuoi  cum       a       b       s")
for i in thap:
    a, b, si = ket_qua[i]
    print(f"  {i:4d}  {df.thu_nhap[i]:8.1f}  {df.diem_chi_tieu[i]:8d}"
          f"  {df.tuoi[i]:4d}  {nhan[i]:3d}  {a:.4f}  {b:.4f}  {si:.4f}")

# Cụm gần thứ hai của khách tệ nhất, để biết khách nằm giữa hai cụm nào
i = thap[0]
kc_cum = {c: np.sqrt(((Xs[nhan == c] - Xs[i]) ** 2).sum(axis=1)).mean()
          for c in np.unique(nhan) if c != nhan[i]}
cum_gan = min(kc_cum, key=kc_cum.get)
tam_goc = pd.DataFrame(
    StandardScaler().fit(X).inverse_transform(km.cluster_centers_),
    columns=["thu_nhap", "diem_chi_tieu"])
print()
print(f"Khach {i}: thu nhap {df.thu_nhap[i]}, chi tieu {df.diem_chi_tieu[i]},"
      f" thuoc cum {nhan[i]}, cum gan nhat khac la cum {cum_gan}")
print(f"  tam cum {nhan[i]}: ({tam_goc.iloc[nhan[i], 0]:.1f},"
      f" {tam_goc.iloc[nhan[i], 1]:.1f}),"
      f" tam cum {cum_gan}: ({tam_goc.iloc[cum_gan, 0]:.1f},"
      f" {tam_goc.iloc[cum_gan, 1]:.1f})")

# Vẽ lại Hình 3 và đánh dấu khách có s nhỏ nhất
plt.figure(figsize=(7, 5))
for c in range(5):
    plt.scatter(df.thu_nhap[nhan == c], df.diem_chi_tieu[nhan == c], s=18,
                label=f"Cum {c}")
plt.scatter(tam_goc.thu_nhap, tam_goc.diem_chi_tieu, marker="X", s=150,
            c="black", label="Tam cum")
plt.scatter(df.thu_nhap[i], df.diem_chi_tieu[i], s=300, facecolors="none",
            edgecolors="red", linewidths=2, label=f"Khach {i}, s = {s[i]:.3f}")
plt.xlabel("Thu nhap (trieu dong mot thang)")
plt.ylabel("Diem chi tieu")
plt.title("K-Means voi K = 5, khach co silhouette nho nhat")
plt.legend(fontsize=8, loc="upper center", ncol=2)
plt.tight_layout()
plt.savefig(THU_MUC / "silhouette_khach_yeu_nhat.png", dpi=150)
plt.close()
print("Da luu hinh:", (THU_MUC / "silhouette_khach_yeu_nhat.png")
      .relative_to(THU_MUC.parent))
