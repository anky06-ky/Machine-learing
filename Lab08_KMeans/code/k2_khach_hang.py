"""Mục 3: phân khúc hai đặc trưng và mô tả thêm tuổi."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import read_data, savefig, plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = read_data("khach_hang.csv")
print("Kich thuoc bang:", df.shape)
print(df.describe().round(1).to_string())
columns = ["thu_nhap", "diem_chi_tieu"]
scaler = StandardScaler().fit(df[columns])
Xs = scaler.transform(df[columns])
km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Xs)
df["cum"] = km.labels_
centers = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=columns)
centers["so_khach"] = np.bincount(km.labels_, minlength=5)
centers["tuoi_tb"] = df.groupby("cum").tuoi.mean()
print("Tam cum, don vi goc:\n", centers.round(1).to_string())
print(f"Inertia: {km.inertia_:.2f}")
new = pd.DataFrame({"thu_nhap": [29., 9.], "diem_chi_tieu": [85, 75]})
print("Khach moi:\n", new.assign(cum=km.predict(scaler.transform(new))).to_string(index=False))
plt.scatter(df.thu_nhap, df.diem_chi_tieu, c=km.labels_, cmap="tab10", s=24)
plt.scatter(centers.thu_nhap, centers.diem_chi_tieu, c="black", marker="X", s=140)
for i, row in centers.iterrows():
    plt.annotate(f" {i}", (row.thu_nhap, row.diem_chi_tieu))
plt.xlabel("Thu nhap (trieu/thang)"); plt.ylabel("Diem chi tieu")
savefig("phan_khuc_khach_hang.png")
print("Doc y nghia cum theo tam; so thu tu cum khong phai thu hang hay nhan that.")
