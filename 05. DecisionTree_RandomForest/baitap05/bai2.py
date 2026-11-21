# -*- coding: utf-8 -*-
"""Bài tập 2: tự tìm câu hỏi gốc trên dữ liệu thật bằng hàm tim_cau_hoi."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)


def gini(nhan):
    """Chi so Gini cua mot nut co hai lop."""
    if len(nhan) == 0:
        return 0.0
    p1 = nhan.mean()
    return 1 - p1 ** 2 - (1 - p1) ** 2


def tim_cau_hoi(X, y):
    """Thu moi cot, moi nguong; tra ve cot va nguong lam Gini giam nhieu nhat."""
    nhan = np.asarray(y)
    n = len(nhan)
    g_goc = gini(nhan)
    tot_nhat = (None, None, -1.0)
    tot_moi_cot = {}
    for cot in X.columns:
        gia_tri = X[cot].to_numpy()
        muc = np.sort(np.unique(gia_tri))
        tot_cot = (None, -1.0)
        # Thử mọi ngưỡng nằm giữa hai giá trị liền nhau
        for a, b in zip(muc[:-1], muc[1:]):
            nguong = (a + b) / 2
            di_trai = gia_tri <= nguong
            g = (di_trai.sum() / n * gini(nhan[di_trai])
                 + (~di_trai).sum() / n * gini(nhan[~di_trai]))
            giam = g_goc - g
            if giam > tot_cot[1]:
                tot_cot = (nguong, giam)
            if giam > tot_nhat[2]:
                tot_nhat = (cot, nguong, giam)
        tot_moi_cot[cot] = tot_cot
    return tot_nhat, tot_moi_cot


(cot, nguong, giam), tot_moi_cot = tim_cau_hoi(X_tr, y_tr)
print(f"Gini nut goc cua {len(y_tr)} khach: {gini(np.asarray(y_tr)):.4f}")
print()
print("Cau hoi tot nhat cua tung cot")
print("  cot                   nguong   Gini giam")
for c, (ng, g) in tot_moi_cot.items():
    print(f"  {c:20s} {ng:7.2f}   {g:9.4f}")
print()
print(f"Ham tim_cau_hoi chon: {cot} <= {nguong:.2f}, Gini giam {giam:.4f}")

# So với câu hỏi gốc của cây scikit-learn
cay = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
cot_sk = X.columns[cay.tree_.feature[0]]
nguong_sk = cay.tree_.threshold[0]
print(f"Cay scikit-learn chon: {cot_sk} <= {nguong_sk:.2f}")
if cot == cot_sk and abs(nguong - nguong_sk) < 0.01:
    print("Hai ket qua trung nhau.")
else:
    print("Hai ket qua khac nhau, can kiem tra lai.")

print()
print("Giai thich: ham thu moi cot va moi nguong nam giua hai gia tri lien")
print("nhau, roi giu cau hoi lam Gini sau chia nho nhat. Cot thu_nhap co")
print("muc giam Gini lon nhat nen duoc chon o goc, giong cay scikit-learn.")
print("Cay chi chon cau hoi tot nhat ngay luc do (tham lam), khong nhin truoc.")
