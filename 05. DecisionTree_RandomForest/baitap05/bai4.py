# -*- coding: utf-8 -*-
"""Bài tập 4: cắt tỉa bằng min_samples_leaf, so với cây sâu 5."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Không giới hạn độ sâu, chỉ đổi số khách tối thiểu ở lá
print("min_samples_leaf   so la   tap hoc   kiem dinh cheo")
diem = {}
for m in [1, 2, 5, 10, 20, 40]:
    mh = DecisionTreeClassifier(min_samples_leaf=m, random_state=0)
    diem[m] = cross_val_score(mh, X_tr, y_tr, cv=5).mean()
    mh.fit(X_tr, y_tr)
    hoc = accuracy_score(y_tr, mh.predict(X_tr))
    print(f"   {m:14d}   {mh.get_n_leaves():5d}   {hoc:.4f}      {diem[m]:.4f}")

m_tot = max(diem, key=diem.get)
print()

# Chọn xong mới mở tập kiểm tra, và chỉ mở một lần
tot = DecisionTreeClassifier(min_samples_leaf=m_tot, random_state=0)
tot.fit(X_tr, y_tr)
acc_la = accuracy_score(y_te, tot.predict(X_te))
print(f"min_samples_leaf tot nhat = {m_tot}, cay co {tot.get_n_leaves()} la")
print(f"Diem kiem dinh cheo = {diem[m_tot]:.4f}")
print(f"Accuracy tren tap kiem tra = {acc_la:.4f}")
print()

# Cây sâu 5 của mục 4 để so sánh
sau5 = DecisionTreeClassifier(max_depth=5, random_state=0)
cv_sau5 = cross_val_score(sau5, X_tr, y_tr, cv=5).mean()
sau5.fit(X_tr, y_tr)
acc_sau5 = accuracy_score(y_te, sau5.predict(X_te))
print("So sanh hai cach cat tia")
print("  cach                      kiem dinh cheo   tap kiem tra")
print(f"  min_samples_leaf = {m_tot:<3d}    {diem[m_tot]:14.4f}   {acc_la:12.4f}")
print(f"  max_depth = 5             {cv_sau5:14.4f}   {acc_sau5:12.4f}")
print()

if acc_la > acc_sau5:
    print("Nhan xet: min_samples_leaf cho accuracy kiem tra cao hon cay sau 5.")
elif acc_la < acc_sau5:
    print("Nhan xet: cay sau 5 cho accuracy kiem tra cao hon min_samples_leaf.")
else:
    print("Nhan xet: hai cach cat tia cho accuracy kiem tra bang nhau.")
if diem[m_tot] > cv_sau5:
    print("Theo kiem dinh cheo, min_samples_leaf cung cho diem cao hon.")
elif diem[m_tot] < cv_sau5:
    print("Theo kiem dinh cheo, cay sau 5 cho diem cao hon.")
else:
    print("Theo kiem dinh cheo, hai cach co diem bang nhau.")
