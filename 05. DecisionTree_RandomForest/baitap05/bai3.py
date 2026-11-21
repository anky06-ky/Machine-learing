# -*- coding: utf-8 -*-
"""Bài tập 3: chạy lại f3_do_sau.py với criterion="entropy", so với Gini."""

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


def chon_do_sau(criterion, in_bang):
    """Chon do sau bang kiem dinh cheo, roi cham mot lan tren tap kiem tra."""
    if in_bang:
        print("Do sau   tap hoc   kiem dinh cheo")
    diem = {}
    for d in range(1, 16):
        mh = DecisionTreeClassifier(max_depth=d, criterion=criterion,
                                    random_state=0)
        diem[d] = cross_val_score(mh, X_tr, y_tr, cv=5).mean()
        hoc = accuracy_score(y_tr, mh.fit(X_tr, y_tr).predict(X_tr))
        if in_bang:
            print(f"   {d:2d}     {hoc:.4f}      {diem[d]:.4f}")
    d_tot = max(diem, key=diem.get)
    tot = DecisionTreeClassifier(max_depth=d_tot, criterion=criterion,
                                 random_state=0).fit(X_tr, y_tr)
    acc = accuracy_score(y_te, tot.predict(X_te))
    return d_tot, diem[d_tot], tot.get_n_leaves(), acc


print("Criterion = entropy")
d_e, cv_e, la_e, acc_e = chon_do_sau("entropy", True)
print()
d_g, cv_g, la_g, acc_g = chon_do_sau("gini", False)

print("Tong ket")
print("  criterion   do sau tot nhat   so la   kiem dinh cheo   tap kiem tra")
print(f"  gini        {d_g:15d}   {la_g:5d}   {cv_g:14.4f}   {acc_g:12.4f}")
print(f"  entropy     {d_e:15d}   {la_e:5d}   {cv_e:14.4f}   {acc_e:12.4f}")
print()

if abs(acc_e - acc_g) <= 0.02:
    print("Nhan xet: doi tu Gini sang entropy gan nhu khong lam ket qua khac")
    print("di nhieu, accuracy tren tap kiem tra chenh nhau rat it.")
else:
    print("Nhan xet: doi thuoc do lam accuracy tren tap kiem tra khac di")
    print("dang ke tren tap kiem tra nho nay.")
print("Ca hai deu do muc lan lon cua nut va nho nhat khi nut thuan khiet,")
print("nen cay hoc ra thuong gan giong nhau.")
