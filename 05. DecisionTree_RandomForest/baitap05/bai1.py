# -*- coding: utf-8 -*-
"""Bài tập 1: tự vẽ cây năm lá bằng plot_tree, rồi vẽ cây sâu 2."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

thu_muc = Path(__file__).resolve().parent
df = pd.read_csv(thu_muc.parent / "data" / "khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
ten_lop = ["tre han", "dung han"]


def ve_cay(cay, ten_anh):
    plt.figure(figsize=(14, 7))
    plot_tree(cay, feature_names=list(X.columns), class_names=ten_lop,
              filled=True, fontsize=9)
    plt.savefig(thu_muc / ten_anh, dpi=120, bbox_inches="tight")
    plt.close()


# Cây năm lá của mục 3
cay5 = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0).fit(X_tr, y_tr)
ve_cay(cay5, "cay_nam_la.png")
print("Da luu cay_nam_la.png")

# Cây sâu 2
cay2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_tr, y_tr)
ve_cay(cay2, "cay_sau_2.png")
print("Da luu cay_sau_2.png")
print()

# Xem hai lá bên trái của cây sâu 2
t = cay2.tree_
trai = t.children_left[0]
print(f"Nut ben trai (cha cua hai la): gini {t.impurity[trai]:.3f}")
print("Hai la ben trai cua cay sau 2")
ket_luan = []
for la in (t.children_left[trai], t.children_right[trai]):
    so_khach = t.value[la][0]
    lop = int(np.argmax(so_khach))
    ket_luan.append(lop)
    print(f"  gini {t.impurity[la]:.3f}, value {so_khach}, "
          f"ket luan: {ten_lop[lop]}")
print()

if ket_luan[0] == ket_luan[1]:
    print("Hai la ben trai co cung ket luan.")
    print("Cay van chia nut do vi muc tieu cua cay la lam Gini giam, khong")
    print("phai doi ket luan. Sau khi chia, moi la thuan hon nut cha (ti le")
    print("khach dung han o hai la khac nhau ro), du da so van cung mot lop.")
else:
    print("Hai la ben trai co ket luan khac nhau.")
