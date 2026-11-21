# -*- coding: utf-8 -*-
"""Bài tập 2: classification_report của SVM nhận dạng chữ số (bài 6)."""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Đúng mô hình của bài 6: SVC mặc định, kernel RBF, không chuẩn hoá
du = SVC().fit(X_tr, y_tr).predict(X_te)
print(classification_report(y_te, du, digits=4))

bao_cao = classification_report(y_te, du, output_dict=True)
for thuoc_do in ["recall", "precision"]:
    gia_tri = {c: bao_cao[str(c)][thuoc_do] for c in range(10)}
    thap = min(gia_tri.values())
    cac_so = [c for c, v in gia_tri.items() if np.isclose(v, thap)]
    print(f"{thuoc_do:9s} thap nhat: chu so {cac_so}, bang {thap:.4f}")

so_mau = [int(bao_cao[str(c)]["support"]) for c in range(10)]
print("Support cua 10 chu so:", so_mau)
print()
print("Nhan xet:")
print("- Macro avg lay trung binh cong 10 chu so, moi chu so nang nhu nhau.")
print("- Weighted avg lay trung binh theo support. O day support chi tu "
      f"{min(so_mau)} toi {max(so_mau)},")
print("  gan nhu bang nhau, nen hai cach lay trung binh cho gan cung ket qua.")
