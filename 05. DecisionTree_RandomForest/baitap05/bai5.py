# -*- coding: utf-8 -*-
"""Bài tập 5: độ quan trọng bằng cách xáo trộn (permutation_importance)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "khach_vay.csv")
X = df.drop(columns="tra_dung_han")
y = df["tra_dung_han"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Rừng có cột ngẫu nhiên như mục 7; tập kiểm tra cũng thêm một cột tương tự
rng = np.random.default_rng(0)
X_tr_nn = X_tr.copy()
X_tr_nn["so_ngau_nhien"] = rng.random(len(X_tr))
X_te_nn = X_te.copy()
X_te_nn["so_ngau_nhien"] = rng.random(len(X_te))

rung = RandomForestClassifier(n_estimators=200, random_state=0)
rung.fit(X_tr_nn, y_tr)

# Cách 1: độ quan trọng theo Gini, tính trên tập học
qt = pd.Series(rung.feature_importances_, index=X_tr_nn.columns)
qt = qt.sort_values(ascending=False)
print("feature_importances_ (tinh tren tap hoc)")
print(qt.round(3).to_string())
hang_gini = list(qt.index).index("so_ngau_nhien") + 1
print()

# Cách 2: xáo trộn từng cột trên tập kiểm tra, xem accuracy giảm bao nhiêu
kq = permutation_importance(rung, X_te_nn, y_te, n_repeats=20, random_state=0)
xao = pd.Series(kq.importances_mean, index=X_te_nn.columns)
xao = xao.sort_values(ascending=False)
print("permutation_importance (trung binh 20 lan, tap kiem tra)")
print(xao.round(4).to_string())
hang_xao = list(xao.index).index("so_ngau_nhien") + 1
print()

print(f"Cot so_ngau_nhien xep thu {hang_xao} theo cach xao tron "
      f"({xao['so_ngau_nhien']:.4f}).")
print(f"Cot so_ngau_nhien xep thu {hang_gini} theo feature_importances_ "
      f"({qt['so_ngau_nhien']:.3f}).")
print()
print("Giai thich:")
print("- feature_importances_ cong muc giam Gini khi cot duoc dung de chia,")
print("  tinh tren tap hoc. Cot ngau nhien co 450 gia tri khac nhau nen cay")
print("  luon tim duoc nguong tach vai khach, va cot duoc cong diem du khong")
print("  lien quan toi viec tra no.")
print("- Xao tron lam do tren tap kiem tra: xao cot nao thi mat moi lien he")
print("  cua cot do voi nhan, roi xem accuracy giam bao nhieu. Cot ngau nhien")
print("  khong giup gi cho khach moi nen xao tron gan nhu khong lam mo hinh")
print("  kem di, do quan trong gan 0 va xep thap.")
