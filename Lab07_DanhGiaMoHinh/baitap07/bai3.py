# -*- coding: utf-8 -*-
"""Bài tập 3: chọn ngưỡng lớn nhất cho recall từ 0.8 trở lên.

Ngưỡng chỉ được chọn trên tập học, bằng xác suất từ cross_val_predict
như mục 5. Tập kiểm tra chỉ dùng một lần sau khi đã chọn xong.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

df = pd.read_csv(DATA_DIR / "sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

lr = make_pipeline(StandardScaler(), LogisticRegression())

# Mỗi bạn trong tập học nhận xác suất từ mô hình chưa từng thấy bạn ấy
xs_hoc = cross_val_predict(lr, X_tr, y_tr, cv=5,
                           method="predict_proba")[:, 1]

# Cùng lưới ngưỡng như mục 5, xét từ lớn xuống nhỏ
cac_nguong = np.round(np.arange(0.05, 0.96, 0.05), 2)
print("Tap hoc, kiem dinh cheo 5 phan")
print("  nguong  so bao  recall  precision")
nguong_chon = None
for t in cac_nguong[::-1]:
    du_hoc = (xs_hoc >= t).astype(int)
    r = recall_score(y_tr, du_hoc)
    p = precision_score(y_tr, du_hoc, zero_division=0)
    print(f"  {t:.2f}    {du_hoc.sum():5d}  {r:.4f}  {p:.4f}")
    if nguong_chon is None and r >= 0.8:
        nguong_chon = t
print()
print(f"Nguong lon nhat co recall kiem dinh cheo >= 0.8: {nguong_chon}")

# Kiểm tra thêm: ngưỡng lớn nhất thật sự, không giới hạn trong lưới 0.05.
# Recall chỉ đổi khi ngưỡng vượt qua xác suất của một bạn bỏ học, nên
# ngưỡng lớn nhất là xác suất của bạn bỏ học thứ ceil(0.8 * số bạn bỏ học).
xs_bo_hoc = np.sort(xs_hoc[y_tr.to_numpy() == 1])[::-1]
can_tim = int(np.ceil(0.8 * len(xs_bo_hoc)))
print(f"Tap hoc co {len(xs_bo_hoc)} ban bo hoc, can tim it nhat {can_tim} ban;"
      f" nguong lon nhat khong theo luoi = {xs_bo_hoc[can_tim - 1]:.4f}")
print()

# Chọn xong mới mở tập kiểm tra, và chỉ một lần
xs_te = lr.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
du = (xs_te >= nguong_chon).astype(int)
r_te = recall_score(y_te, du)
tp = int(((du == 1) & (y_te == 1)).sum())
print(f"Tap kiem tra voi nguong {nguong_chon}:")
print(f"  so ban bi bao: {du.sum()} tren {len(du)}, trong do {tp} ban"
      f" bo hoc that tren {y_te.sum()}")
print(f"  recall {r_te:.4f}, precision {precision_score(y_te, du):.4f},"
      f" F1 {f1_score(y_te, du):.4f}")
print("  Recall dat 0.8:", "co" if r_te >= 0.8 else "khong")
print()
print("Nhan xet:")
print("- Ha nguong thi bao nhieu hon, recall tang, precision giam. Muon recall")
print("  >= 0.8 ma bao it nhat co the thi lay nguong LON NHAT con dat 0.8.")
print("- Nguong chon tren xac suat kiem dinh cheo cua tap hoc, khong nhin tap")
print("  kiem tra. Recall tren tap kiem tra la con so trung thuc, co the cao")
print("  hon hoac thap hon 0.8 vi tap kiem tra chi co 30 ban bo hoc.")
