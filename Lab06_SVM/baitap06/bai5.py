# -*- coding: utf-8 -*-
"""Bài tập 5: Chữ số viết tay, những lỗi nào hay gặp."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

svm = SVC(kernel="rbf").fit(X_tr, y_tr)
du_doan = svm.predict(X_te)

cm = confusion_matrix(y_te, du_doan)
print("Bang nham lan (Confusion Matrix 10x10):")
print(cm)
print()

# Tìm và vẽ các ảnh bị đoán sai
sai = np.flatnonzero(du_doan != y_te)
anh_dir = Path(__file__).resolve().parent / "anh_sai"
anh_dir.mkdir(parents=True, exist_ok=True)

print(f"Co {len(sai)} anh bi doan sai:")
for i in sai:
    print(f"  Anh {i}: That la {y_te[i]}, Doan la {du_doan[i]}")
    plt.figure(figsize=(2, 2))
    plt.imshow(X_te[i].reshape(8, 8), cmap="gray_r")
    plt.title(f"That: {y_te[i]} - Doan: {du_doan[i]}")
    plt.axis("off")
    plt.savefig(anh_dir / f"anh_{i}_that_{y_te[i]}_doan_{du_doan[i]}.png", bbox_inches='tight')
    plt.close()

print("Da luu cac anh bi doan sai vao thu muc baitap06/anh_sai/")
print()
print("Nhan xet khi xem cac anh: anh 8x8 co do phan giai thap, net viet thieu ro rang.")
print("Chu 9 co the giong 3, chu 8 giong 1, chu 4 giong 8 va chu 7 giong 9.")
