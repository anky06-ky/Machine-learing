# -*- coding: utf-8 -*-
"""Bài tập 6: phân cụm chữ số viết tay, chỉ dùng nhãn sau khi đã chia cụm."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits

X, nhan_that = load_digits(return_X_y=True)

# Học không dùng nhãn
cum = KMeans(n_clusters=10, n_init=10, random_state=0).fit_predict(X)

# Xong mới dùng nhãn thật để xem mỗi cụm chứa những chữ số nào
bang = pd.crosstab(pd.Series(nhan_that, name="nhan_that"),
                   pd.Series(cum, name="cum"))
print(bang.to_string())
print()

print("Moi chu so: cum chua nhieu nhat va ty le")
ty_le = (bang.max(axis=1) / bang.sum(axis=1)).sort_values(ascending=False)
for so in ty_le.index:
    print(f"  chu so {so}: {bang.loc[so].max():3d}/{bang.loc[so].sum()}"
          f" vao cum {bang.loc[so].idxmax()}, {ty_le[so]:.1%}")
print()

print("Cum chua nhieu chu so (moi chu so chiem it nhat 20 anh):")
for c in bang.columns:
    cot = bang[c][bang[c] >= 20].sort_values(ascending=False)
    if len(cot) >= 2:
        print(f"  cum {c}: " + ", ".join(f"so {s} ({n} anh)" for s, n in cot.items()))
