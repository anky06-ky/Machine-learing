"""Bài 6: học KMeans không nhãn rồi mới lập crosstab bằng nhãn thật."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans

X, y = load_digits(return_X_y=True)
km = KMeans(n_clusters=10, n_init=10, random_state=0).fit(X)
table = pd.crosstab(pd.Series(y, name="nhan_that"), pd.Series(km.labels_, name="cum"))
print(table.to_string())
fractions = table.max(axis=1)/table.sum(axis=1)
print("Ty le nam trong cum dong nhat cua moi chu so:\n", fractions.round(4).to_string())
best = fractions.max()
print("Gom gon nhat:", fractions.index[np.isclose(fractions, best)].tolist())
for cluster in table.columns:
    top = table[cluster].nlargest(2)
    print(f"Cum {cluster}: hai chu so dong nhat {top.to_dict()}")
print("Nhom hinh dang giong nhau co the bi gom chung; cum khong duoc danh so theo chu so.")
