# -*- coding: utf-8 -*-
"""Bài tập 6: so sánh thời gian học SVC và LinearSVC."""

from time import perf_counter

from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC, SVC

print("     n   SVC (s)   LinearSVC (s)   n/n_truoc   SVC lan   LinearSVC lan")
truoc = None
for n in [2000, 5000, 10000, 20000]:
    X, y = make_classification(n_samples=n, n_features=20, random_state=0)
    thoi_gian = []
    for mh in [SVC(), LinearSVC()]:
        bat_dau = perf_counter()
        mh.fit(X, y)
        thoi_gian.append(perf_counter() - bat_dau)
    svc_s, linear_s = thoi_gian
    if truoc is None:
        ty_le = "       -         -            -"
    else:
        n_cu, svc_cu, linear_cu = truoc
        ty_le = f"    {n / n_cu:5.2f}     {svc_s / svc_cu:7.2f}      {linear_s / linear_cu:7.2f}"
    print(f"{n:6d}   {svc_s:7.4f}   {linear_s:13.4f}   {ty_le}")
    if n == 5000:
        svc_5000, linear_5000 = svc_s, linear_s
    truoc = (n, svc_s, linear_s)

print()
print("Nhan xet: du lieu tang gap doi 2 lan (5000 -> 10000 -> 20000), thoi gian hoc")
print("trung binh moi lan gap doi tang:")
print(f"  SVC       : {(svc_s / svc_5000) ** 0.5:.2f} lan")
print(f"  LinearSVC : {(linear_s / linear_5000) ** 0.5:.2f} lan")
print("SVC mac dinh dung RBF, LinearSVC dung mo hinh tuyen tinh; hai mo hinh khac nhau.")
print("Thoi gian chi la phep do tren may hien tai, phu thuoc du lieu va tai he thong.")
