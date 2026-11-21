# -*- coding: utf-8 -*-
"""Bài tập 2: từ nào tố cáo tin rác mạnh nhất."""

from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data_path = Path(__file__).resolve().parent.parent / "data" / "tin_nhan.csv"
df = pd.read_csv(data_path)
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"])
y = df["la_rac"]

# Huấn luyện trên toàn bộ 300 tin nhắn
mo_hinh = MultinomialNB(alpha=1.0).fit(X, y)

# Dòng 0 là lớp thường, dòng 1 là lớp rác; hiệu càng lớn càng nghiêng về rác
chenh_lech = mo_hinh.feature_log_prob_[1] - mo_hinh.feature_log_prob_[0]
bang = pd.Series(chenh_lech, index=vec.get_feature_names_out())
bang = bang.sort_values(ascending=False)

print("5 tu to cao tin rac manh nhat:")
for tu, d in bang.head(5).items():
    print(f"  {tu:10s} {d:+.4f}")
print()
print("5 tu cho thay tin thuong manh nhat:")
for tu, d in bang.tail(5)[::-1].items():
    print(f"  {tu:10s} {d:+.4f}")
