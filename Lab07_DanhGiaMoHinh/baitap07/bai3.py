"""Bài 3: ngưỡng lớn nhất có recall OOF >= 0.8, không dùng test để chọn."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import split_data
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_tr, X_te, y_tr, y_te = split_data()
model = make_pipeline(StandardScaler(), LogisticRegression())
prob_oof = cross_val_predict(model, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
# Với phép >=, ngưỡng tối đa là điểm của mẫu dương thứ ceil(0.8*n_pos).
# Cách này xét chính xác các bước nhảy recall, không làm tròn vào lưới 0.05.
positive_scores = np.sort(prob_oof[y_tr.to_numpy() == 1])[::-1]
required = int(np.ceil(0.8 * len(positive_scores)))
threshold = positive_scores[required - 1]
print(f"Nguong lon nhat: {threshold:.10f}; recall OOF={recall_score(y_tr, prob_oof >= threshold):.4f}")
du = model.fit(X_tr, y_tr).predict_proba(X_te)[:, 1] >= threshold
recall = recall_score(y_te, du)
print(f"Test: recall={recall:.4f}, precision={precision_score(y_te, du, zero_division=0):.4f}, "
      f"F1={f1_score(y_te, du):.4f}, so ban bi bao={du.sum()}")
print("Recall test dat 0.8:", recall >= 0.8)
print("Dieu kien 80% tren OOF khong bao dam 80% tren du lieu moi; khong sua nguong theo test.")
