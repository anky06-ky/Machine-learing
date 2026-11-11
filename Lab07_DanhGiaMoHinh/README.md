# Bài 07: Đánh giá mô hình

Bài thực hành theo [Lab07_DanhGiaMoHinh.pdf](./Lab07_DanhGiaMoHinh.pdf), gồm sáu chương trình hướng dẫn tương ứng mục 2-7 và sáu bài tập ở mục 8. Mỗi chương trình chạy độc lập; đường dẫn dữ liệu và hình được xác định theo vị trí file Python.

## Cài đặt

Từ thư mục gốc của repository, tạo môi trường nếu chưa có và cài thư viện:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\Lab07_DanhGiaMoHinh\requirements.txt
```

Nếu đã có `.venv`, dùng lại môi trường đó. Không cần kích hoạt môi trường khi gọi trực tiếp Python như các lệnh dưới đây.

## Chạy phần hướng dẫn

Chạy lần lượt từ thư mục gốc của repository:

```powershell
.\.venv\Scripts\python.exe .\Lab07_DanhGiaMoHinh\h1_do_bua.py
.\.venv\Scripts\python.exe .\Lab07_DanhGiaMoHinh\h2_ma_tran.py
.\.venv\Scripts\python.exe .\Lab07_DanhGiaMoHinh\h3_precision_recall.py
.\.venv\Scripts\python.exe .\Lab07_DanhGiaMoHinh\h4_nguong.py
.\.venv\Scripts\python.exe .\Lab07_DanhGiaMoHinh\h5_roc_auc.py
.\.venv\Scripts\python.exe .\Lab07_DanhGiaMoHinh\h6_can_bang.py
```

1. **`h1_do_bua.py`:** đếm số mẫu mỗi lớp, so hồi quy logistic với mô hình lười luôn đoán lớp đông nhất.
2. **`h2_ma_tran.py`:** tự đếm TP, FP, FN, TN bằng numpy rồi so với `confusion_matrix`.
3. **`h3_precision_recall.py`:** tính precision, recall, F1 bằng tay và bằng thư viện, in `classification_report`.
4. **`h4_nguong.py`:** hạ ngưỡng từ 0.5 xuống 0.1, rồi chọn ngưỡng đúng cách bằng `cross_val_predict` trên tập học.
5. **`h5_roc_auc.py`:** đường ROC, AUC của logistic và so sáu mô hình đã học trong môn.
6. **`h6_can_bang.py`:** chia không phân tầng, kiểm định chéo với nhiều thước đo và `class_weight="balanced"`.

Với scikit-learn 1.9, `h5_roc_auc.py` in một `FutureWarning` vì `SVC(probability=True)` sẽ bị bỏ ở bản 1.11. Đây chỉ là cảnh báo, mã giữ đúng như PDF và kết quả không đổi.

## Chạy bài tập

Sáu bài tập được đặt riêng trong `baitap07` theo yêu cầu của PDF:

```powershell
.\.venv\Scripts\python.exe .\Lab07_DanhGiaMoHinh\baitap07\bai1.py
.\.venv\Scripts\python.exe .\Lab07_DanhGiaMoHinh\baitap07\bai2.py
.\.venv\Scripts\python.exe .\Lab07_DanhGiaMoHinh\baitap07\bai3.py
.\.venv\Scripts\python.exe .\Lab07_DanhGiaMoHinh\baitap07\bai4.py
.\.venv\Scripts\python.exe .\Lab07_DanhGiaMoHinh\baitap07\bai5.py
.\.venv\Scripts\python.exe .\Lab07_DanhGiaMoHinh\baitap07\bai6.py
```

1. **Bài 1:** hàm `danh_gia(y_that, y_doan)` chỉ dùng numpy, chạy cho logistic và mô hình lười, so với scikit-learn.
2. **Bài 2:** `classification_report` cho SVM nhận dạng chữ số của bài 6.
3. **Bài 3:** tìm ngưỡng lớn nhất cho recall kiểm định chéo từ 0.8 trở lên, áp lên tập kiểm tra một lần.
4. **Bài 4:** vẽ đường ROC của logistic và rừng ngẫu nhiên, lưu `baitap07/roc.png`.
5. **Bài 5:** thêm `class_weight="balanced"` cho rừng ngẫu nhiên và SVM.
6. **Bài 6:** MAE, RMSE, R² của hồi quy tuyến tính giá nhà trước và sau khi một dự đoán bị cộng 5 tỷ.

## Dữ liệu và cách đánh giá

- `data/sinh_vien_bo_hoc.csv`: dữ liệu gốc của bài, 1200 sinh viên, 119 bạn bỏ học (tỷ lệ 0.099).
- `data/gia_nha.csv`: dữ liệu giá nhà của bài 1, dùng cho bài tập 6.
- Chữ số viết tay: `load_digits()` có sẵn trong scikit-learn.
- Phân loại chia 75% học và 25% kiểm tra, `random_state=17`, phân tầng theo nhãn. Giá nhà chia 80/20 với `random_state=42` như bài 1.
- Chuẩn hoá nằm trong pipeline, ngưỡng được chọn bằng kiểm định chéo trên tập học; tập kiểm tra chỉ dùng để chấm một lần.

## Giải thích kết quả

Sáu chương trình hướng dẫn cho **đúng mọi con số** trong PDF:

- `h1`: mô hình lười accuracy **0.9000**, tìm ra 0/30 bạn bỏ học; logistic **0.9467**, tìm ra **17/30**.
- `h2`: TP = **17**, FP = **3**, FN = **13**, TN = **267**.
- `h3`: precision **0.8500**, recall **0.5667**, F1 **0.6800**.
- `h4`: kiểm định chéo chọn ngưỡng **0.25**; trên tập kiểm tra precision **0.6571**, recall **0.7667**, F1 **0.7077**.
- `h5`: AUC logistic **0.9449**, cao nhất trong sáu mô hình; KNN có AUC 0.9274 nhưng recall ở ngưỡng 0.5 chỉ 0.2333.
- `h6`: recall kiểm định chéo **0.4810**; cân bằng lớp đưa recall từ **0.5667** lên **0.8667**, precision từ 0.8500 xuống **0.4483**.

Bài tập:

- **Bài 1:** hàm tự viết khớp scikit-learn: logistic 0.9467 / 0.8500 / 0.5667 / 0.6800, mô hình lười 0.9000 / 0 / 0 / 0. Mô hình lười không báo bạn nào nên TP + FP = 0, precision = 0/0 **không tính được**. Hàm kiểm tra mẫu số trước khi chia và quy ước trả 0, giống `zero_division=0`.
- **Bài 2:** SVM đạt accuracy **0.9911**. Chữ số **8** vừa có recall thấp nhất vừa có precision thấp nhất, cùng bằng **0.9767**. Support của 10 chữ số chỉ từ 43 tới 46 ảnh, gần như bằng nhau, nên macro avg (trung bình cộng) và weighted avg (trung bình theo support) gần như trùng nhau.
- **Bài 3:** trên lưới ngưỡng của mục 5, ngưỡng lớn nhất còn recall kiểm định chéo ≥ 0.8 là **0.15** (recall 0.8315; ở 0.20 chỉ còn 0.7865). Áp một lần lên tập kiểm tra: recall **0.8333**, **đạt** yêu cầu; precision **0.5102**; báo **49** bạn, trong đó 25 bạn bỏ học thật. Nếu không giới hạn trong lưới, ngưỡng lớn nhất là 0.1718; chương trình chỉ in để tham khảo, vẫn dùng ngưỡng lưới như mục 5.
- **Bài 4:** AUC logistic **0.9449**, rừng **0.9249**; đường logistic ôm sát góc trên bên trái hơn ở phần lớn trục. Hai đường **có cắt nhau**: rừng nằm trên ở đoạn rất đầu (FPR ≈ 0.004-0.007, recall 0.5333 so với 0.4000) và ở đoạn FPR ≈ 0.47-0.73.
- **Bài 5:** rừng: recall **0.5667 → 0.7000**, precision 0.7727 → 0.5385. SVM: recall **0.5000 → 0.8333**, precision 0.8824 → 0.4464. **SVM thay đổi nhiều nhất**, số bạn bị báo tăng từ 17 lên 56. F1 của cả hai đều giảm nhẹ.
- **Bài 6:** một dự đoán hỏng 5 tỷ làm MAE tăng **0.2578 → 0.6745** (gấp 2.62 lần), RMSE **0.3730 → 1.5505** (gấp 4.16 lần), R² giảm **0.9622 → 0.3475**. MAE chỉ cộng thêm 5/12 tỷ vì sai số được chia đều cho 12 căn; RMSE bình phương sai số trước khi lấy trung bình nên 5 tỷ thành 25 và lấn át mọi sai số nhỏ khác.

Nhận xét:

- Với dữ liệu lệch lớp, accuracy của mô hình lười là mức sàn. Phải đọc recall và precision của lớp hiếm.
- Hạ ngưỡng và cân bằng lớp cùng đẩy mô hình về phía báo nhiều hơn, ít sót hơn; đổi lại precision giảm. Mọi lựa chọn ngưỡng hay tham số phải dựa trên tập học, không dựa trên tập kiểm tra.
- AUC đo khả năng xếp hạng tổng thể, không bảo đảm mô hình tốt hơn ở mọi ngưỡng; bài 4 cho thấy hai đường ROC vẫn có thể cắt nhau.
