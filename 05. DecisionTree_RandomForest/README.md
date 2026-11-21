# Bài 05: Cây quyết định và rừng ngẫu nhiên

Bài thực hành theo [DecisionTree_RandomForest.pdf](./DecisionTree_RandomForest.pdf), gồm bảy chương trình hướng dẫn tương ứng mục 2-8 và sáu bài tập ở mục 9. Mỗi chương trình chạy độc lập; đường dẫn dữ liệu và ảnh được xác định theo vị trí file Python.

## Cấu trúc thư mục

```text
05. DecisionTree_RandomForest/
├── README.md
├── DecisionTree_RandomForest.pdf
├── requirements.txt
├── f1_gini.py
├── f2_cay_nam_la.py
├── f3_do_sau.py
├── f4_khong_chuan_hoa.py
├── f5_rung.py
├── f6_do_quan_trong.py
├── f7_cay_hoi_quy.py
├── baitap05/
│   ├── bai1.py ... bai6.py
│   └── cay_nam_la.png, cay_sau_2.png   (do bai1.py sinh ra)
└── data/
    ├── khach_vay.csv
    └── gia_nha.csv
```

## Cài đặt

Từ thư mục gốc của repository, tạo môi trường nếu chưa có và cài thư viện:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ".\05. DecisionTree_RandomForest\requirements.txt"
```

Nếu đã có `.venv`, dùng lại môi trường đó. Không cần kích hoạt môi trường khi gọi trực tiếp Python như các lệnh dưới đây.

## Chạy phần hướng dẫn

Chạy lần lượt từ thư mục gốc của repository:

```powershell
.\.venv\Scripts\python.exe ".\05. DecisionTree_RandomForest\f1_gini.py"
.\.venv\Scripts\python.exe ".\05. DecisionTree_RandomForest\f2_cay_nam_la.py"
.\.venv\Scripts\python.exe ".\05. DecisionTree_RandomForest\f3_do_sau.py"
.\.venv\Scripts\python.exe ".\05. DecisionTree_RandomForest\f4_khong_chuan_hoa.py"
.\.venv\Scripts\python.exe ".\05. DecisionTree_RandomForest\f5_rung.py"
.\.venv\Scripts\python.exe ".\05. DecisionTree_RandomForest\f6_do_quan_trong.py"
.\.venv\Scripts\python.exe ".\05. DecisionTree_RandomForest\f7_cay_hoi_quy.py"
```

1. **`f1_gini.py`:** dùng 10 khách vay để tự tính Gini nút gốc và mức giảm Gini tại từng ngưỡng thu nhập, so với câu hỏi về việc làm, rồi đối chiếu với cây một tầng của scikit-learn.
2. **`f2_cay_nam_la.py`:** học cây tối đa 5 lá trên 450 hồ sơ, in cây bằng `export_text`, đo accuracy và dự đoán cho hai khách mới.
3. **`f3_do_sau.py`:** cho cây mọc tự do, rồi chọn độ sâu từ 1 đến 15 bằng kiểm định chéo 5 phần và chấm một lần trên tập kiểm tra.
4. **`f4_khong_chuan_hoa.py`:** so cây sâu 5 trên dữ liệu gốc, dữ liệu chuẩn hoá và dữ liệu đổi thu nhập sang nghìn đồng.
5. **`f5_rung.py`:** minh hoạ bốc mẫu có hoàn lại, rừng 200 cây, khảo sát số cây và bảng so ba mô hình.
6. **`f6_do_quan_trong.py`:** đọc `feature_importances_`, thêm một cột ngẫu nhiên để thấy giới hạn của độ quan trọng.
7. **`f7_cay_hoi_quy.py`:** cây hồi quy sâu 3 trên dữ liệu giá nhà, so MAE với đường thẳng và dự đoán bốn căn mới.

## Chạy bài tập

Sáu bài tập được đặt trong `baitap05`, đường dẫn dữ liệu vẫn tìm theo vị trí file nên chạy từ thư mục nào cũng được:

```powershell
.\.venv\Scripts\python.exe ".\05. DecisionTree_RandomForest\baitap05\bai1.py"
.\.venv\Scripts\python.exe ".\05. DecisionTree_RandomForest\baitap05\bai2.py"
.\.venv\Scripts\python.exe ".\05. DecisionTree_RandomForest\baitap05\bai3.py"
.\.venv\Scripts\python.exe ".\05. DecisionTree_RandomForest\baitap05\bai4.py"
.\.venv\Scripts\python.exe ".\05. DecisionTree_RandomForest\baitap05\bai5.py"
.\.venv\Scripts\python.exe ".\05. DecisionTree_RandomForest\baitap05\bai6.py"
```

Cũng có thể chuyển vào thư mục bài rồi dùng `..\.venv\Scripts\python.exe .\f3_do_sau.py` hoặc `..\.venv\Scripts\python.exe .\baitap05\bai3.py`.

1. **Bài 1:** vẽ cây năm lá và cây sâu 2 bằng `plot_tree`, lưu `cay_nam_la.png` và `cay_sau_2.png` trong `baitap05`; so kết luận của hai lá bên trái cây sâu 2 và giải thích vì sao cây vẫn chia nút đó.
2. **Bài 2:** viết hàm `tim_cau_hoi(X, y)` thử mọi cột, mọi ngưỡng trên 450 khách của tập học; kết quả trùng câu hỏi gốc của cây scikit-learn (thu nhập, ngưỡng 15.05).
3. **Bài 3:** chạy lại bài chọn độ sâu với `criterion="entropy"` và so với Gini về độ sâu tốt nhất, điểm kiểm định chéo và accuracy kiểm tra.
4. **Bài 4:** cắt tỉa bằng `min_samples_leaf` trong 1, 2, 5, 10, 20, 40 (kiểm định chéo 5 phần), chấm một lần trên tập kiểm tra và so với cây sâu 5.
5. **Bài 5:** thêm cột `so_ngau_nhien` vào cả tập học và tập kiểm tra, dùng `permutation_importance` với `n_repeats=20`, so thứ hạng của cột ngẫu nhiên với `feature_importances_`.
6. **Bài 6:** `RandomForestRegressor` 200 cây trên dữ liệu giá nhà, so MAE với cây sâu 3 và đường thẳng, dự đoán căn 250 m2 và xem dự đoán của từng cây để giải thích vì sao rừng không ngoại suy.

## Dữ liệu và cách đánh giá

- `data/khach_vay.csv`: 600 khách vay mô phỏng; đặc trưng `thu_nhap`, `ty_le_no`, `co_viec_on_dinh`, `so_nguoi_phu_thuoc`; nhãn `tra_dung_han` bằng 1 khi trả đúng hạn, bằng 0 khi trễ hạn.
- `data/gia_nha.csv`: dữ liệu giá nhà của bài 1; bài này dùng `dien_tich` để đoán `gia` (tỷ đồng).
- Phân loại: chia 75% học, 25% kiểm tra, `random_state=17`, `stratify=y`.
- Hồi quy: chia 80% học, 20% kiểm tra, `random_state=42`.
- Mọi cây và rừng dùng `random_state=0`.

## Kết quả mẫu trong PDF

- Mười khách: Gini nút gốc 0.48; ngưỡng tốt nhất 12.0 làm Gini giảm 0.1633; câu hỏi về việc làm chỉ giảm 0.0610.
- Cây năm lá: accuracy tập học 0.7978, tập kiểm tra 0.7467.
- Cây mọc tự do: sâu 13, 92 lá, tập học 1.0000, tập kiểm tra 0.7667.
- Kiểm định chéo chọn độ sâu 5 (điểm 0.8333), 27 lá, tập kiểm tra 0.8533.
- Ba cây ở `f4` cùng accuracy 0.8533 và không có dự đoán nào khác nhau.
- Rừng 200 cây: tập học 1.0000, tập kiểm tra 0.9000.
- Độ quan trọng: thu nhập 0.479, tỷ lệ nợ 0.339; cột ngẫu nhiên được 0.150.
- Giá nhà: MAE cây sâu 3 là 0.5039, đường thẳng là 0.2578 (tỷ đồng).
