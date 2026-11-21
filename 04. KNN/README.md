# Bài 04: K láng giềng gần nhất (KNN)

Thư mục này gồm năm chương trình hướng dẫn `e1` đến `e5` của Lab04 (đúng Bảng 1 trong `KNN.pdf`) và sáu bài tập trong `baitap04/`. Mỗi tệp chạy độc lập và đọc dữ liệu trong `data/` theo vị trí của tệp, nên chạy từ thư mục làm việc nào cũng được. Chú thích dùng tiếng Việt có dấu; terminal in chữ không dấu để tương thích Windows PowerShell.

## Cấu trúc thư mục

```text
04. KNN/
├── README.md
├── KNN.pdf                   # Tài liệu thực hành và bài tập
├── requirements.txt          # numpy, pandas, scikit-learn
├── data/
│   ├── gia_nha.csv           # Dữ liệu giá nhà, dùng cho hồi quy
│   ├── hai_vang_trang.csv    # 300 điểm hai vầng trăng
│   └── sinh_vien_mo_rong.csv # Sinh viên có thêm cột chi tiêu
├── e1_khoang_cach.py         # Mục 2: tự tính khoảng cách và tìm láng giềng
├── e2_chuan_hoa.py           # Mục 3: hậu quả của việc quên chuẩn hoá
├── e3_chon_k.py              # Mục 4: chọn k bằng kiểm định chéo
├── e4_so_sanh.py             # Mục 5: so KNN với logistic và Naive Bayes
├── e5_knn_hoi_quy.py         # Mục 6: KNN cho bài toán hồi quy
└── baitap04/
    ├── bai1.py               # Tự viết KNN không dùng thư viện
    ├── bai2.py               # Khoảng cách Manhattan
    ├── bai3.py               # Đổi đơn vị đo cột chi tiêu
    ├── bai4.py               # Láng giềng có trọng số
    ├── bai5.py               # Chọn k cho KNN hồi quy
    └── bai6.py               # Đo thời gian dự đoán
```

## Cài đặt và chạy

Từ thư mục gốc repository, tạo môi trường và cài thư viện:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r ".\04. KNN\requirements.txt"
```

Chương trình hướng dẫn, chạy theo thứ tự trong Bảng 1:

```powershell
python ".\04. KNN\e1_khoang_cach.py"
python ".\04. KNN\e2_chuan_hoa.py"
python ".\04. KNN\e3_chon_k.py"
python ".\04. KNN\e4_so_sanh.py"
python ".\04. KNN\e5_knn_hoi_quy.py"
```

Bài tập:

```powershell
python ".\04. KNN\baitap04\bai1.py"
python ".\04. KNN\baitap04\bai2.py"
python ".\04. KNN\baitap04\bai3.py"
python ".\04. KNN\baitap04\bai4.py"
python ".\04. KNN\baitap04\bai5.py"
python ".\04. KNN\baitap04\bai6.py"
```

Chụp màn hình phần kết quả để nộp cùng mã nguồn.

## Kết quả chương trình hướng dẫn

| Đại lượng | Giá trị |
| :--- | :---: |
| d(A, B) và d(A, C) khi chưa chuẩn hoá | 102.2 và 3000.0 |
| d(A, B) và d(A, C) khi đã chuẩn hoá | 3.85 và 1.79 |
| Bạn mới (12 giờ, 6.5 điểm): 5 láng giềng | 4 qua, 1 rớt, dự đoán qua môn |
| Đoán bừa cả lớp qua | 0.6000 |
| Ba cột, không chuẩn hoá | 0.5333 |
| Ba cột, có chuẩn hoá | 0.8667 |
| Bỏ cột chi tiêu, có chuẩn hoá | 0.9000 |
| k tốt nhất theo kiểm định chéo | 13, điểm 0.8667 |
| Hai vầng trăng, KNN k = 13 | 0.9200 |
| Hai vầng trăng, logistic và Naive Bayes | 0.8533 |
| Giá nhà, MAE của KNN (k = 5) và đường thẳng | 0.4030 và 0.2578 |

Căn 150 m2 và 250 m2 cùng được KNN đoán 9.32 tỷ, vì cả hai có chung năm láng giềng là những căn rộng nhất trong tập học (104.2 đến 117.5 m2). KNN không ngoại suy được.

## Bài tập

1. `bai1.py`: hàm `knn_du_doan(X_hoc, y_hoc, x_moi, k)` dùng `np.argsort`, dự đoán cho tập kiểm tra hai vầng trăng với k = 13 và so với `KNeighborsClassifier`. Hai accuracy phải bằng nhau (0.9200).
2. `bai2.py`: chọn k bằng kiểm định chéo với `metric="manhattan"` và so với Euclid, in nhận xét.
3. `bai3.py`: đổi cột `chi_tieu` sang triệu đồng, chạy k = 7 có và không chuẩn hoá, in giải thích.
4. `bai4.py`: `weights="distance"` với k = 1, 13, 101 trên hai vầng trăng, đặt cạnh bảng `uniform`.
5. `bai5.py`: chọn k từ 1 đến 20 cho `KNeighborsRegressor` bằng `neg_mean_absolute_error` và so MAE với đường thẳng.
6. `bai6.py`: đo thời gian `.fit` và `.predict` cho 1000 mẫu mới với tập học 1000, 10000, 100000 mẫu. Thời gian tuỳ máy, chỉ cần so tương đối giữa các cỡ dữ liệu.

## Lỗi thường gặp

- Accuracy thấp hơn đoán bừa: quên chuẩn hoá, dùng `make_pipeline(StandardScaler(), ...)`.
- Accuracy trên tập học bằng 1.0: đang dùng k = 1 và chấm trên tập học, hãy chấm trên tập kiểm tra.
- `ValueError: Expected n_neighbors <= n_samples`: k lớn hơn số mẫu của tập học.
- Kết quả khác tài liệu: quên `random_state=17` hoặc `stratify=y` khi chia dữ liệu.
