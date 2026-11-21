# Bài 01: Hồi quy tuyến tính

Bài thực hành giới thiệu quy trình xây dựng mô hình hồi quy tuyến tính để dự đoán giá nhà. Bài làm đi từ việc đọc và khám phá dữ liệu đến tính toán thủ công, sử dụng scikit-learn, đánh giá mô hình và mở rộng sang hồi quy nhiều biến.

## Mục tiêu

- Hiểu mô hình đường thẳng `y = wx + b` và cách đo sai số bằng MSE.
- Tìm hệ số bằng công thức bình phương tối thiểu.
- Huấn luyện mô hình với `LinearRegression`.
- Chia tập huấn luyện/kiểm tra và đánh giá bằng MAE, MSE, RMSE, R².
- Hiểu vai trò của chuẩn hóa dữ liệu và tốc độ học trong gradient descent.
- So sánh hồi quy một biến với hồi quy nhiều biến.

## Dữ liệu

Tệp [`data/gia_nha.csv`](./data/gia_nha.csv) gồm 60 mẫu nhà với bốn thuộc tính:

| Cột | Ý nghĩa |
| --- | --- |
| `dien_tich` | Diện tích nhà (m²) |
| `so_phong` | Số phòng |
| `tuoi_nha` | Tuổi nhà (năm) |
| `gia` | Giá nhà (tỷ đồng) |

## Nội dung bài làm

Notebook [`Linear_Regression.ipynb`](./Linear_Regression.ipynb) tổng hợp toàn bộ bài thực hành, gồm:

1. Đọc và khám phá dữ liệu.
2. Đo sai số của một đường thẳng.
3. Tìm hệ số bằng công thức bình phương tối thiểu.
4. Xây dựng mô hình bằng scikit-learn.
5. Chia dữ liệu và đánh giá mô hình.
6. Tìm lời giải bằng gradient descent.
7. Xây dựng hồi quy tuyến tính nhiều biến.

Sáu bài tập cuối tài liệu (lọc dữ liệu, trực quan hóa giá theo số phòng, thay đổi biến đầu vào, bổ sung đặc trưng, thử nhiều tốc độ học và viết hàm dự đoán có cảnh báo) nằm trong thư mục `baitap01/`, mỗi bài một tệp `bai1.py` đến `bai6.py`. Notebook cũng có lời giải của sáu bài này.

## Cấu trúc thư mục

```text
01. Linear_Regression/
├── README.md
├── Linear_Regression.ipynb
├── Linear_Regression.pdf
├── code/
│   ├── b1_doc_du_lieu.py
│   ├── b2_do_sai_so.py
│   ├── b3_cong_thuc.py
│   ├── b4_sklearn.py
│   ├── b5_danh_gia.py
│   ├── b6_gradient_descent.py
│   └── b7_nhieu_bien.py
├── baitap01/
│   ├── bai1.py
│   ├── bai2.py
│   ├── bai3.py
│   ├── bai4.py
│   ├── bai5.py
│   └── bai6.py
├── data/
│   └── gia_nha.csv
└── figures/
    └── bai2_so_phong_gia.png
```

## Cài đặt và chạy

Yêu cầu Python 3. Từ thư mục gốc của kho lưu trữ, cài các thư viện:

```bash
python -m pip install numpy pandas matplotlib scikit-learn jupyter
```

Mở notebook:

```bash
jupyter notebook "01. Linear_Regression/Linear_Regression.ipynb"
```

Hoặc chạy từng bước và từng bài tập từ thư mục gốc của kho lưu trữ. Các tệp đọc dữ liệu theo vị trí tệp mã nên chạy từ thư mục nào cũng được; trong PowerShell cần đặt đường dẫn trong dấu nháy vì tên thư mục có dấu cách:

```powershell
python ".\01. Linear_Regression\code\b1_doc_du_lieu.py"
python ".\01. Linear_Regression\code\b2_do_sai_so.py"
python ".\01. Linear_Regression\code\b3_cong_thuc.py"
python ".\01. Linear_Regression\code\b4_sklearn.py"
python ".\01. Linear_Regression\code\b5_danh_gia.py"
python ".\01. Linear_Regression\code\b6_gradient_descent.py"
python ".\01. Linear_Regression\code\b7_nhieu_bien.py"

python ".\01. Linear_Regression\baitap01\bai1.py"
python ".\01. Linear_Regression\baitap01\bai2.py"
python ".\01. Linear_Regression\baitap01\bai3.py"
python ".\01. Linear_Regression\baitap01\bai4.py"
python ".\01. Linear_Regression\baitap01\bai5.py"
python ".\01. Linear_Regression\baitap01\bai6.py"
```

Bài tập 2 lưu hình `bai2.png` trong thư mục `baitap01/` khi chạy.

## Kết luận

- Công thức bình phương tối thiểu và `LinearRegression` cho cùng hệ số ở mô hình một biến.
- Đánh giá trên tập kiểm tra phản ánh tốt hơn khả năng dự đoán dữ liệu chưa từng thấy.
- Gradient descent cần dữ liệu được chuẩn hóa và tốc độ học phù hợp.
- Việc bổ sung biến có thể cải thiện R², nhưng cần kiểm chứng bằng kết quả thực nghiệm.
