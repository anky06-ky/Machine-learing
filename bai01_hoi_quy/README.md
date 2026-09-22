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

Notebook [`Tong_hop_Bai_01_Hoi_quy_tuyen_tinh.ipynb`](./Tong_hop_Bai_01_Hoi_quy_tuyen_tinh.ipynb) tổng hợp toàn bộ bài thực hành, gồm:

1. Đọc và khám phá dữ liệu.
2. Đo sai số của một đường thẳng.
3. Tìm hệ số bằng công thức bình phương tối thiểu.
4. Xây dựng mô hình bằng scikit-learn.
5. Chia dữ liệu và đánh giá mô hình.
6. Tìm lời giải bằng gradient descent.
7. Xây dựng hồi quy tuyến tính nhiều biến.

Phần bài tập vận dụng gồm lọc dữ liệu, trực quan hóa giá theo số phòng, thay đổi biến đầu vào, bổ sung đặc trưng, thử nhiều tốc độ học và viết hàm dự đoán có cảnh báo.

## Cấu trúc thư mục

```text
bai01_hoi_quy/
├── README.md
├── Tong_hop_Bai_01_Hoi_quy_tuyen_tinh.ipynb
├── Lab01_Hoi_quy_tuyen_tinh.pdf
├── code/
│   ├── b1_doc_du_lieu.py
│   ├── b2_do_sai_so.py
│   ├── b3_cong_thuc.py
│   ├── b4_sklearn.py
│   ├── b5_danh_gia.py
│   ├── b6_gradient_descent.py
│   └── b7_nhieu_bien.py
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
jupyter notebook bai01_hoi_quy/Tong_hop_Bai_01_Hoi_quy_tuyen_tinh.ipynb
```

Hoặc chạy từng bước riêng lẻ từ thư mục bài:

```bash
cd bai01_hoi_quy
python code/b1_doc_du_lieu.py
python code/b2_do_sai_so.py
python code/b3_cong_thuc.py
python code/b4_sklearn.py
python code/b5_danh_gia.py
python code/b6_gradient_descent.py
python code/b7_nhieu_bien.py
```

## Kết luận

- Công thức bình phương tối thiểu và `LinearRegression` cho cùng hệ số ở mô hình một biến.
- Đánh giá trên tập kiểm tra phản ánh tốt hơn khả năng dự đoán dữ liệu chưa từng thấy.
- Gradient descent cần dữ liệu được chuẩn hóa và tốc độ học phù hợp.
- Việc bổ sung biến có thể cải thiện R², nhưng cần kiểm chứng bằng kết quả thực nghiệm.
