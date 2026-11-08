# Học máy ứng dụng

Kho lưu trữ các bài thực hành môn **Học máy ứng dụng**. Nội dung được tổ chức theo từng bài; mỗi thư mục gồm tài liệu, dữ liệu, mã nguồn, notebook tổng hợp và README hướng dẫn riêng.

## Thông tin sinh viên

| Thông tin | Chi tiết |
| --- | --- |
| Họ và tên | Trần An Kỳ |
| Mã số sinh viên | 2474802010460 |
| Môn học | Học máy ứng dụng |
| Mã lớp học phần | 261_71ITAI41203_0101 |
| Khóa | Khóa 30 |
| Năm học | Sinh viên năm 3 |

## Thông tin giảng viên

- **Học vị:** Thạc sĩ ngành Khoa học Dữ liệu và Trí tuệ Nhân tạo (Data Science & Artificial Intelligence).
- **Họ và tên:** Nguyễn Thái Anh (theo tài liệu thực hành).
- **Đơn vị công tác:** Giảng viên Khoa Công nghệ Thông tin, Trường Đại học Văn Lang.
- **Lĩnh vực quan tâm và nghiên cứu:** Khoa học dữ liệu, trí tuệ nhân tạo và học máy.
- **GitHub:** [AnhNguyenVLU](https://github.com/AnhNguyenVLU)

## Danh sách bài thực hành

| Bài | Chủ đề | Nội dung chính |
| --- | --- | --- |
| [Bài 01](./01_Linear_Regression/) | Hồi quy tuyến tính | Đọc dữ liệu, tính sai số, bình phương tối thiểu, scikit-learn, đánh giá mô hình, gradient descent và hồi quy nhiều biến |
| [Bài 02](./02_Logistic_Regression/) | Hồi quy logistic | Sigmoid, phân loại nhị phân, LogisticRegression, ma trận nhầm lẫn, precision/recall, ngưỡng quyết định và nhiều biến |
| [Bài 03](./03_Naive_Bayes/) | Naive Bayes | Định lý Bayes bằng tay, túi từ (Bag of Words), bộ lọc thư rác MultinomialNB, làm trơn Laplace và GaussianNB |
| [Bài 04](./Lab04_KNN/) | K láng giềng gần nhất (KNN) | Khoảng cách Euclid, chuẩn hoá dữ liệu, chọn k bằng kiểm định chéo, so sánh mô hình và hồi quy KNN |
| [Bài 05](./Lab05_CayQuyetDinh/) | Cây quyết định và rừng ngẫu nhiên | Chỉ số Gini, đọc cây, chọn độ sâu, ảnh hưởng của chuẩn hoá, Random Forest, độ quan trọng đặc trưng và cây hồi quy |
| [Bài 07](./Lab07_DanhGiaMoHinh/) | Đánh giá mô hình | 6 thực hành và 6 bài tập: mô hình lười, confusion matrix, precision/recall/F1, ngưỡng OOF, ROC/AUC, cân bằng lớp, sai số hồi quy |
| [Bài 08](./Lab08_KMeans/) | Phân cụm K-Means | 5 thực hành và 6 bài tập: tự viết K-Means, phân khúc khách hàng, silhouette, khởi tạo, nén màu, DBSCAN, phân cụm chữ số |
| [Bài 09](./Lab09_MLP/) | Mạng nơ-ron nhiều lớp | 5 thực hành và 6 bài tập: lan truyền tiến, XOR, điều chuẩn, chuẩn hoá, chữ số, tốc độ học, dò lưới và hồi quy |

## Công nghệ sử dụng

- Python 3
- Jupyter Notebook
- NumPy, pandas, Matplotlib và scikit-learn

Hướng dẫn cài đặt và chạy chi tiết nằm trong README của từng bài.

## Chạy Lab07, Lab08, Lab09 bằng môi trường ảo

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
.\.venv\Scripts\python.exe .\Lab07_DanhGiaMoHinh\chay_tat_ca.py
.\.venv\Scripts\python.exe .\Lab08_KMeans\chay_tat_ca.py
.\.venv\Scripts\python.exe .\Lab09_MLP\chay_tat_ca.py
```

Nếu `.venv` đã tồn tại thì bỏ lệnh tạo và dùng lại. `.gitignore` loại `.venv/`, `venv/`, cache Python và file tạm; chỉ đưa mã nguồn, dữ liệu, README, tài liệu và kết quả bài làm lên Git.

Mỗi lab có thư mục `code/`, `baitap07/08/09/`, `data/`, `figures/`, `ket_qua/`, README và requirements riêng. Tổng cộng **34 chương trình học tập: 16 thực hành mẫu và 18 bài tập**. Runner lưu log thực tế kèm ảnh kết xuất văn bản; ảnh này không thay cho ảnh chụp terminal nếu giảng viên yêu cầu đúng ảnh chụp màn hình.

**Nguồn dữ liệu:** theo yêu cầu, bộ hai vầng trăng cho Lab08/09 lấy đúng cấu hình tạo dữ liệu trong [Bài06 của giảng viên](https://github.com/AnhNguyenVLU/Machine_Learning_VLU/blob/4461e7ff473f7d34a6a2f2ec2473f808012a4972/06.%20SVM/SVM.ipynb): 200 mẫu, noise=0.2, seed 42. Giá nhà và sinh viên mở rộng dùng nguyên dữ liệu lab trước. Ba CSV `sinh_vien_bo_hoc.csv`, `khach_hang.csv`, `xor.csv` không có trong nguồn Bài06 và hiện là **dữ liệu mô phỏng thay thế có seed cố định**; README và `data/NGUON.md` từng lab ghi rõ. Thay dữ liệu gốc cùng tên/cột rồi chạy lại để cập nhật kết quả. Bài 4 Lab08 hỗ trợ ảnh cá nhân qua `--anh`; mặc định chỉ minh hoạ bằng ảnh mẫu scikit-learn. Không bổ sung bài làm Lab06.

## Cấu trúc kho lưu trữ

```text
Machine-learing/
├── README.md
├── 01_Linear_Regression/
│   ├── README.md
│   ├── Tong_hop_Bai_01_Hoi_quy_tuyen_tinh.ipynb
│   ├── Lab01_Hoi_quy_tuyen_tinh.pdf
│   ├── code/
│   ├── data/
│   └── figures/
├── 02_Logistic_Regression/
│   ├── README.md
│   ├── bai1.py ... bai6.py
│   ├── data/
│   └── requirements.txt
├── 03_Naive_Bayes/
│   ├── README.md
│   ├── Lab03_Naive_Bayes.pdf
│   ├── d1_bayes_tay.py ... d5_gaussian.py
│   ├── data/
│   └── requirements.txt
├── Lab04_KNN/
│   ├── README.md
│   ├── Lab04_KNN.pdf
│   ├── e1_khoang_cach.py ... e5_knn_hoi_quy.py
│   ├── code/
│   ├── data/
│   └── requirements.txt
├── Lab05_CayQuyetDinh/
│   ├── README.md
│   ├── Lab05_CayQuyetDinh_RungNgauNhien.pdf
│   ├── fi_gini.py
│   ├── f2_cay_nam_la.py ... f7_cay_hoi_quy.py
│   ├── data/
│   └── requirements.txt
├── Lab07_DanhGiaMoHinh/
│   ├── README.md, KET_QUA.md, requirements.txt
│   ├── Lab07_DanhGiaMoHinh.pdf
│   ├── code/ (h1 ... h6)
│   ├── baitap07/ (bai1 ... bai6)
│   ├── data/, figures/, ket_qua/
│   └── chay_tat_ca.py
├── Lab08_KMeans/
│   ├── README.md, KET_QUA.md, requirements.txt
│   ├── Lab08_KMeans.pdf
│   ├── code/ (k1 ... k5)
│   ├── baitap08/ (bai1 ... bai6)
│   ├── data/, figures/, ket_qua/
│   └── chay_tat_ca.py
└── Lab09_MLP/
    ├── README.md, KET_QUA.md, requirements.txt
    ├── Lab09_MLP.pdf
    ├── code/ (m1 ... m5)
    ├── baitap09/ (bai1 ... bai6)
    ├── data/, figures/, ket_qua/
    └── chay_tat_ca.py
```

## Liên kết

Kho lưu trữ: [github.com/anky06-ky/Machine-learing](https://github.com/anky06-ky/Machine-learing)
