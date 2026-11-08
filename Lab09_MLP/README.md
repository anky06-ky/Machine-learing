# Bài 09: Mạng nơ-ron nhiều lớp

Sáu bài tập trong `Lab09_MLP.pdf`.

## Cài đặt và chạy

Từ thư mục gốc repository:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\Lab09_MLP\requirements.txt
python .\Lab09_MLP\bai1.py
python .\Lab09_MLP\bai2.py
python .\Lab09_MLP\bai3.py
python .\Lab09_MLP\bai4.py
python .\Lab09_MLP\bai5.py
python .\Lab09_MLP\bai6.py
```

Nếu venv đã có thì dùng lại. Chụp kết quả terminal và đồ thị để nộp. Dò lưới bài 5 chạy lâu hơn các bài khác.

## Dữ liệu

- `data/xor.csv`: **mô phỏng thay thế**, seed 9, 400 điểm, 24 điểm đảo nhãn; chưa có CSV gốc của PDF. Lớp 1 nằm ở hai góc cùng dấu như mô tả dữ liệu trong đề.
- `data/hai_vang_trang.csv`: đúng `make_moons(n_samples=200, noise=0.2, random_state=42)` từ [Bài06 của giảng viên](https://github.com/AnhNguyenVLU/Machine_Learning_VLU/blob/4461e7ff473f7d34a6a2f2ec2473f808012a4972/06.%20SVM/SVM.ipynb).
- `data/gia_nha.csv`: dữ liệu Lab01. Bộ chữ số dùng `load_digits` có sẵn trong sklearn.

Kết quả dưới đây dựa trên dữ liệu đi kèm, có thể khác số mẫu trong PDF.

## Bài 1: tự lan truyền tiến

Học MLP 8 nơ-ron ẩn rồi lấy `coefs_`, `intercepts_`. Tự tính `H=ReLU(X@W1+b1)`, `p=sigmoid(H@W2+b2)` cho 5 điểm test, đối chiếu với cột xác suất lớp 1.

Hai kết quả khớp tới sai số **1e-12**. Mỗi cột W1 là trọng số của một nơ-ron ẩn, bias được cộng theo cột.

## Bài 2: mạng XOR khác

`h1=max(0,x1-x2)`, `h2=max(0,x2-x1)`, `y=h1+h2=|x1-x2|`. Trên bốn đầu vào Boolean, kết quả là **[0, 1, 1, 0]**: hai bit giống nhau cho 0, khác nhau cho 1.

Nếu bỏ ReLU, hai hiệu triệt tiêu và đầu ra luôn bằng 0.

## Bài 3: so hàm kích hoạt

Cùng mạng (100,100), alpha=1, solver lbfgs. Accuracy test: logistic **0.8200**, tanh **0.8200**, ReLU **0.9800**. ReLU cao nhất trong phép thử này; muốn chọn hàm cho triển khai phải dùng CV trên tập học.

## Bài 4: tốc độ học

Với tốc độ **0.0001 / 0.001 / 0.01 / 0.1**, mạng học **582 / 144 / 38 / 30** vòng, accuracy test **0.9889 / 0.9844 / 0.9778 / 0.9489**. Lưu đủ bốn đường mất mát trong `toc_do_hoc.png`.

Tốc độ nhỏ học chậm; quá lớn có thể làm loss dao động và giảm độ chính xác.

## Bài 5: GridSearchCV

Thử 4 kiến trúc × 3 alpha, CV 5-fold. Cặp tốt nhất là **(50,), alpha=1**, accuracy CV **0.9785**, test **0.9822**. SVM RBF trên cùng test đạt **0.9911**, cao hơn MLP trong phép thử này.

Scaler nằm trong pipeline nên mỗi fold chỉ chuẩn hoá từ phần huấn luyện của fold.

## Bài 6: MLP cho hồi quy

MLPRegressor (20,), max_iter=5000, có chuẩn hoá đạt MAE **0.5008 tỷ**; hồi quy tuyến tính đạt **0.2578 tỷ**. Dự đoán căn 250m² lần lượt **28.1758** và **19.9139 tỷ**.

250m² nằm ngoài miền học 37.6–117.5m² nên là ngoại suy; chưa có giá thật để khẳng định mô hình nào đúng. Lưu `mlp_hoi_quy_ngoai_suy.png` để so hai đường dự đoán.

Phân loại chia 75/25, seed 17, phân tầng; giá nhà chia 80/20, seed 42. MLP dùng seed 0. Kết quả có thể thay đổi theo phiên bản thư viện.
