# Lab09: Mạng nơ-ron nhiều lớp (MLP)

**Sinh viên:** Trần An Kỳ - **MSSV:** 2474802010460. Môn Học máy ứng dụng, LHP 261_71ITAI41203_0101, năm 3 khóa 30.

Hoàn thành 5 chương trình mẫu và 6 bài tập trong `Lab09_MLP.pdf`.

## Dữ liệu và giới hạn tái lập

- `data/xor.csv`: **bản mô phỏng thay thế**, 400 điểm, seed 9, 24 điểm đảo nhãn, tạo từ `tao_du_lieu.py`. Lớp 1 ở hai góc cùng dấu như mô tả trong đề; các mạng Boolean 0/1 của m1 và bài 2 tính XOR đúng bảng `[0,1,1,0]` riêng. Chưa có CSV gốc nên không ép accuracy khớp PDF.
- `data/hai_vang_trang.csv`: tái tạo đúng dữ liệu [Bài06 trên repo giảng viên](https://github.com/AnhNguyenVLU/Machine_Learning_VLU/blob/4461e7ff473f7d34a6a2f2ec2473f808012a4972/06.%20SVM/SVM.ipynb), `make_moons(n_samples=200, noise=0.2, random_state=42)`. Có `lay_du_lieu_bai06.py` để tái lập.
- `data/sinh_vien_mo_rong.csv`: sao chép nguyên Lab04. Chi tiết nguồn tất cả file: [data/NGUON.md](./data/NGUON.md).
- `data/gia_nha.csv`: sao chép nguyên từ Lab01; trùng file giá nhà bạn bổ sung tại thư mục gốc.
- `load_digits`: ảnh chữ số 8x8 có sẵn trong scikit-learn, không cần tải dữ liệu.

## Cài đặt và chạy

Từ thư mục gốc repository trong PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\Lab09_MLP\requirements.txt
.\.venv\Scripts\python.exe .\Lab09_MLP\chay_tat_ca.py
```

Bỏ lệnh tạo nếu `.venv` đã có. Đã kiểm tra với Python 3.14.8; dùng đúng phiên bản trong `requirements.txt` để dễ đối chiếu. `.venv/` không được Git theo dõi. Runner giới hạn thư viện số học còn một luồng để tránh chi phí chia luồng trên mạng nhỏ; vẫn thử đủ lưới và 5 fold.

## Năm chương trình thực hành

1. `code/m1_lan_truyen_tien.py`: một nơ-ron z=1.8, sigmoid; tự đặt hai ReLU giải bảng XOR bằng phép nhân ma trận.
2. `code/m2_xor.py`: logistic và MLP 1/2/4/8 nơ-ron ẩn, solver lbfgs, max_iter=5000; in số tham số, accuracy học/test.
3. `code/m3_do_lon_mang.py`: mạng (2,), (100,100) với alpha nhỏ/lớn; thử đủ alpha 1e-5, 1e-3, 0.1, 1, 3, 10 bằng CV 5 phần trên tập học, max_iter=20000.
4. `code/m4_chuan_hoa.py`: mạng (20,), Adam, max_iter=3000 có/không StandardScaler trên sinh viên mở rộng; in số bạn được đoán qua môn.
5. `code/m5_chu_so.py`: mạng (100,), Adam, max_iter=1000; đọc shape trọng số, xác nhận 7.510 tham số, loss và số vòng. So early stopping và logistic/KNN/SVM trên cùng cách chia; lưu đường loss/validation và ảnh đoán sai.

## Sáu bài tập và giải thích

1. `baitap09/bai1.py`: học lại mạng XOR 8 nơ-ron; tính `H=ReLU(X@W1+b1)`, `p=sigmoid(H@W2+b2)` cho 5 điểm test. So với cột xác suất lớp 1 bằng `assert_allclose` tới sai số 1e-12. Mỗi cột W1 là trọng số một nơ-ron; bias cộng theo cột.
2. `baitap09/bai2.py`: h1=max(0,x1-x2), h2=max(0,x2-x1), y=h1+h2=|x1-x2|. Hai bit bằng nhau cho 0; khác nhau cho 1. Bỏ ReLU thì hai hiệu triệt tiêu và mạng luôn trả 0.
3. `baitap09/bai3.py`: so logistic/tanh/relu, mạng (100,100), alpha=1, solver lbfgs, in accuracy học/test và tên hàm cao nhất. Kết luận chỉ cho phép thử này; muốn chọn mô hình triển khai phải dùng CV.
4. `baitap09/bai4.py`: tốc độ 0.0001/0.001/0.01/0.1, in n_iter_, loss cuối, test accuracy; lưu đủ bốn đường trong `figures/toc_do_hoc.png`. Tốc độ nhỏ học chậm; quá lớn có thể dao động, phải nhìn đường loss thực tế.
5. `baitap09/bai5.py`: GridSearchCV đủ 12 cặp kiến trúc (10,), (50,), (100,), (50,50) và alpha 0.0001/0.01/1, CV=5 trên tập học; scaler trong pipeline mỗi fold. In cặp tốt nhất, test accuracy và so với SVM RBF; lưu toàn bộ `ket_qua/grid_search.csv`.
6. `baitap09/bai6.py`: MLPRegressor (20,), max_iter=5000, có chuẩn hoá; so MAE với LinearRegression cùng một cột diện tích, dự đoán 250m² và vẽ tới miền đó. Ngoài miền học không có nhãn thật để kết luận bên nào chính xác; MLP ReLU không bị ràng buộc giữ xu hướng tuyến tính.

Phân loại chia 75/25, seed 17, phân tầng; giá nhà chia 80/20, seed 42. Mọi MLP dùng seed 0. Các bảng test khảo sát theo đề chỉ dùng minh hoạ; lựa chọn alpha/lưới dựa trên CV tập học. [API MLPClassifier và các thuộc tính](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html).

## Chạy riêng và xem kết quả

```powershell
.\.venv\Scripts\python.exe .\Lab09_MLP\code\m1_lan_truyen_tien.py
.\.venv\Scripts\python.exe .\Lab09_MLP\baitap09\bai1.py
.\.venv\Scripts\python.exe .\Lab09_MLP\chay_tat_ca.py --chi bai4 bai5
```

- `ket_qua/*.txt`: đầu ra và cảnh báo thật từng chương trình.
- `ket_qua/*.png`: ảnh **kết xuất log, không phải ảnh chụp terminal**. Chạy từng bài rồi chụp terminal khi giảng viên yêu cầu đúng ảnh chụp màn hình.
- `ket_qua/tong_hop.json`: mã thoát, thời gian và cờ có stderr của lần chạy gần nhất của từng bài; `--chi` giữ kết quả các bài khác.
- `figures/`: loss, ảnh chữ số sai, tốc độ học và ngoại suy hồi quy.
- [Kết quả chạy](./KET_QUA.md): số liệu đo từ lần chạy thực tế.

`ConvergenceWarning` nghĩa là bộ tối ưu chưa đạt điều kiện dừng trong giới hạn vòng/số lần tính hàm, không phải crash; runner lưu nguyên cảnh báo để kiểm tra. Mạng lớn và dò lưới có thể mất vài phút. Nếu thay dữ liệu XOR gốc, chạy lại runner; kết quả có thể đổi theo dữ liệu, thư viện và bộ tối ưu.
