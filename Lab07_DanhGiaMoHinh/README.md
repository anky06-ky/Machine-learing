# Lab07: Đánh giá mô hình

**Sinh viên:** Trần An Kỳ - **MSSV:** 2474802010460. Môn Học máy ứng dụng, LHP 261_71ITAI41203_0101, năm 3 khóa 30.

Triển khai các mục 2-7 và đủ 6 bài tập mục 8 của `Lab07_DanhGiaMoHinh.pdf`. Chú thích có dấu; phần in kết quả không dấu theo quy ước của đề.

## Dữ liệu và nguồn

- `data/sinh_vien_bo_hoc.csv`: **bản mô phỏng thay thế**, tạo bằng `tao_du_lieu.py`, seed 7, 1.200 dòng, 119 nhãn bỏ học. Chưa có CSV gốc của giảng viên; không coi dữ liệu hoặc kết quả này là bản gốc trong PDF.
- Các cột: `diem_tb`, `so_buoi_vang`, `tin_chi_no`, `gio_lam_them`, nhãn `bo_hoc` (1 = bỏ học).
- `data/gia_nha.csv`: sao chép nguyên dữ liệu Lab01; trùng nội dung file `gia_nha.csv` bạn bổ sung ở thư mục gốc.
- Thay CSV bằng dữ liệu gốc cùng tên/cột, chạy lại để cập nhật kết quả. Không cần và không nên sửa mô hình để ép khớp các con số mẫu trong PDF.
- Đã kiểm tra repo giảng viên theo yêu cầu lấy dữ liệu từ Bài06: notebook SVM dùng blobs/moons/Iris và không có CSV sinh viên bỏ học. [Chi tiết nguồn](./data/NGUON.md).

## Cài môi trường

Từ thư mục gốc repository, trong PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\Lab07_DanhGiaMoHinh\requirements.txt
.\.venv\Scripts\python.exe .\Lab07_DanhGiaMoHinh\chay_tat_ca.py
```

Nếu `.venv` đã tồn tại thì dùng luôn, bỏ lệnh tạo. Có thể chạy trực tiếp Python trong venv, không cần thay ExecutionPolicy. Đã kiểm tra với Python 3.14.8; phiên bản thư viện được cố định trong `requirements.txt`. `.venv/` được Git bỏ qua.

## Thực hành theo đúng thứ tự PDF

1. `code/h1_do_bua.py`: đọc dữ liệu, đếm lớp, DummyClassifier và logistic.
2. `code/h2_ma_tran.py`: tự đếm TP/FP/FN/TN, kiểm tra khớp sklearn; lưu hình ma trận.
3. `code/h3_precision_recall.py`: tính tay precision, recall, F1 và classification_report.
4. `code/h4_nguong.py`: các ngưỡng 0.5, 0.4, 0.3, 0.2, 0.1; chọn F1 từ xác suất OOF 5-fold trên tập học, lưới 0.05-0.95, bước 0.05.
5. `code/h5_roc_auc.py`: ROC logistic và so AUC/recall/F1 của logistic, GaussianNB, KNN 15, cây sâu 4, rừng 200 cây, SVM RBF.
6. `code/h6_can_bang.py`: 10 cách chia không phân tầng, CV accuracy/precision/recall/F1 và logistic cân bằng lớp.

## Bài tập và nhận xét

1. `baitap07/bai1.py`: `danh_gia(y_that, y_doan)` chỉ dùng NumPy. Tự kiểm tra với sklearn cho logistic và mô hình lười. Khi TP+FP=0, precision không xác định; quy ước trả 0 tương đương `zero_division=0`.
2. `baitap07/bai2.py`: SVM RBF trên `load_digits`, report 10 chữ số, tự tìm tất cả lớp có precision/recall thấp nhất. Support gần cân bằng nên macro và weighted gần bằng nhau.
3. `baitap07/bai3.py`: ngưỡng **lớn nhất chính xác** để recall OOF >= 0.8 là điểm xác suất của mẫu dương thứ `ceil(0.8*n_duong)` khi sắp giảm dần. Đánh giá test một lần; yêu cầu OOF không bảo đảm recall 0.8 trên dữ liệu mới.
4. `baitap07/bai4.py`: lưu `figures/roc.png`, so AUC và kiểm tra hai ROC có đổi thứ tự cao/thấp trên các FPR chung.
5. `baitap07/bai5.py`: rừng và SVC trước/sau cân bằng; in precision, recall, F1, độ thay đổi và mô hình đổi recall nhiều nhất.
6. `baitap07/bai6.py`: MAE/RMSE/R² trước và sau khi cộng đúng một dự đoán thêm 5 tỷ. In cả chênh lệch và mức tăng tương đối MAE/RMSE; RMSE phạt mạnh lỗi lớn do bình phương sai số. R² không có cùng đơn vị để so mức đổi trực tiếp với MAE/RMSE.

Phân loại: 75% học/25% test, `random_state=17`, `stratify=y`. Giá nhà: 80% học/20% test, `random_state=42`, dùng một cột diện tích như Lab01. Scaler luôn nằm trong pipeline và chỉ học từ tập huấn luyện. Những bảng so nhiều mô hình/ngưỡng trên test là minh hoạ theo đề, không dùng làm tiêu chí chọn mô hình triển khai. [Tài liệu sklearn về chọn ngưỡng](https://scikit-learn.org/stable/modules/classification_threshold.html).

## Chạy riêng và xem kết quả

```powershell
.\.venv\Scripts\python.exe .\Lab07_DanhGiaMoHinh\code\h1_do_bua.py
.\.venv\Scripts\python.exe .\Lab07_DanhGiaMoHinh\baitap07\bai3.py
.\.venv\Scripts\python.exe .\Lab07_DanhGiaMoHinh\chay_tat_ca.py --chi bai1 bai3
```

Các lệnh hoạt động cả từ thư mục repository lẫn thư mục lab nhờ đường dẫn theo `__file__`.

- `ket_qua/*.txt`: stdout/stderr của từng lần chạy thực tế.
- `ket_qua/*.png`: bản kết xuất log thành ảnh để đọc/xem kết quả; **không phải ảnh chụp terminal**. Nếu giảng viên yêu cầu ảnh chụp terminal, mở terminal chạy từng bài rồi chụp màn hình khi nộp.
- `ket_qua/tong_hop.json`: mã thoát và thời gian của lần chạy gần nhất của từng chương trình; `--chi` giữ lại thông tin các bài khác.
- `figures/`: ma trận nhầm lẫn, F1 OOF theo ngưỡng, ROC.
- [Kết quả chạy](./KET_QUA.md): các giá trị đo trên dữ liệu đi kèm.

scikit-learn 1.9.1 báo `FutureWarning` cho `SVC(probability=True)` trong h5. Tham số này được giữ đúng ví dụ PDF và vẫn chạy được trong phiên bản đã cố định; cảnh báo được lưu nguyên trong log.

Không chạy `tao_du_lieu.py --ghi-de` sau khi đã thay bằng dữ liệu gốc: lệnh này cố ý ghi lại bản mô phỏng.
