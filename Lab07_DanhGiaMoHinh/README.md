# Bài 07: Đánh giá mô hình

Sáu bài tập trong `Lab07_DanhGiaMoHinh.pdf`.

## Cài đặt và chạy

Từ thư mục gốc repository:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\Lab07_DanhGiaMoHinh\requirements.txt
python .\Lab07_DanhGiaMoHinh\bai1.py
python .\Lab07_DanhGiaMoHinh\bai2.py
python .\Lab07_DanhGiaMoHinh\bai3.py
python .\Lab07_DanhGiaMoHinh\bai4.py
python .\Lab07_DanhGiaMoHinh\bai5.py
python .\Lab07_DanhGiaMoHinh\bai6.py
```

Nếu venv đã có thì dùng lại, bỏ lệnh tạo. Chụp kết quả terminal và đồ thị để nộp theo yêu cầu giảng viên.

## Dữ liệu

- `data/sinh_vien_bo_hoc.csv`: dữ liệu **mô phỏng thay thế**, seed 7, 1.200 sinh viên, 119 nhãn bỏ học; chưa có CSV gốc của đề.
- `data/gia_nha.csv`: dữ liệu giá nhà từ Lab01.
- Bài 2 dùng bộ chữ số có sẵn `load_digits`.

Các kết quả dưới đây đo trên dữ liệu đi kèm, có thể khác PDF.

## Bài 1: tự viết hàm đánh giá

`danh_gia(y_that, y_doan)` dùng NumPy tính accuracy, precision, recall, F1 và kiểm tra khớp scikit-learn. Logistic đạt **0.9300 / 0.7143 / 0.5000 / 0.5882**; mô hình lười đạt **0.9000 / 0 / 0 / 0**.

Mô hình lười không báo lớp 1 nên TP+FP=0, precision không xác định. Hàm quy ước trả 0, giống `zero_division=0`.

## Bài 2: đánh giá chữ số viết tay

SVM RBF đạt accuracy **0.9911** trên 450 ảnh. Chữ số **8** có precision và recall thấp nhất, đều **0.9767**. Macro và weighted gần bằng nhau vì support các chữ số gần cân bằng.

## Bài 3: chọn ngưỡng đạt recall 80%

Chỉ dùng xác suất OOF 5-fold trên tập học. Ngưỡng lớn nhất là **0.1453320871**, recall OOF **0.8090**. Trên test: recall **0.8333**, precision **0.3788**, báo **66** sinh viên.

Ngưỡng là xác suất của mẫu dương thứ `ceil(0.8*n_duong)` khi sắp giảm dần. Đạt 80% trên OOF không bảo đảm 80% trên dữ liệu mới; không sửa ngưỡng theo test.

## Bài 4: vẽ ROC

Lưu `roc.png` ngay trong thư mục lab. AUC logistic **0.9154**, rừng **0.8825**; hai đường có đổi thứ tự cao/thấp. AUC lớn hơn cho thấy khả năng xếp hạng tổng thể tốt hơn, không bảo đảm tốt hơn tại mọi ngưỡng.

## Bài 5: cân bằng lớp

Recall của rừng tăng **0.5333 → 0.6667**, SVM tăng **0.2333 → 0.8667**. SVM thay đổi recall nhiều nhất. Precision giảm ở cả hai; cân bằng lớp không bảo đảm F1 tăng.

## Bài 6: một dự đoán sai lớn

Cộng 5 tỷ vào đúng một dự đoán: MAE tăng **0.2578 → 0.6745 tỷ**, RMSE **0.3730 → 1.5505 tỷ**, R² giảm **0.9622 → 0.3475**. RMSE tăng mạnh hơn MAE vì bình phương sai số trước khi lấy trung bình. R² không cùng đơn vị để so mức thay đổi trực tiếp với hai độ đo còn lại.

Phân loại chia 75/25, seed 17, phân tầng; hồi quy chia 80/20, seed 42. Scaler nằm trong pipeline để tránh rò rỉ dữ liệu.
