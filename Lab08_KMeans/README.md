# Bài 08: Phân cụm K-Means

Sáu bài tập trong `Lab08_KMeans.pdf`.

## Cài đặt và chạy

Từ thư mục gốc repository:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\Lab08_KMeans\requirements.txt
python .\Lab08_KMeans\bai1.py
python .\Lab08_KMeans\bai2.py
python .\Lab08_KMeans\bai3.py
python .\Lab08_KMeans\bai4.py
python .\Lab08_KMeans\bai5.py
python .\Lab08_KMeans\bai6.py
```

Nếu venv đã có thì dùng lại. Chụp kết quả terminal và hình để nộp.

## Dữ liệu

- `data/khach_hang.csv`: **mô phỏng thay thế**, seed 8, 210 khách; các cột tuổi, thu nhập, điểm chi tiêu. Chưa có CSV gốc, nên kết quả có thể khác PDF.
- `data/hai_vang_trang.csv`: tạo đúng cấu hình [Bài06 của giảng viên](https://github.com/AnhNguyenVLU/Machine_Learning_VLU/blob/4461e7ff473f7d34a6a2f2ec2473f808012a4972/06.%20SVM/SVM.ipynb): `make_moons(n_samples=200, noise=0.2, random_state=42)`. Giữ `x1`, `x2`, `nhan`; nhãn chỉ dùng chấm điểm.
- Ảnh china/flower và bộ chữ số lấy từ dữ liệu tích hợp scikit-learn. Ảnh mẫu không phải ảnh cá nhân.

## Bài 1: tự tính silhouette

`silhouette_mot_diem(X, nhan, i)` tính a, b, s bằng NumPy. Bỏ chính điểm i khỏi trung bình cùng cụm; cụm một điểm có s=0. Kết quả trung bình **0.7063508042**, khớp sklearn.

Khách index **132** có s thấp nhất, khoảng **0.1441**, thu nhập 24.6 triệu, điểm chi tiêu 57. Lưu `silhouette_khach_yeu_nhat.png`, đánh dấu khách ở gần ranh giới cụm.

## Bài 2: thêm tuổi

Chuẩn hoá cả ba cột rồi thử K=2–10. K tốt nhất vẫn là **5**, silhouette **0.5086**. Chương trình in tâm theo đơn vị gốc để so với phân cụm hai chiều; số thứ tự cụm không có ý nghĩa thứ hạng.

## Bài 3: so cách khởi tạo

Chạy 50 seed, `n_init=1`. Random đạt inertia tốt nhất quan sát được **34/50** lần; K-Means++ đạt **50/50** lần. Inertia tốt nhất là **25.398732**, nên số lần đạt 23.0 theo số mẫu PDF là 0 ở cả hai.

K-Means++ ưu tiên tâm xa nhau, đáng tin hơn trong phép thử này; vẫn nên dùng nhiều lần khởi tạo. Lưu `so_sanh_khoi_tao.png`.

## Bài 4: nén ảnh 16 màu

Lưu `anh_goc.png`, `anh_16_mau.png`; ảnh china nén từ **550.154 xuống 134.431 byte**, khoảng **4.09 lần**. Dung lượng PNG thực tế khác tỷ lệ lý thuyết 6 lần vì còn phụ thuộc định dạng và cách nén.

Mặc định minh hoạ thêm bằng flower. Dùng ảnh cá nhân bằng lệnh sau, thay đường dẫn ví dụ bằng ảnh thật:

```powershell
python .\Lab08_KMeans\bai4.py --anh "D:\Anh\anh_cua_toi.png"
```

PNG có alpha sẽ được bỏ kênh thứ tư trước khi phân cụm.

## Bài 5: DBSCAN

Với eps **0.2 / 0.3 / 0.4**, số cụm là **7 / 2 / 1**, số điểm nhiễu **31 / 1 / 0**. Trên bộ Bài06 này, cả ba cấu hình có ARI thấp hơn K-Means (**0.2462**). Hai cụm không có nghĩa là đã tách đúng hai vầng trăng.

eps nhỏ dễ chia vụn; eps lớn có thể nối các nhóm thành một. Lưu `dbscan_hai_vang_trang.png`.

## Bài 6: phân cụm chữ số

Fit K-Means K=10 không dùng nhãn, sau đó lập crosstab nhãn/cụm. Chữ số **0** gom gọn nhất: **176/178**, khoảng **98.88%**, vào cùng một cụm. Một cụm chứa cùng lúc 100 ảnh số 1 và 100 ảnh số 8; hình dạng tương tự có thể bị gom chung.
