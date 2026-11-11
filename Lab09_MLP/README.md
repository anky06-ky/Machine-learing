# Bài 09: Mạng nơ-ron nhiều lớp (MLP)

Bài thực hành theo [Lab09_MLP.pdf](./Lab09_MLP.pdf), gồm năm chương trình hướng dẫn tương ứng mục 2-6 và sáu bài tập ở mục 7. Mỗi chương trình chạy độc lập; đường dẫn dữ liệu và hình được xác định theo vị trí file Python.

## Cài đặt

Từ thư mục gốc của repository, tạo môi trường nếu chưa có và cài thư viện:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\Lab09_MLP\requirements.txt
```

Nếu đã có `.venv`, dùng lại môi trường đó. Không cần kích hoạt môi trường khi gọi trực tiếp Python như các lệnh dưới đây.

## Chạy phần hướng dẫn

Chạy lần lượt từ thư mục gốc của repository:

```powershell
.\.venv\Scripts\python.exe .\Lab09_MLP\m1_lan_truyen_tien.py
.\.venv\Scripts\python.exe .\Lab09_MLP\m2_xor.py
.\.venv\Scripts\python.exe .\Lab09_MLP\m3_do_lon_mang.py
.\.venv\Scripts\python.exe .\Lab09_MLP\m4_chuan_hoa.py
.\.venv\Scripts\python.exe .\Lab09_MLP\m5_chu_so.py
```

1. **`m1_lan_truyen_tien.py`:** tự tính một nơ-ron sigmoid và một mạng ReLU nhỏ giải XOR bằng numpy.
2. **`m2_xor.py`:** hồi quy logistic và MLP 1, 2, 4, 8 nơ-ron ẩn trên bộ XOR, đếm số tham số.
3. **`m3_do_lon_mang.py`:** ba mạng trên hai vầng trăng để thấy học vẹt, rồi chọn alpha bằng kiểm định chéo 5 phần.
4. **`m4_chuan_hoa.py`:** MLP trên dữ liệu sinh viên mở rộng khi không và có chuẩn hoá.
5. **`m5_chu_so.py`:** MLP nhận dạng chữ số viết tay, số tham số, đường mất mát, dừng sớm và so với các mô hình đã học.

## Chạy bài tập

Sáu bài tập được đặt riêng trong `baitap09` theo yêu cầu của PDF:

```powershell
.\.venv\Scripts\python.exe .\Lab09_MLP\baitap09\bai1.py
.\.venv\Scripts\python.exe .\Lab09_MLP\baitap09\bai2.py
.\.venv\Scripts\python.exe .\Lab09_MLP\baitap09\bai3.py
.\.venv\Scripts\python.exe .\Lab09_MLP\baitap09\bai4.py
.\.venv\Scripts\python.exe .\Lab09_MLP\baitap09\bai5.py
.\.venv\Scripts\python.exe .\Lab09_MLP\baitap09\bai6.py
```

1. **Bài 1:** lấy `coefs_`, `intercepts_` của mạng 8 nơ-ron ẩn trên XOR, tự lan truyền tiến cho 5 điểm đầu của tập kiểm tra.
2. **Bài 2:** kiểm chứng mạng h1 = max(0, x1 - x2), h2 = max(0, x2 - x1), y = h1 + h2.
3. **Bài 3:** so `activation` bằng `"logistic"`, `"tanh"`, `"relu"` trên hai vầng trăng.
4. **Bài 4:** bốn giá trị `learning_rate_init` trên chữ số, lưu bốn đường mất mát vào `baitap09/toc_do_hoc.png`.
5. **Bài 5:** `GridSearchCV` với 4 kiến trúc và 3 giá trị alpha, so với SVM. Bài này chạy lâu nhất, khoảng vài phút.
6. **Bài 6:** `MLPRegressor` cho giá nhà, so MAE với hồi quy tuyến tính, dự đoán căn 250 m², lưu `baitap09/mlp_hoi_quy_ngoai_suy.png`.

## Dữ liệu và cách đánh giá

- `data/xor.csv`: dữ liệu gốc của bài, 400 điểm, lớp 1 nằm ở góc trên phải và góc dưới trái.
- `data/hai_vang_trang.csv`: hai vầng trăng 300 điểm, dùng lại từ bài 4 và bài 6.
- `data/sinh_vien_mo_rong.csv`: dữ liệu sinh viên có thêm cột `chi_tieu` tính bằng nghìn đồng, dùng lại từ bài 4.
- `data/gia_nha.csv`: dữ liệu giá nhà của bài 1, dùng cho bài tập 6.
- Chữ số viết tay: `load_digits()` có sẵn trong scikit-learn.
- Phân loại chia 75% học và 25% kiểm tra, `random_state=17`, phân tầng theo nhãn; giá nhà chia 80/20 với `random_state=42`. Mọi mạng dùng `random_state=0`, chuẩn hoá nằm trong pipeline.

## Giải thích kết quả

Các con số của chương trình hướng dẫn, đối chiếu với PDF:

- `m1`: z = **1.80**, sigmoid(z) = **0.8581**; mạng ReLU cho đúng XOR **[0, 1, 1, 0]**. Khớp PDF.
- `m2`: logistic **0.4500**; MLP 1 và 2 nơ-ron **0.6900**, 4 nơ-ron **0.9300**, 8 nơ-ron **0.9100** (33 tham số). Khớp PDF.
- `m3`: mạng (2,) **0.8444 / 0.8533** và chọn alpha = **1.0** bằng kiểm định chéo (**0.8578**), tập kiểm tra **0.9067**, khớp PDF. Mạng (100, 100) gần như không điều chuẩn cho tập học **1.0000**, tập kiểm tra **0.7600** (PDF 0.7333); cùng mạng với alpha = 1.0 cho tập học 0.8844 (PDF 0.8800). Điểm kiểm định chéo với alpha 1e-5, 1e-3, 0.1 là 0.7867, 0.7867, 0.7689 (PDF 0.7778, 0.7733, 0.7644).
- `m4`: đoán bừa **0.6000**, không chuẩn hoá **0.4000** (đoán rớt cả 30 bạn), có chuẩn hoá **0.9000**. Khớp PDF.
- `m5`: **7510** tham số, **144** vòng học, mất mát 2.5206 → 0.4448 → 0.0052; accuracy **0.9844**, sai **7/450** ảnh đúng như danh sách trong PDF; dừng sớm sau 44 vòng còn **0.9578**; logistic 0.9711, KNN 0.9867, SVM 0.9911. Khớp PDF.

Độ lệch ở `m3` chỉ là 1-2 mẫu, xảy ra với mạng (100, 100) có khoảng 10 nghìn tham số học bằng `solver="lbfgs"`. Dữ liệu là bản gốc: mạng (2,) và các dòng alpha ≥ 1 khớp tuyệt đối, mọi mạng đều hội tụ không cảnh báo. L-BFGS trên mạng lớn như vậy nhạy với sai khác làm tròn số giữa các phiên bản SciPy/OpenBLAS, nên lệch vài mẫu là bình thường (môi trường kiểm tra: Python 3.14.8, scikit-learn 1.9.1, SciPy 1.18.1). Kết luận của PDF giữ nguyên: mạng to không điều chuẩn học vẹt, alpha = 1.0 được chọn và đạt 0.9067 trên tập kiểm tra.

Bài tập:

- **Bài 1:** W1 có dạng (2, 8), W2 có dạng (8, 1). Tự tính H = ReLU(X @ W1 + b1), p = sigmoid(H @ W2 + b2) cho 5 điểm, kết quả **trùng** cột thứ hai của `predict_proba` tới 12 chữ số thập phân. Mỗi cột của W1 là trọng số vào một nơ-ron ẩn; lớp ra dùng sigmoid vì bài toán có hai lớp, nên cột thứ hai của `predict_proba` chính là sigmoid(z) của nơ-ron ra. Mạng chỉ gồm nhân ma trận, cộng hệ số chặn và hàm kích hoạt.
- **Bài 2:** mạng cho đúng **[0, 1, 1, 0]**. h1 chỉ khác 0 khi x1 > x2, h2 chỉ khác 0 khi x2 > x1, nên y = h1 + h2 = |x1 - x2|: hai bit giống nhau cho 0, khác nhau cho 1, đúng XOR. Bỏ ReLU thì (x1 - x2) + (x2 - x1) luôn bằng 0; chính hàm kích hoạt phi tuyến giúp mạng làm được điều một đường thẳng không làm được.
- **Bài 3:** tập học / tập kiểm tra: logistic **0.8400 / 0.8533**, tanh **0.8711 / 0.9067**, relu **0.8844 / 0.9067**. Tanh và ReLU bằng nhau trên tập kiểm tra, ReLU nhỉnh hơn trên tập học; logistic kém nhất, accuracy gần bằng mạng hai nơ-ron ở mục 4 (0.8444 / 0.8533). Đây là so sánh minh hoạ trên một lần chia; muốn chọn hàm kích hoạt thật thì phải chọn bằng kiểm định chéo trên tập học.
- **Bài 4:** với tốc độ **0.0001 / 0.001 / 0.01 / 0.1**: số vòng **582 / 144 / 38 / 30**, mất mát cuối **0.0189 / 0.0052 / 0.0011 / 0.1126**, tập kiểm tra **0.9889 / 0.9844 / 0.9778 / 0.9622**. Tốc độ quá nhỏ làm mất mát giảm rất chậm, cần nhiều vòng mà mất mát cuối vẫn cao hơn tốc độ 0.001. Tốc độ quá lớn làm mất mát nhảy lên xuống (đường 0.1 trên hình), dừng ở mức cao nhất và accuracy thấp nhất.
- **Bài 5:** cặp tốt nhất **(50,), alpha = 1**, kiểm định chéo **0.9785**. Trên tập kiểm tra MLP đạt **0.9822**, **không vượt** SVM kernel RBF (**0.9911**). Với cả bốn kiến trúc, alpha = 1 đều cho điểm kiểm định chéo cao nhất, cho thấy điều chuẩn mạnh có ích với 1347 ảnh học.
- **Bài 6:** MAE trên tập kiểm tra: MLP **0.5008** tỷ, hồi quy tuyến tính **0.2578** tỷ. Căn 250 m²: MLP **28.18** tỷ, tuyến tính **19.91** tỷ. Tập học chỉ có diện tích 37.6-117.5 m², nên 250 m² là ngoại suy. Mạng ReLU ngoài miền học chỉ kéo dài đoạn thẳng cuối cùng của nó: giá tăng 1.38 tỷ cho mỗi 10 m² ở 240 m², so với 0.60 tỷ ở 60 m², còn đường thẳng giữ đúng 0.78 tỷ ở mọi nơi. Trên hình, dưới khoảng 45 m² đường MLP lại cho giá tăng khi diện tích giảm, trái với xu hướng của dữ liệu. Mạng nơ-ron **không ngoại suy tốt** như đường thẳng; chưa có giá thật của căn 250 m² nên không khẳng định con số nào đúng.

Nhận xét:

- Mạng nơ-ron chỉ là các phép nhân ma trận và hàm kích hoạt phi tuyến; nhờ ReLU mà một lớp ẩn giải được XOR.
- Mạng càng lớn càng dễ học vẹt; alpha và độ lớn mạng là nút vặn giữa cẩu thả và học vẹt, chọn bằng kiểm định chéo.
- Mạng nơ-ron bắt buộc chuẩn hoá. Với dữ liệu nhỏ như chữ số 8 x 8, SVM và KNN vẫn nhỉnh hơn MLP.
