# Bài 06: Máy vectơ hỗ trợ (SVM)

Bài thực hành theo [Lab06_SVM.pdf](./Lab06_SVM.pdf), gồm năm chương trình hướng dẫn tương ứng mục 2-6 và sáu bài tập ở mục 7. Mỗi chương trình chạy độc lập; đường dẫn dữ liệu và ảnh được xác định theo vị trí file Python.

## Cài đặt

Từ thư mục gốc của repository, tạo môi trường nếu chưa có và cài thư viện:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\Lab06_SVM\requirements.txt
```

Nếu đã có `.venv`, dùng lại môi trường đó. Không cần kích hoạt môi trường khi gọi trực tiếp Python như các lệnh dưới đây.

## Chạy phần hướng dẫn

Chạy lần lượt từ thư mục gốc của repository:

```powershell
.\.venv\Scripts\python.exe .\Lab06_SVM\g1_le_rong_nhat.py
.\.venv\Scripts\python.exe .\Lab06_SVM\g2_tham_so_c.py
.\.venv\Scripts\python.exe .\Lab06_SVM\g3_kernel.py
.\.venv\Scripts\python.exe .\Lab06_SVM\g4_chon_tham_so.py
.\.venv\Scripts\python.exe .\Lab06_SVM\g5_chu_so.py
```

1. **`g1_le_rong_nhat.py`:** sinh 24 điểm, tính đường ranh giới và độ rộng lề, liệt kê vectơ hỗ trợ, học lại bằng các vectơ hỗ trợ và dự đoán hai điểm mới.
2. **`g2_tham_so_c.py`:** khảo sát C từ 0.01 tới 100 trên dữ liệu sinh viên; so sánh có và không chuẩn hoá khi thêm cột chi tiêu.
3. **`g3_kernel.py`:** thêm đặc trưng x bình phương cho 14 điểm một chiều, rồi so sánh SVM tuyến tính, đa thức bậc 3, RBF, hồi quy logistic và KNN với k=13 trên hai vầng trăng.
4. **`g4_chon_tham_so.py`:** khảo sát gamma và dò lưới 16 cặp C/gamma bằng kiểm định chéo 5 phần, sau đó đánh giá cặp được chọn trên tập kiểm tra.
5. **`g5_chu_so.py`:** nhận dạng chữ số viết tay, liệt kê ảnh đoán sai và so sánh với SVM có chuẩn hoá, KNN và hồi quy logistic.

## Chạy bài tập

Sáu bài tập được đặt riêng trong `baitap06` theo yêu cầu của PDF:

```powershell
.\.venv\Scripts\python.exe .\Lab06_SVM\baitap06\bai1.py
.\.venv\Scripts\python.exe .\Lab06_SVM\baitap06\bai2.py
.\.venv\Scripts\python.exe .\Lab06_SVM\baitap06\bai3.py
.\.venv\Scripts\python.exe .\Lab06_SVM\baitap06\bai4.py
.\.venv\Scripts\python.exe .\Lab06_SVM\baitap06\bai5.py
.\.venv\Scripts\python.exe .\Lab06_SVM\baitap06\bai6.py
```

Cũng có thể chuyển vào thư mục Lab 6, rồi dùng `..\.venv\Scripts\python.exe .\g3_kernel.py` hoặc `..\.venv\Scripts\python.exe .\baitap06\bai3.py`.

1. **Bài 1:** tự tính f(x) bằng phép nhân và cộng, tính khoảng cách của hai điểm mới và ba vectơ hỗ trợ tới ranh giới; giải thích khoảng cách của vectơ hỗ trợ bằng nửa độ rộng lề.
2. **Bài 2:** lần lượt thêm điểm lớp 1 tại (-1, -1) và (4.5, 4), học lại và giải thích thay đổi của ranh giới.
3. **Bài 3:** chuẩn hoá trong pipeline, thử kernel đa thức bậc 2, 3, 4 và dò C trong 0.1, 1, 10, 100. Chọn bậc bằng điểm CV và so sánh với RBF ở mục 5 trên cùng tập chia.
4. **Bài 4:** dò lưới mịn với C trong 3, 10, 30 và gamma trong 0.3, 1, 3; nhận xét điểm CV và accuracy kiểm tra.
5. **Bài 5:** in bảng nhầm lẫn 10 hàng 10 cột, liệt kê các cặp bị nhầm và lưu từng ảnh đoán sai vào [baitap06/anh_sai](./baitap06/anh_sai/).
6. **Bài 6:** sinh dữ liệu 20 đặc trưng với 2000, 5000, 10000, 20000 mẫu; đo riêng thời gian `.fit()` của `SVC()` và `LinearSVC()`, in thời gian và tỷ lệ tăng so với cỡ trước. Hai lần cuối tăng số mẫu gấp đôi; 2000 lên 5000 tăng 2.5 lần. Thời gian được đo lại mỗi lần chạy và phụ thuộc máy.

## Dữ liệu và cách đánh giá

- `data/sinh_vien.csv`: dùng lại dữ liệu của Lab 2, gồm `gio_on`, `diem_giua_ky` và nhãn `qua_mon`.
- `data/sinh_vien_mo_rong.csv`: dùng lại dữ liệu của Lab 4, bổ sung cột `chi_tieu`.
- `data/hai_vang_trang.csv`: dùng lại đúng dữ liệu hai vầng trăng của Lab 4, gồm `x1`, `x2`, `nhan`.
- Chữ số viết tay: `load_digits()` có sẵn trong scikit-learn, gồm 1797 ảnh 8 x 8; không cần tải dữ liệu ngoài.
- Các bài phân loại chia 75% học và 25% kiểm tra, `random_state=17`, phân tầng theo nhãn. Hai ví dụ 24 điểm và dữ liệu tổng hợp bài 6 dùng `random_state=0`.
- Kiểm định chéo và chuẩn hoá trong pipeline chỉ học từ tập huấn luyện. Việc so sánh các mô hình trên tập kiểm tra trong phần hướng dẫn phục vụ minh hoạ theo PDF.

## Giải thích kết quả

Các kết quả đã kiểm tra với dữ liệu đi kèm:

- `g1`: lề rộng **2.2126**, có **3** vectơ hỗ trợ.
- `g2`: không chuẩn hoá đạt **0.6000**, có chuẩn hoá đạt **0.9000** trên dữ liệu ba cột.
- `g3`: accuracy ngưỡng một chiều **0.7143**, sau khi thêm x bình phương **1.0000**; SVM RBF đạt **0.9200**, có **89/225** vectơ hỗ trợ.
- `g4`: chọn **C=10, gamma=1**, điểm CV **0.8711**, accuracy kiểm tra **0.9200**.
- `g5`: accuracy **0.9911**, sai **4/450** ảnh, giữ **626/1347** vectơ hỗ trợ.
- Bài tập 3: đa thức bậc **3**, C=1 tốt nhất theo CV (**0.8178**), accuracy kiểm tra **0.8267**, chưa vượt RBF.
- Bài tập 4: chọn **C=30, gamma=1**, CV **0.8756**; accuracy kiểm tra vẫn **0.9200**.
- Bài tập 5: các cặp nhầm theo chiều nhãn thật sang dự đoán là **9 sang 3**, **8 sang 1**, **4 sang 8**, **7 sang 9**.

Nhận xét:

- Các vectơ hỗ trợ của ví dụ 24 điểm nằm trên hai mép lề f(x)=+1 và f(x)=-1. Khoảng cách đến ranh giới là 1/||w||, bằng nửa độ rộng lề 2/||w||, sai khác nhỏ có thể do tối ưu số.
- Điểm (-1, -1) nằm xa ranh giới trong vùng lớp 1 nên không làm thay đổi đáng kể đường phân chia. Điểm (4.5, 4) vi phạm lề cũ và trở thành vectơ hỗ trợ, khiến ranh giới đổi.
- Khi các đặc trưng khác thang đo, chuẩn hoá giúp tránh để cột chi tiêu chi phối khoảng cách. Với ảnh chữ số, các pixel đã cùng thang 0-16; chuẩn hoá không bảo đảm tăng accuracy.
- Gamma quá lớn có thể làm accuracy tập học tăng nhưng accuracy kiểm tra giảm. Dò lưới mịn hơn cũng không bảo đảm cải thiện accuracy kiểm tra, dù điểm CV có tăng.
- Các ảnh chữ số bị nhầm có độ phân giải thấp và nét viết dễ giống chữ số khác. Xem các ảnh đã lưu cùng nhãn thật và nhãn dự đoán để giải thích từng trường hợp.
