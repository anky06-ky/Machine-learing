# Bài 02: Hồi quy logistic

Thư mục này gồm sáu chương trình hướng dẫn `c1` đến `c6` làm theo tài liệu Lab02 và sáu bài tập `bai1` đến `bai6`. Bài toán là đoán sinh viên qua môn (nhãn 1) hay rớt môn (nhãn 0) từ số giờ ôn tập và điểm giữa kỳ. Mọi chương trình đọc dữ liệu `data/sinh_vien.csv` theo vị trí tệp, nên chạy từ thư mục nào cũng được. Chú thích viết tiếng Việt có dấu, phần in ra màn hình viết không dấu để hiện đúng trên Windows PowerShell.

## Cấu trúc thư mục

```text
02. Logistic_Regression/
├── README.md               # Hướng dẫn chạy và kết quả
├── Logistic_Regression.pdf # Tài liệu thực hành Lab02 và đề bài tập
├── requirements.txt        # Thư viện cần cài (numpy, pandas, matplotlib, scikit-learn)
├── data/
│   └── sinh_vien.csv       # 120 sinh viên: gio_on, diem_giua_ky, qua_mon
├── c1_doc_du_lieu.py       # Mục 2: đọc dữ liệu và xem qua một lượt
├── c2_sigmoid.py           # Mục 3: tự cài đặt hàm sigmoid
├── c3_khop_mo_hinh.py      # Mục 4: khớp mô hình bằng scikit-learn
├── c4_danh_gia.py          # Mục 5: ma trận nhầm lẫn và bốn thước đo
├── c5_doi_nguong.py        # Mục 6: đổi ngưỡng quyết định
├── c6_hai_bien.py          # Mục 7: thêm biến điểm giữa kỳ
└── bai1.py ... bai6.py     # Mục 8: sáu bài tập
```

## Cài đặt và chạy

Từ thư mục gốc repository, tạo môi trường và cài thư viện:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r ".\02. Logistic_Regression\requirements.txt"
```

Chạy sáu chương trình hướng dẫn theo thứ tự (đường dẫn đặt trong dấu nháy vì tên thư mục có dấu cách):

```powershell
python ".\02. Logistic_Regression\c1_doc_du_lieu.py"
python ".\02. Logistic_Regression\c2_sigmoid.py"
python ".\02. Logistic_Regression\c3_khop_mo_hinh.py"
python ".\02. Logistic_Regression\c4_danh_gia.py"
python ".\02. Logistic_Regression\c5_doi_nguong.py"
python ".\02. Logistic_Regression\c6_hai_bien.py"
```

Chạy sáu bài tập:

```powershell
python ".\02. Logistic_Regression\bai1.py"
python ".\02. Logistic_Regression\bai2.py"
python ".\02. Logistic_Regression\bai3.py"
python ".\02. Logistic_Regression\bai4.py"
python ".\02. Logistic_Regression\bai5.py"
python ".\02. Logistic_Regression\bai6.py"
```

Bài 2 lưu đồ thị `sigmoid.png` trong thư mục này. Chụp kết quả terminal và đồ thị để nộp theo yêu cầu giảng viên.

## Các chương trình hướng dẫn

| Mục | Tệp | Nội dung | Kết quả chính |
| :-: | --- | --- | --- |
| 2 | `c1_doc_du_lieu.py` | Đọc dữ liệu, đếm nhãn, so giờ ôn trung bình hai nhóm | 120 dòng, 71 qua và 49 rớt, tỷ lệ qua 0.5917, giờ ôn trung bình 8.29 (rớt) và 19.89 (qua) |
| 3 | `c2_sigmoid.py` | Bảng sigmoid, kiểm tra σ(-z) = 1 - σ(z), mô hình giả định w = 0.4, b = -5 | σ(0) = 0.5000, z = 0 ứng với 12.5 giờ |
| 4 | `c3_khop_mo_hinh.py` | Khớp `LogisticRegression` một biến, `predict_proba` cho năm bạn mới | w = 0.392819, b = -5.064890, mốc 12.89 giờ |
| 5 | `c4_danh_gia.py` | Chia 90/30 có `stratify=y`, ma trận nhầm lẫn, bốn thước đo | TN=9, FP=3, FN=1, TP=17; 0.8667, 0.8500, 0.9444, 0.8947 |
| 6 | `c5_doi_nguong.py` | Tự áp ngưỡng 0.2 đến 0.8, xem precision và recall | Ngưỡng 0.5: 20 bạn bị đoán qua, precision 0.8500, recall 0.9444 |
| 7 | `c6_hai_bien.py` | Thêm `diem_giua_ky` làm biến thứ hai | Accuracy 0.8667 lên 0.9000; hệ số +0.3035, +0.5885, chặn -6.9811 |

Quy tắc chia dữ liệu giữ nguyên ở mọi chương trình: `test_size=0.25`, `random_state=17`, `stratify=y`. Quên một trong hai tham số cuối thì kết quả sẽ khác tài liệu.

## Bài tập

### Bài 1: lọc theo điểm giữa kỳ

`df[df["diem_giua_ky"] >= 7]` lấy những dòng đạt điều kiện.
`len(...)` đếm số bạn trong nhóm. Vì `qua_mon` chỉ có 0 và 1 nên `.mean()`
chính là số bạn qua chia cho tổng số bạn trong nhóm.

Kết quả:

- Có **31 bạn** đạt điểm giữa kỳ từ 7 trở lên.
- Tỷ lệ qua của nhóm này: **0.9032**, khoảng **90.32%**.
- Tỷ lệ qua của nhóm dưới 7: **0.4831**, khoảng **48.31%**.

Nhận xét: Nhóm có điểm giữa kỳ từ 7 trở lên có tỷ lệ qua môn cao hơn rõ rệt,
nên điểm giữa kỳ giúp phân biệt hai nhóm trong bộ dữ liệu này.

### Bài 2: vẽ sigmoid

`np.linspace(-8, 8, 400)` tạo 400 điểm để vẽ đường cong mượt.
`plt.axhline(0.5)` vẽ đường ngang; `plt.axvline(0)` vẽ đường dọc.
`plt.savefig("sigmoid.png", dpi=200)` lưu hình.

Đường cong tăng từ gần 0 đến gần 1 và đi qua `(0, 0.5)`.
Mở `sigmoid.png` trong VS Code để xem đồ thị sau khi chạy.

### Bài 3: tự dự đoán bằng w và b

Mô hình học từ toàn bộ 120 dòng, giống mục 4 của đề. Hàm `du_doan(gio)` làm:

1. Tính `z = w * gio + b`.
2. Tính xác suất `1 / (1 + np.exp(-z))`.
3. Gán nhãn 1 nếu xác suất >= 0.5, ngược lại gán nhãn 0.

| Giờ ôn | z         | Xác suất qua | Nhãn |
| ------ | --------- | ------------ | ---- |
| 3      | -3.886432 | 0.020106     | 0    |
| 8      | -1.922336 | 0.127601     | 0    |
| 12.89  | -0.001451 | 0.499637     | 0    |
| 18     | 2.005855  | 0.881410     | 1    |
| 26     | 5.148409  | 0.994225     | 1    |

Mốc chính xác là `-b / w ≈ 12.89369302` giờ. Khi đó `z = 0`, sigmoid bằng 0.5.
12.89 chỉ là mốc làm tròn nên xác suất gần 0.5 và vẫn nhỏ hơn 0.5 một chút.
Luôn so ngưỡng với xác suất chưa làm tròn.

### Bài 4: tự tính bốn thước đo từ TP, TN, FP, FN

Chia dữ liệu với `test_size=0.25`, `random_state=17`, `stratify=y`, được
90 dòng huấn luyện và 30 dòng kiểm tra. Ma trận có TN=9, FP=3, FN=1, TP=17.
Lớp dương ở bài này là **qua môn (1)**.

| Thước đo  | Công thức tự tính               | Kết quả |
| --------- | ------------------------------- | ------- |
| Accuracy  | `(TP + TN) / (TP + TN + FP + FN)` = 26/30 | 0.8667 |
| Precision | `TP / (TP + FP)` = 17/20        | 0.8500  |
| Recall    | `TP / (TP + FN)` = 17/18        | 0.9444  |
| F1        | `2 * P * R / (P + R)`           | 0.8947  |

Accuracy xét tất cả dự đoán đúng.

Precision xét 20 bạn được dự đoán qua, trong đó 17 bạn thật sự qua.

Recall xét 18 bạn thật sự qua, mô hình nhận ra 17 bạn.
F1 là trung bình điều hòa của precision và recall.

Chương trình chỉ dùng `confusion_matrix` để lấy bốn ô, rồi tự tính các thước đo.
Cột tài liệu lấy từ kết quả trong Lab02; cả bốn kết quả khớp khi in bốn chữ số thập phân.

### Bài 5: dò ngưỡng có F1 lớn nhất

Dùng cùng cách chia dữ liệu như bài 4. Lấy xác suất qua môn, rồi thử 19 ngưỡng
từ 0.05 đến 0.95 với bước 0.05. Mỗi lần lấy nhãn bằng `p >= nguong` và tính F1.

F1 cao nhất là **0.9412**, tại cả **0.60 và 0.65**. Chương trình chọn 0.60,
là ngưỡng nhỏ nhất trong các ngưỡng đồng hạng. Ngưỡng 0.50 có F1=0.8947.

Nhận xét: Trong phép dò này, ngưỡng tốt nhất theo F1 khác 0.5.
Hai ngưỡng có thể có cùng F1 vì chúng tạo cùng bộ nhãn dự đoán trên tập kiểm tra.

### Bài 6: lớp dương là rớt môn

Giữ nguyên mô hình và dự đoán; đổi `pos_label` khi chấm điểm.

| Lớp dương   | Precision | Recall |
| ----------- | --------- | ------ |
| Qua môn (1) | 0.8500    | 0.9444 |
| Rớt môn (0) | 0.9000    | 0.7500 |

Khi lớp dương là 0: TP mới = TN cũ = 9, FP mới = FN cũ = 1,
FN mới = FP cũ = 3, TN mới = TP cũ = 17.

- Precision lớp 0 = 9/(9+1) = 0.9: trong 10 bạn được dự đoán rớt, 9 bạn thật sự rớt.
- Recall lớp 0 = 9/(9+3) = 0.75: trong 12 bạn thật sự rớt, mô hình nhận ra 9 bạn.

Hai bộ số khác nhau vì nhóm được đếm và mẫu số thay đổi theo lớp dương.
Dự đoán và dữ liệu giữ nguyên nên accuracy vẫn là 26/30 = 0.8667.
