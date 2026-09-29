# Bài 02: Hồi quy logistic

Sáu chương trình trong thư mục này làm các bài tập của Lab02 về hồi quy logistic. Mỗi bài chạy độc lập và đọc dữ liệu trong `data/sinh_vien.csv` theo vị trí file, nên có thể chạy từ bất kỳ thư mục làm việc nào. Chú thích dùng tiếng Việt có dấu; terminal in chữ không dấu để tương thích Windows PowerShell.

## Cài đặt và chạy

Từ thư mục gốc repository, tạo môi trường và cài thư viện:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\Buoi02\02_Logistic_Regression\requirements.txt
```

Chạy từng bài từ thư mục gốc repository:

```powershell
python .\Buoi02\02_Logistic_Regression\bai1.py
python .\Buoi02\02_Logistic_Regression\bai2.py
python .\Buoi02\02_Logistic_Regression\bai3.py
python .\Buoi02\02_Logistic_Regression\bai4.py
python .\Buoi02\02_Logistic_Regression\bai5.py
python .\Buoi02\02_Logistic_Regression\bai6.py
```

Bài 2 lưu đồ thị `sigmoid.png` trong thư mục này. Chụp kết quả terminal và đồ thị để nộp theo yêu cầu giảng viên.
## Bài 1: lọc theo điểm giữa kỳ

`df[df["diem_giua_ky"] >= 7]` lấy những dòng đạt điều kiện.
`len(...)` đếm số bạn trong nhóm. Vì `qua_mon` chỉ có 0 và 1 nên `.mean()`
chính là số bạn qua chia cho tổng số bạn trong nhóm.

Kết quả:

- Có **31 bạn** đạt điểm giữa kỳ từ 7 trở lên.
- Tỷ lệ qua của nhóm này: **0.9032**, khoảng **90.32%**.
- Tỷ lệ qua của nhóm dưới 7: **0.4831**, khoảng **48.31%**.

Nhận xét: Nhóm có điểm giữa kỳ từ 7 trở lên có tỷ lệ qua môn cao hơn rõ rệt,
nên điểm giữa kỳ giúp phân biệt hai nhóm trong bộ dữ liệu này.

## Bài 2: vẽ sigmoid

`np.linspace(-8, 8, 400)` tạo 400 điểm để vẽ đường cong mượt.
`plt.axhline(0.5)` vẽ đường ngang; `plt.axvline(0)` vẽ đường dọc.
`plt.savefig("sigmoid.png", dpi=200)` lưu hình.

Đường cong tăng từ gần 0 đến gần 1 và đi qua `(0, 0.5)`.
Mở `sigmoid.png` trong VS Code để xem đồ thị sau khi chạy.

## Bài 3: tự dự đoán bằng w và b

Mô hình học từ toàn bộ 120 dòng, giống mục 4 của đề. Hàm `du_doan(gio)` làm:

1. Tính `z = w * gio + b`.
2. Tính xác suất `1 / (1 + np.exp(-z))`.
3. Gán nhãn 1 nếu xác suất >= 0.5, ngược lại gán nhãn 0.

| Giờ ôn | z         | Xác suất qua | Nhãn |
| -------- | --------- | -------------- | ----- |
| 3        | -3.886432 | 0.020106       | 0     |
| 8        | -1.922336 | 0.127601       | 0     |
| 12.89    | -0.001451 | 0.499637       | 0     |
| 18       | 2.005855  | 0.881410       | 1     |
| 26       | 5.148409  | 0.994225       | 1     |

Mốc chính xác là `-b / w ≈ 12.89369302` giờ. Khi đó `z = 0`, sigmoid bằng 0.5.
12.89 chỉ là mốc làm tròn nên xác suất gần 0.5 và vẫn nhỏ hơn 0.5 một chút.
Luôn so ngưỡng với xác suất chưa làm tròn.

## Bài 4: tính bằng tay từ ma trận nhầm lẫn

Chia dữ liệu với `test_size=0.25`, `random_state=17`, `stratify=y`, được
90 dòng huấn luyện và 30 dòng kiểm tra. Ma trận có TN=9, FP=3, FN=1, TP=17.
Lớp dương ở bài này là **qua môn (1)**.

| Thước đo | Công thức tự tính                       | Kết quả |
| ----------- | ------------------------------------------- | --------- |
| Accuracy    | `(TP + TN) / (TP + TN + FP + FN)` = 26/30 | 0.8667    |
| Precision   | `TP / (TP + FP)` = 17/20                  | 0.8500    |
| Recall      | `TP / (TP + FN)` = 17/18                  | 0.9444    |
| F1          | `2*TP / (2*TP + FP + FN)` = 34/38         | 0.8947    |

Accuracy xét tất cả dự đoán đúng.

Precision xét 20 bạn được dự đoán qua,
trong đó 17 bạn thật sự qua.

Recall xét 18 bạn thật sự qua, mô hình nhận ra 17 bạn.
F1 là trung bình điều hòa của precision và recall.

File này chỉ dùng `confusion_matrix` để lấy bốn ô, rồi tự tính các thước đo.
Cột tham chiếu lấy từ tài liệu; cả bốn kết quả khớp khi in bốn chữ số thập phân.

## Bài 5: dò ngưỡng có F1 lớn nhất

Dùng cùng cách chia dữ liệu như bài 4. Lấy xác suất qua môn, rồi thử 19 ngưỡng
từ 0.05 đến 0.95 với bước 0.05. Mỗi lần lấy nhãn bằng `p >= nguong` và tính F1.

F1 cao nhất là **0.9412**, tại cả **0.60 và 0.65**. Chương trình chọn 0.60,
là ngưỡng nhỏ nhất trong các ngưỡng đồng hạng. Ngưỡng 0.50 có F1=0.8947.

Nhận xét: Trong phép dò này, ngưỡng tốt nhất theo F1 khác 0.5.
Hai ngưỡng có thể có cùng F1 vì chúng tạo cùng bộ nhãn dự đoán trên tập kiểm tra.

## Bài 6: lớp dương là rớt môn

Giữ nguyên mô hình và dự đoán; đổi `pos_label` khi chấm điểm.

| Lớp dương  | Precision | Recall |
| ------------- | --------- | ------ |
| Qua môn (1)  | 0.8500    | 0.9444 |
| Rớt môn (0) | 0.9000    | 0.7500 |

Khi lớp dương là 0: TP mới = TN cũ = 9, FP mới = FN cũ = 1,
FN mới = FP cũ = 3, TN mới = TP cũ = 17.

- Precision lớp 0 = 9/(9+1) = 0.9: trong 10 bạn được dự đoán rớt, 9 bạn thật sự rớt.
- Recall lớp 0 = 9/(9+3) = 0.75: trong 12 bạn thật sự rớt, mô hình nhận ra 9 bạn.

Hai bộ số khác nhau vì nhóm được đếm và mẫu số thay đổi theo lớp dương.
Dự đoán và dữ liệu giữ nguyên nên accuracy vẫn là 26/30 = 0.8667.
