# Kết quả chạy Lab07_DanhGiaMoHinh

Cập nhật: 2026-11-08 20:58 +0700. Python 3.14.8.

Các giá trị dưới đây được lấy trực tiếp từ stdout/stderr, không chép số mẫu trong PDF.

**Nguồn:** `sinh_vien_bo_hoc.csv` đi kèm hiện là dữ liệu mô phỏng thay thế có seed cố định. Xem README trước khi đối chiếu với PDF. Nếu đã thay CSV gốc, cập nhật ghi chú nguồn trong README.

Ảnh trong `ket_qua/` là kết xuất văn bản đầu ra, không phải ảnh chụp terminal.

## Môi trường

- numpy: 2.5.4
- pandas: 3.1.0
- scikit-learn: 1.9.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

## Dấu vân tay dữ liệu (SHA-256)

- `gia_nha.csv`: `97a74b22fb939c16f317e337e9b9d2de8a2ae678930475514bf9ea3789a8f8c3`
- `sinh_vien_bo_hoc.csv`: `7b5b9da495dc150c1afa6d272fa3401292bbf13680536d4ec0edea3aa4102a4e`

## Đầu ra từng chương trình


### code/h1_do_bua.py

[Log TXT](./ket_qua/h1_do_bua.txt) · [Ảnh kết quả](./ket_qua/h1_do_bua.png)

```text
Tran An Ky - 2474802010460 | Lab07_DanhGiaMoHinh
LENH: python code/h1_do_bua.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Kich thuoc bang: (1200, 5)
 diem_tb  so_buoi_vang  tin_chi_no  gio_lam_them  bo_hoc
     6.5             7           0             9       0
     7.0             3           1            14       0
     6.1             4           4            15       0
     5.1             2           1            17       0
     5.8             4           1            14       0
So sinh vien moi lop:
 bo_hoc
0    1081
1     119
Ty le bo hoc: 0.099
Tap kiem tra: 300 ban, 30 ban bo hoc
Mo hinh luoi      accuracy 0.9000, tim ra 0 tren 30 ban bo hoc
Hoi quy logistic  accuracy 0.9300, tim ra 15 tren 30 ban bo hoc
Nhan xet: accuracy cao chua du; can xem so ban bo hoc duoc tim ra.
```


### code/h2_ma_tran.py

[Log TXT](./ket_qua/h2_ma_tran.txt) · [Ảnh kết quả](./ket_qua/h2_ma_tran.png)

```text
Tran An Ky - 2474802010460 | Lab07_DanhGiaMoHinh
LENH: python code/h2_ma_tran.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Tu dem: TP=15, FP=6, FN=15, TN=264; tong=300
confusion_matrix:
 [[264   6]
 [ 15  15]]
Thu tu TN, FP, FN, TP: [264   6  15  15]
Accuracy: 0.9300; so ban bi bao: 21
Da luu: figures/ma_tran_nham_lan.png
```


### code/h3_precision_recall.py

[Log TXT](./ket_qua/h3_precision_recall.txt) · [Ảnh kết quả](./ket_qua/h3_precision_recall.png)

```text
Tran An Ky - 2474802010460 | Lab07_DanhGiaMoHinh
LENH: python code/h3_precision_recall.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

                 precision  recall    F1
Tinh tay:        0.7143  0.5000  0.5882
Scikit-learn:    0.7143  0.5000  0.5882
              precision    recall  f1-score   support

    khong bo     0.9462    0.9778    0.9617       270
      bo hoc     0.7143    0.5000    0.5882        30

    accuracy                         0.9300       300
   macro avg     0.8303    0.7389    0.7750       300
weighted avg     0.9230    0.9300    0.9244       300

Macro: moi lop nang nhu nhau. Weighted: trong so theo support.
```


### code/h4_nguong.py

[Log TXT](./ket_qua/h4_nguong.txt) · [Ảnh kết quả](./ket_qua/h4_nguong.png)

```text
Tran An Ky - 2474802010460 | Lab07_DanhGiaMoHinh
LENH: python code/h4_nguong.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Nguong chon tren tap hoc: 0.40; F1 OOF=0.5854
Minh hoa tren test; khong dung bang nay de chon nguong.
nguong  so bao   TP  FP  FN   precision  recall    F1
  0.50      21   15   6  15    0.7143  0.5000  0.5882
  0.40      31   19  12  11    0.6129  0.6333  0.6230
  0.30      45   22  23   8    0.4889  0.7333  0.5867
  0.20      58   23  35   7    0.3966  0.7667  0.5227
  0.10      85   26  59   4    0.3059  0.8667  0.4522
  0.40      31   19  12  11    0.6129  0.6333  0.6230
Da luu: figures/chon_nguong_oof.png
Ha nguong thuong tang recall, dong thoi tang so bao dong nham.
```


### code/h5_roc_auc.py

[Log TXT](./ket_qua/h5_roc_auc.txt) · [Ảnh kết quả](./ket_qua/h5_roc_auc.png)

```text
Tran An Ky - 2474802010460 | Lab07_DanhGiaMoHinh
LENH: python code/h5_roc_auc.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Mo hinh          AUC     recall     F1 (nguong 0.5)
Logistic        0.9154   0.5000   0.5882
ROC logistic co 42 diem; 6 diem dau: [(np.float64(0.0), np.float64(0.0)), (np.float64(0.0), np.float64(0.03333333333333333)), (np.float64(0.0), np.float64(0.2)), (np.float64(0.003703703703703704), np.float64(0.2)), (np.float64(0.003703703703703704), np.float64(0.23333333333333334)), (np.float64(0.007407407407407408), np.float64(0.23333333333333334))]
Naive Bayes     0.9126   0.3667   0.4889
KNN, k=15       0.8488   0.1667   0.2632
Cay sau 4       0.8649   0.3333   0.3922
Rung 200 cay    0.8825   0.5333   0.5714
SVM RBF         0.8493   0.3333   0.4348
AUC do kha nang xep hang; recall va F1 con phu thuoc nguong.

STDERR (canh bao/loi):
D:\Nam3_ki1\Mayhoc\.venv\Lib\site-packages\sklearn\svm\_base.py:236: FutureWarning: The `probability` parameter was deprecated in 1.9 and will be removed in version 1.11. Use `CalibratedClassifierCV(SVC(), ensemble=False)` instead of `SVC(probability=True)`
  warnings.warn(
```


### code/h6_can_bang.py

[Log TXT](./ket_qua/h6_can_bang.txt) · [Ảnh kết quả](./ket_qua/h6_can_bang.png)

```text
Tran An Ky - 2474802010460 | Lab07_DanhGiaMoHinh
LENH: python code/h6_can_bang.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Khong stratify, so ban bo hoc tren test: [32, 23, 39, 39, 30, 31, 34, 32, 31, 33]
CV accuracy  : 0.9233
CV precision : 0.6708
CV recall    : 0.4373
CV f1        : 0.5250
Tap kiem tra   precision  recall    F1
None          0.7143    0.5000  0.5882
balanced      0.3133    0.8667  0.4602
Can bang lop phat nang hon khi doan sai lop hiem; van phai xem ca precision.
```


### baitap07/bai1.py

[Log TXT](./ket_qua/bai1.txt) · [Ảnh kết quả](./ket_qua/bai1.png)

```text
Tran An Ky - 2474802010460 | Lab07_DanhGiaMoHinh
LENH: python baitap07/bai1.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Logistic {'accuracy': 0.93, 'precision': 0.7142857142857143, 'recall': 0.5, 'f1': 0.588235294117647} - khop scikit-learn
Luoi {'accuracy': 0.9, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0} - khop scikit-learn
Mo hinh luoi khong bao lop 1: TP+FP=0, precision khong xac dinh.
Quy uoc tra 0, tuong duong zero_division=0; khong phai precision do duoc.
```


### baitap07/bai2.py

[Log TXT](./ket_qua/bai2.txt) · [Ảnh kết quả](./ket_qua/bai2.png)

```text
Tran An Ky - 2474802010460 | Lab07_DanhGiaMoHinh
LENH: python baitap07/bai2.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

              precision    recall  f1-score   support

           0     1.0000    1.0000    1.0000        45
           1     0.9787    1.0000    0.9892        46
           2     1.0000    1.0000    1.0000        44
           3     0.9787    1.0000    0.9892        46
           4     1.0000    0.9778    0.9888        45
           5     1.0000    1.0000    1.0000        46
           6     1.0000    1.0000    1.0000        45
           7     1.0000    0.9778    0.9888        45
           8     0.9767    0.9767    0.9767        43
           9     0.9778    0.9778    0.9778        45

    accuracy                         0.9911       450
   macro avg     0.9912    0.9910    0.9911       450
weighted avg     0.9912    0.9911    0.9911       450

recall thap nhat: chu so [8], gia tri 0.9767
precision thap nhat: chu so [8], gia tri 0.9767
Support cac chu so gan bang nhau, nen macro va weighted gan bang nhau.
```


### baitap07/bai3.py

[Log TXT](./ket_qua/bai3.txt) · [Ảnh kết quả](./ket_qua/bai3.png)

```text
Tran An Ky - 2474802010460 | Lab07_DanhGiaMoHinh
LENH: python baitap07/bai3.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Nguong lon nhat: 0.1453320871; recall OOF=0.8090
Test: recall=0.8333, precision=0.3788, F1=0.5208, so ban bi bao=66
Recall test dat 0.8: True
Dieu kien 80% tren OOF khong bao dam 80% tren du lieu moi; khong sua nguong theo test.
```


### baitap07/bai4.py

[Log TXT](./ket_qua/bai4.txt) · [Ảnh kết quả](./ket_qua/bai4.png)

```text
Tran An Ky - 2474802010460 | Lab07_DanhGiaMoHinh
LENH: python baitap07/bai4.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Logistic: AUC=0.9154
Random Forest: AUC=0.8825
Da luu: figures/roc.png
AUC lon hon (tong the): Logistic
Hai duong doi thu tu cao/thap: True
AUC lon hon khong dong nghia tot hon tai moi nguong; xem cac doan giao tren hinh.
```


### baitap07/bai5.py

[Log TXT](./ket_qua/bai5.txt) · [Ảnh kết quả](./ket_qua/bai5.png)

```text
Tran An Ky - 2474802010460 | Lab07_DanhGiaMoHinh
LENH: python baitap07/bai5.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Mo hinh   class_weight  precision  recall    F1
Rung     None         0.6154  0.5333  0.5714
Rung     balanced     0.4651  0.6667  0.5479
Thay doi precision, recall, F1: [-0.1503, 0.1333, -0.0235]
SVM      None         0.5833  0.2333  0.3333
SVM      balanced     0.3171  0.8667  0.4643
Thay doi precision, recall, F1: [-0.2663, 0.6333, 0.131]
Thay doi recall nhieu nhat: SVM
Cac mo hinh phan ung khac nhau; can bang lop khong bao dam F1 tang.
```


### baitap07/bai6.py

[Log TXT](./ket_qua/bai6.txt) · [Ảnh kết quả](./ket_qua/bai6.png)

```text
Tran An Ky - 2474802010460 | Lab07_DanhGiaMoHinh
LENH: python baitap07/bai6.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Ban dau         MAE=0.2578 ty, RMSE=0.3730 ty, R2=0.9622
Mot can +5 ty   MAE=0.6745 ty, RMSE=1.5505 ty, R2=0.3475
Thay doi MAE, RMSE, R2: [ 0.41666667  1.17747486 -0.61476138]
Tang tuong doi MAE, RMSE (%): [161.63403668 315.64241079]
RMSE binh phuong sai so truoc khi lay trung binh nen phat nang du doan hong.
Chi so khong cung thang do: chi so sanh truc tiep MAE va RMSE (ty dong); R2 khong don vi.
```
