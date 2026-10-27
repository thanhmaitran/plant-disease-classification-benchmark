# Đối chiếu báo cáo Task 1 với file kết quả (task1_full_results.json)

## A. Khớp hoàn toàn với JSON (không cần sửa)
- Test: SVM acc 0,9628 / F1 0,9610; CNN acc 0,9938 / F1 0,9938
- Split 3.012 / 646 / 646; test = 300 / 300 / 46
- Healthy recall SVM 0,674 (15/46 sai), CNN 0,957
- CNN: val acc tốt nhất 99,54% tại epoch 44; epoch cuối (50) train 99,44%, val 93,65%

## B. Sai trong báo cáo (đối chiếu được từ JSON)
| Vị trí | Báo cáo ghi | Đúng theo JSON | Sửa thành |
|---|---|---|---|
| IV.A đoạn 2 và III.A | "train acc 99,44% đồng thời với val 99,54% tại epoch 44" | Epoch 44: train **98,44%**, val 99,54%. 99,44% là của epoch 50 | "train accuracy 98,44% vs val 99,54% at epoch 44" và bỏ lập luận "gần khớp nhau" hoặc viết lại theo số đúng |
| IV.A đoạn 3 | "~107 ảnh healthy train (70% của 152)" | Dataset trong kết quả có healthy = 304 → train ≈ 213 | "approximately 213 (70% of 304)" |

## C. Không khớp giữa mô tả (lấy từ code) và lần chạy thật (JSON) – chưa thể tự quyết
| Mục | Báo cáo / code trong ZIP | JSON |
|---|---|---|
| Kích thước ảnh | 96 × 96 (HOG 4.356 chiều) | **128 × 128** |
| Số epoch | tối đa 30 | **50 epoch đã chạy** |
| Early stopping | patience 5 | best tại 44, vẫn chạy đến 50 (patience 5 sẽ dừng ở 49) |
| Số ảnh | dataset trong ZIP nộp: **2.152** | 4.304 |

Kết luận: phần kết quả trong báo cáo lấy từ JSON, còn phần mô tả cấu hình lấy từ file code
khác phiên bản. Các thông số thật của lần chạy chỉ xác nhận được từ máy của cậu:
1. Mở console/notebook cũ hoặc file code đã chạy: IMG_SIZE, epochs, patience.
2. Đếm ảnh trong data_task1.zip gốc (4.304 hay 2.152?) và xem có thư mục lớp bị lặp
   không (Early_blight 2.000 ảnh = 2 × 1.000).

## D. Hệ quả cho cách viết
- Nếu data_task1.zip có **4.304 ảnh khác nhau**: báo cáo đúng về số ảnh; chỉ cần sửa B và C.
  Khi đó cần giải thích nguồn của 2.000/2.000/304 ảnh (khác bản 1.000/1.000/152).
- Nếu data_task1.zip chứa **bản nhân đôi của 2.152 ảnh**: các con số 96,28% / 99,38%
  bị thổi phồng do ảnh trùng nằm ở cả train và test. Thử nghiệm HOG+SVM của repo cho
  81,1% trên tập sạch so với 96,4% trên tập nhân đôi.
- Báo cáo đã nộp: nếu cần đính chính, nên hỏi giảng viên. Với GitHub/portfolio, chỉ dùng
  số đã xác minh (xem README của repo).

## E. Câu chữ nên bỏ hoặc nói rõ là ước lượng
- Thời gian "< 5 phút" cho SVM, "10–30 phút" cho CNN: JSON không có số đo.
- "HOG+SVM 96,28% đủ cho screening": chỉ nói sau khi xác nhận dữ liệu.
