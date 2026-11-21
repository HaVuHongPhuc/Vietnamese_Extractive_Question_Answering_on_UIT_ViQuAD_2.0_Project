# Tổng duyệt và kiểm tra bản nộp UIT-ViQuAD 2.0

## Đối chiếu yêu cầu và bằng chứng

| Yêu cầu | Minh chứng trong dự án | Việc nhóm kiểm tra |
| --- | --- | --- |
| (A) Baseline | Notebook 02, kết quả Jaccard và TF-IDF + Logistic Regression | Đối chiếu F1 22,48% và 23,15% với bảng dev |
| (B) Zero-shot | Notebook 03 và dự đoán Quy bàn giao | Kiểm tra phần trình bày đúng là dùng checkpoint QA có sẵn, không huấn luyện thêm cho lượt Zero-shot |
| (C) 3 kiến trúc × 3 seed | Notebook 04/05/06, `results/dev_model_comparison.json` | Hiện đủ seed 42/123/1024 và mean +- std |
| (D) Demo mô hình đã chọn | `demo/app.py`, PhoBERT seed 1024 và XLM-R dự phòng | Chạy ba ca dưới đây và lưu ảnh, video hoặc link |
| Phân tích lỗi | `results/dev_error_analysis.json` | Chọn một ID cho mỗi nhóm và đối chiếu ba mô hình trên cùng câu hỏi |
| Mục nâng cao | Accuracy, recall NoAns và ma trận trong bảng dev | Giải thích được ví dụ hai seed mDeBERTa luôn trả rỗng |
| Tài liệu và bản nộp | README, bảng metrics, slide và báo cáo nhóm | Xác nhận đường dẫn repo, dữ liệu được phép chia sẻ, hai báo cáo tiến độ, đóng góp bốn vai trò, báo cáo cuối, ảnh/video/link demo |

`27_24DH111441.docx` là file báo cáo tiến độ đợt 1 tìm thấy trong workspace. Nhóm xác nhận riêng tình trạng đã nộp đợt 1/2 và chốt tên, phần việc, tỷ lệ công sức để đưa vào báo cáo nhóm. Giữ báo cáo Word do nhóm soạn và đối chiếu với file JSON cuối trước khi nộp.

## Chạy và ghi minh chứng demo

1. Mở terminal ở thư mục gốc, chạy `.\.venv\Scripts\python.exe -m streamlit run demo/app.py`.
2. Trong chế độ **Tự động**, chọn từng ca, bấm **Nạp ca thử**, sau đó bấm **Trả lời**. Kiểm tra dòng checkpoint là **PhoBERT-base-v2 · seed 1024**.
3. Thử **Kiểm tra XLM-R dự phòng** với một ca và kiểm tra hiển thị cảnh báo, tên seed 42 và đúng checkpoint.
4. Lưu ảnh hoặc video có cả câu hỏi, kết quả, tên checkpoint và số cửa sổ của ca dài. Chèn ảnh/video hoặc link vào tài liệu nhóm. Địa chỉ localhost thuộc máy đang chạy, nên dùng bản ghi hoặc link do nhóm tự triển khai nếu người chấm cần mở từ máy khác.

| Ca thử | ID dev | Điểm cần đối chiếu |
| --- | --- | --- |
| Có đáp án | 0017-0029-0002 | Đáp án chuẩn 5,3%. Dự đoán PhoBERT đã lưu là 5,3 % |
| Không có đáp án | 0001-0013-0002 | Dự đoán PhoBERT đã lưu là chuỗi rỗng |
| Đoạn văn dài | 0017-0041-0003 | Trên giao diện có hơn một cửa sổ. So đáp án với cấm cho các quốc gia vỡ nợ tiền mượn trong thời Chiến tranh thế giới thứ nhất mượn thêm tiền |

Các giá trị trên là dự đoán/nhãn đã lưu để đối chiếu, và cần kiểm tra lại trên ứng dụng. Nếu kết quả khác, giữ ảnh, ID, checkpoint và số cửa sổ để khoanh vùng môi trường, tokenizer hoặc khác biệt suy luận.

## Kịch bản tổng duyệt

- 1 phút giới thiệu bài toán, nguồn dữ liệu và ca không có đáp án.
- 2 phút nói về chọn 8.000 mẫu, sửa offset, sliding window và nhãn CLS.
- 2 phút trình bày bảng dev chung, mean/std và accuracy/recall NoAns.
- 2 phút giải thích ba nhóm lỗi và vì sao chọn PhoBERT seed 1024.
- 2 phút chạy demo ba ca và mở ảnh/video dự phòng nếu cần.
- 1 phút chốt kết quả và chuyển sang phản biện.

## Câu hỏi bảo vệ

| Câu hỏi | Gợi ý trả lời |
| --- | --- |
| Offset mapping dùng để làm gì? | Đổi vị trí ký tự trong context thành vị trí token khi học. Dùng offset của token context để trích ngược khi dự đoán. |
| Khi nào EM và F1 khác nhau? | EM yêu cầu khớp toàn bộ sau chuẩn hóa. F1 cho điểm khi trùng một phần token. Ví dụ lấy thừa chức danh trong câu về Santa Anna làm EM bằng 0 nhưng F1 vẫn dương. |
| Vì sao accuracy và recall NoAns khác? | Accuracy tính trên toàn dev. Recall NoAns chỉ chia cho số câu không có đáp án thật. Hai seed mDeBERTa trả rỗng mọi câu có recall 100%, accuracy 30,44% và F1 HasAns bằng 0. |
| Seed và std nói lên gì? | Giữ cấu hình và đổi seed để quan sát dao động. Std nhỏ cho thấy điểm ít đổi trong ba lượt đã chạy. |
| Vì sao báo điểm dev thay vì Test? | Dev có đáp án chuẩn và được dùng chọn checkpoint. Test hiện thiếu gold, nên có thể lưu dự đoán mà chưa tính điểm test độc lập. |
| Tại sao chọn PhoBERT seed 1024? | Kiến trúc có F1 dev trung bình 56,01 +- 0,09% và seed 1024 cao nhất theo F1 (56,14%). Trình bày kèm hạn chế recall NoAns 34,51% theo trung bình ba seed. |
| Có thể nói BF16 gây lỗi NaN không? | Có quan sát loss NaN ở hai seed mDeBERTa, nhưng cả ba chạy BF16. Cần phân tích logits, loss và đối chứng trước khi quy nguyên nhân. |
