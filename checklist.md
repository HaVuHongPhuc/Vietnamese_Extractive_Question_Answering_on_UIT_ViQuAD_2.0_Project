# Checklist hoàn thành đồ án UIT-ViQuAD 2.0

> Quy ước: `[x]` là đã có bằng chứng trong mã nguồn/kết quả hoặc bạn đã xác nhận; `[ ]` là còn cần làm hoặc xác minh. Tôi sẽ cập nhật checklist khi bạn báo hoàn tất hoặc cung cấp kết quả để đối chiếu.

## 1. Dữ liệu và EDA

- [x] Chốt `train.json` làm tập train chính (8.000 mẫu; 5.417 có đáp án, 2.583 không có đáp án theo lần chạy Notebook 01 gần nhất).
- [x] Có `dev.json` gồm 3.814 mẫu.
- [x] Chạy lại notebook 01 từ đầu để EDA và biểu đồ phản ánh đúng `train.json` đã chốt.
- [x] Xuất nguyên split test gốc thành `test.json` (7.301 mẫu); kiểm tra JSONL hợp lệ và số dòng train/dev/test lần lượt là 8.000/3.814/7.301.
- [x] Đã audit nội dung trùng: không có mẫu QA chính xác trùng train/dev hoặc dev/test; có 2 mẫu QA trùng train/test, 110 nhóm QA lặp trong test; context dùng chung giữa train/test là 411 và dev/test là 134. Giữ nguyên `test.json` gốc, không âm thầm loại mẫu. Ghi nhận các overlap này khi báo cáo kết quả.

## 2. A — Baseline khớp từ khóa

- [x] Chạy baseline Jaccard trong notebook 02 trên dev và lưu metrics/predictions.
- [x] Đối chiếu rubric và bổ sung baseline TF-IDF + Logistic Regression bằng scikit-learn; đánh giá cùng Jaccard trên 3.814 mẫu dev, tính EM/F1 và metric phân loại no-answer. Kết quả lưu tại results/baseline_comparison_metrics.json.

## 3. B — Zero-shot

- [x] Notebook 03 có mã Zero-shot và kết quả trên dev được lưu trong `results/`.
- [x] Chạy lại từ kernel sạch trên dev đủ 3.814 mẫu, inference errors = 0; metrics khớp lần chạy trước; đã lưu zeroshot_config.json, zeroshot_metrics.json và zeroshot_predictions.json.
- [x] Tách hoặc ghi nhãn rõ các cell fine-tune thử nghiệm trong notebook 03 để không nhầm chúng là Zero-shot.

## 4. Tiền xử lý dữ liệu cho QA

- [x] Tạo `src/data_preprocessing.py` với sliding window, offset mapping và nhãn vị trí đáp án/no-answer.
- [x] Chạy notebook `check_data_preprocessing.ipynb` để kiểm tra các hàm tiền xử lý.

## 5. C — Fine-tuning các kiến trúc

- [x] Fine-tune XLM-RoBERTa với seed 42, 123 và 1024; đã có kết quả tổng hợp mean ± std.
- [x] Fine-tune mDeBERTa với seed 42, 123 và 1024; đã có kết quả tổng hợp mean ± std.
- [ ] Kiểm tra và giải thích độ dao động lớn giữa các seed của mDeBERTa trong báo cáo.
- [ ] Hoàn thành kiến trúc thứ ba (PhoBERT) với ít nhất 3 seed; tính mean ± std.

## 6. Đánh giá cuối cùng và phân tích lỗi

-## - [ ] Accuracy của baseline A  (notebook 02)
- [ ] Đánh giá baseline, Zero-shot và các kiến trúc fine-tune trên cùng tập test đã khóa.
- [ ] Báo cáo EM và F1 trên test cho từng phương pháp/mô hình.
- [ ] Bổ sung metric phát hiện câu không có đáp án: so sánh `prediction_text == ""` với `is_impossible`, rồi tính accuracy và confusion matrix cho từng mô hình. Hiện decoder đã dự đoán chuỗi rỗng bằng điểm CLS/ngưỡng 0, nhưng `src/evaluation.py` và `metrics.json` chỉ báo cáo EM/F1.
- [ ] Phân loại ít nhất 3 nhóm lỗi và so sánh lỗi giữa các mô hình.
- [ ] Chọn mô hình demo dựa trên kết quả test và độ ổn định qua các seed.

## 7. Demo

- [ ] Hoàn thiện `demo/app.py` bằng Gradio hoặc Streamlit.
- [ ] Chạy thử demo với câu hỏi có đáp án và câu hỏi không có đáp án.

## 8. Báo cáo và nộp đồ án

- [ ] Xác nhận/hoàn thành báo cáo tiến độ lần 1.
- [ ] Xác nhận/hoàn thành báo cáo tiến độ lần 2.
- [ ] Viết báo cáo cuối: mục tiêu và nghiên cứu liên quan; dữ liệu; phương pháp; so sánh baseline, Zero-shot và fine-tuning; thảo luận; đóng góp; hạn chế và hướng phát triển; hướng dẫn chạy mã/demo.
- [ ] Thêm bảng phân công/đóng góp của từng thành viên.
- [ ] Chuẩn bị slide và nội dung bảo vệ.
- [ ] Cập nhật `AGENT.md` để trạng thái dự án khớp với các kết quả hiện tại.

## Ghi chú để tránh nhầm dữ liệu/kết quả

- Notebook 01 tạo `train.json` từ tập con train và xuất split test gốc thành `test.json`. `dev.json` đã có sẵn với 3.814 mẫu từ validation. Các file dùng JSON Lines; EDA được tính trên tập train con.
- Các kết quả XLM-RoBERTa và mDeBERTa hiện có được ghi nhận là đã chạy trên `train.json` và `dev.json`.
- Các cell fine-tune BERT thử nghiệm trong notebook 03 không thay thế yêu cầu Zero-shot hoặc kiến trúc PhoBERT.

## Việc tiếp theo

1. Xác nhận các checkpoint fine-tune hiện có được huấn luyện từ đúng `train.json` mới nhất (8.000 mẫu; 5.417/2.583); nếu dùng bản train cũ, chạy lại các seed bị ảnh hưởng.
2. Hoàn thành PhoBERT với ít nhất 3 seed theo yêu cầu đồ án.
3. Bổ sung và kiểm tra accuracy/confusion matrix cho nhãn không có đáp án trên dev; sau khi chốt cách tính, chạy đánh giá cuối trên `test.json` và báo cáo EM, F1 cùng các metric phát hiện no-answer.
4. Phân tích lỗi, hoàn thiện demo, báo cáo và slide.
