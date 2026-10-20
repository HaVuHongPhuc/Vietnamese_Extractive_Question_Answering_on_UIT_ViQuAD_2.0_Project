# Checklist hoàn thành đồ án UIT-ViQuAD 2.0

> Quy ước: `[x]` là đã có bằng chứng trong mã nguồn/kết quả hoặc bạn đã xác nhận; `[ ]` là còn cần làm hoặc xác minh. Tôi sẽ cập nhật checklist khi bạn báo hoàn tất hoặc cung cấp kết quả để đối chiếu.

## 1. Dữ liệu và EDA

- [x] Chốt `train.json` làm tập train chính (8.000 mẫu; 5.382 answerable, 2.618 unanswerable theo lần kiểm tra gần nhất).
- [x] Có `dev.json` gồm 3.814 mẫu.
- [ ] Chạy lại notebook 01 từ đầu để EDA và biểu đồ phản ánh đúng `train.json` đã chốt.
- [ ] Xuất tập test riêng từ dataset gốc; ghi lại cách tạo và kiểm tra không trùng với train/dev.

## 2. A — Baseline khớp từ khóa

- [x] Chạy baseline Jaccard trong notebook 02 trên dev và lưu metrics/predictions.
- [ ] Đối chiếu yêu cầu rubric: nếu cần baseline TF-IDF hoặc mô hình truyền thống scikit-learn, bổ sung và đánh giá baseline đó.

## 3. B — Zero-shot

- [x] Notebook 03 có mã Zero-shot và kết quả trên dev được lưu trong `results/`.
- [ ] Chạy lại từ kernel sạch để xác nhận kết quả tái lập và lưu cấu hình/métrics cuối cùng.
- [ ] Tách hoặc ghi nhãn rõ các cell fine-tune thử nghiệm trong notebook 03 để không nhầm chúng là Zero-shot.

## 4. Tiền xử lý dữ liệu cho QA

- [x] Tạo `src/data_preprocessing.py` với sliding window, offset mapping và nhãn vị trí đáp án/no-answer.
- [x] Chạy notebook `check_data_preprocessing.ipynb` để kiểm tra các hàm tiền xử lý.

## 5. C — Fine-tuning các kiến trúc

- [x] Fine-tune XLM-RoBERTa với seed 42, 123 và 1024; đã có kết quả tổng hợp mean ± std.
- [x] Fine-tune mDeBERTa với seed 42, 123 và 1024; đã có kết quả tổng hợp mean ± std.
- [ ] Kiểm tra và giải thích độ dao động lớn giữa các seed của mDeBERTa trong báo cáo.
- [ ] Hoàn thành kiến trúc thứ ba (PhoBERT) với ít nhất 3 seed; tính mean ± std.

## 6. Đánh giá cuối cùng và phân tích lỗi

- [ ] Đánh giá baseline, Zero-shot và các kiến trúc fine-tune trên cùng tập test đã khóa.
- [ ] Báo cáo EM và F1 trên test cho từng phương pháp/mô hình.
- [ ] Báo cáo riêng khả năng phát hiện câu không có đáp án: accuracy và confusion matrix cho từng mô hình.
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

- Notebook 01 hiện chỉ xuất `train.json`; các notebook tiền xử lý và fine-tuning cần đọc cùng file này. Tập con JSONL cũ đã được xóa.
- Các kết quả XLM-RoBERTa và mDeBERTa hiện có được ghi nhận là đã chạy trên `train.json` và `dev.json`.
- Các cell fine-tune BERT thử nghiệm trong notebook 03 không thay thế yêu cầu Zero-shot hoặc kiến trúc PhoBERT.

## Việc tiếp theo

1. Chạy lại notebook 01 để EDA khớp với `train.json`.
2. Tạo và khóa tập test riêng.
3. Hoàn thành PhoBERT và các phần đánh giá/phân tích/báo cáo còn lại.
