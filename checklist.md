# Checklist hoàn thành đồ án UIT-ViQuAD 2.0

Cập nhật sau khi đối chiếu `CanPhaiHoanThanh.md`, mã nguồn và kết quả đã lưu ngày 09/11/2026.

- `[x]`: đã có bằng chứng trong mã nguồn, kết quả lưu hoặc xác nhận trước đó của người dùng.
- `[ ]`: chưa làm, chưa hoàn thiện hoặc còn phải xác nhận. Có file kết quả chưa đồng nghĩa kết quả đã được kiểm tra và chốt để báo cáo.
- Thực hiện theo thứ tự các mục dưới đây. Các lượt train đã có được giữ như kết quả thực nghiệm ban đầu; chỉ chạy lại khi thay đổi dữ liệu, nhãn, tiền xử lý hoặc cấu hình huấn luyện ảnh hưởng đến chúng.
- Số liệu đang có là kết quả trên **dev**, không phải điểm test độc lập. Không dùng `null` thay cho điểm 0%.

## 1. Đối chiếu yêu cầu và chốt cách đánh giá

- [x] Đọc và đối chiếu đầy đủ `CanPhaiHoanThanh.md`: (A) baseline cổ điển; (B) Zero-shot không fine-tune; (C) ba kiến trúc khác nhau, mỗi kiến trúc ít nhất ba seed và báo cáo mean +- std; (D) demo mô hình tốt nhất trong ba kiến trúc; phân tích ít nhất ba nhóm lỗi; hai báo cáo tiến độ và báo cáo cuối.
- [x] Xác định yêu cầu nâng cao riêng của đề tài: giữ và xử lý câu hỏi unanswerable, báo cáo accuracy phát hiện có/không có đáp án. Đây là yêu cầu để đạt mức điểm giỏi được nêu trong đề tài.
- [x] Xác nhận `test.json` hiện có 7.301 mẫu và cả 7.301 trường `answers` đều là `null`; `is_impossible=False` trong bản này không đủ để xem là nhãn thật. Không thể tính EM/F1 hoặc accuracy Test từ file hiện tại.
- [ ] Ghi rõ trong báo cáo: các bảng so sánh có nhãn hiện được đánh giá trên dev 3.814 mẫu; dev đã được dùng chọn checkpoint nên không gọi đây là phép đánh giá test độc lập.
- [ ] Nếu cần một tập test có nhãn độc lập: lấy được nhãn test chính thức, hoặc thiết kế tập giữ lại từ dữ liệu có nhãn trước khi huấn luyện/chọn mô hình. Nếu tự chia, kiểm soát trùng context/bài viết, khóa tập này và chạy lại các thí nghiệm bị ảnh hưởng. Đây là nhánh công việc có điều kiện; tài liệu yêu cầu không bắt buộc phải chấm riêng split Test 7.301 mẫu trên mirror Hugging Face.
- [x] Xác nhận bảng phân công bốn vai trò: baseline + Zero-shot; XLM-R; mDeBERTa; PhoBERT + demo. Mỗi thành viên phải phụ trách trọn vẹn phần được giao theo yêu cầu.

## 2. Kiểm tra chất lượng và làm sạch dữ liệu — ưu tiên thực hiện trước

### 2.1. Những gì đã có và đã kiểm tra

- [x] `data/raw/train.json`: bản trước làm sạch, 8.000 mẫu; 5.417 có đáp án, 2.583 không có đáp án; lấy mẫu với seed 1024.
- [x] `data/raw/dev.json`: bản trước làm sạch, 3.814 mẫu; 2.653 có đáp án, 1.161 không có đáp án.
- [x] `data/processed/test.json`: 7.301 mẫu, JSON Lines hợp lệ; giữ riêng để tạo dự đoán khi thiếu nhãn chuẩn.
- [x] Đã có kiểm tra số mẫu, thống kê nhãn và EDA trong notebook 01. `q.lstrip()` trong `src/data_preprocessing.py` chỉ bỏ khoảng trắng đầu câu hỏi; đây chưa phải bước làm sạch đầy đủ.
- [x] Rà soát file train/dev hiện tại: không thấy câu hỏi/context thiếu hoặc chỉ gồm khoảng trắng; không thấy nhãn khác kiểu bool; không thấy độ dài hai danh sách đáp án và offset lệch nhau.
- [x] Phát hiện lỗi offset: train có 48 mẫu có đáp án không khớp vị trí; dev có 22 mẫu có ít nhất một đáp án không khớp, trong đó 13 mẫu sai ở đáp án đầu tiên. Đây là phát hiện kiểm tra, **chưa phải đã sửa**.
- [x] Kiểm tra trùng bằng cặp `(context, question)` nguyên văn: train/dev không có mẫu lặp nội bộ; Test có 110 nhóm lặp, tương ứng 110 dòng dư; train/dev giao nhau 0 mẫu QA, train/test 2, dev/test 0.
- [x] Kiểm tra context nguyên văn: train/dev giao nhau 0; train/test 406; dev/test 133. Checklist cũ ghi 411/134 nhưng chưa lưu rõ quy tắc chuẩn hóa; cần thống nhất định nghĩa trước khi đưa số liệu vào báo cáo.

### 2.2. Công việc cần bổ sung

- [x] Theo quyết định người dùng, chuyển nguyên trạng train/dev chưa sạch sang data/raw; kiểm tra SHA-256 trước/sau để xác nhận nội dung không đổi. Notebook 01 giữ raw đã chốt khi chạy lại và chưa xuất train/dev chưa sạch vào processed.
- [x] Đã tạo mã Bước 5 notebook 01: sửa offset khớp duy nhất và ba trường hợp train đã đối chiếu context, giữ bản raw, kiểm tra trước/sau và ghi reports/data_cleaning_report.json; Bước 9–10 xuất processed có điều kiện.
- [x] Người dùng đã chạy notebook 01 tới Bước 12; xác nhận processed/train 8.000 và processed/dev 3.814 có 0 lỗi. Báo cáo làm sạch ghi đã sửa 48 offset train, 22 dev, không có ca tồn đọng; SHA-256 các file xuất khớp báo cáo và raw vẫn khớp snapshot trước chuyển.
- [x] Đã thêm Bước 3 kiểm tra nguồn/raw/processed, Bước 5 làm sạch và Bước 12 kiểm tra sau xuất trong notebook 01; các bước 1–12 có đường phân cách. Mã và đầu ra đã lưu đều được kiểm tra.
- [x] Người dùng đã chạy bước kiểm tra và gửi kết quả 48 mẫu train lỗi, 22 mẫu dev lỗi, Test thiếu 7.301 nhãn; đã có reports/data_quality_report.json. Chưa xác nhận một lượt chạy lại toàn notebook từ kernel sạch.
- [x] Đã chạy kiểm tra kiểu dữ liệu, chuỗi rỗng, cấu trúc answers, độ dài text/answer_start, nhãn và vị trí đáp án trên raw/processed. Phạm vi này chưa xác nhận toàn bộ chất lượng ngữ nghĩa hoặc giải quyết mọi vấn đề PhoBERT.
- [x] Kiểm tra tất cả đáp án bằng context[start:start + len(text)] == text; bản processed train/dev không còn lỗi offset, có danh sách trước/sau và lý do trong data_cleaning_report.json.
- [x] Đã sửa 48 offset train: 45 vị trí khớp duy nhất và ba vị trí đã đối chiếu context; sửa 22 offset dev bằng vị trí khớp duy nhất. Không đổi answer_text, context, question, ID, nhãn hay thứ tự/số mẫu.
- [ ] Chốt cách xử lý mẫu QA lặp và ghi rõ quy tắc so sánh chuỗi. Context dùng chung không tự động đồng nghĩa cùng câu hỏi; không âm thầm xóa mẫu khỏi tập chuẩn.
- [x] Đối chiếu raw/processed xác nhận giữ nguyên dấu tiếng Việt, dấu câu và nguyên văn context/question/answer_text; chỉ answer_start của 48 mẫu train và 22 mẫu dev thay đổi.
- [x] Giữ answers=null của Test để ghi nhận thiếu nhãn; không tạo nhãn/điểm số giả. Test vẫn có 7.301 mẫu thiếu gold.
- [x] Có reports/data_cleaning_report.json với ID, offset cũ/mới, cách sửa, lý do, lỗi trước/sau, trạng thái xuất và SHA-256 raw/processed. Raw còn nguyên 48/22 lỗi gốc để đối chiếu.
- [ ] Xử lý khác biệt ở PhoBERT: output notebook 06 hiện ghi giữ 7.928/8.000 mẫu train, loại 72 mẫu; dev giữ đủ 3.814 nhưng 33 mẫu lỗi ánh xạ bị gán nhãn loss CLS. Cần xử lý ánh xạ hoặc công bố rõ các ngoại lệ; không trình bày rằng cả ba mô hình đã học trên đúng cùng 8.000 mẫu hợp lệ.
- [ ] Chốt bộ dữ liệu dùng chung sau kiểm tra; nếu thay đổi số mẫu/nhãn thì cập nhật các điều kiện kiểm tra số lượng, không ép dữ liệu phải giữ tỷ lệ 5.417/2.583 bằng cách sửa nhãn.
- [ ] Lưu thông tin phiên bản dữ liệu: nguồn/revision, seed chọn mẫu, danh sách ID, số mẫu, cách sinh train/dev/test và mã băm SHA-256 của các file. Thông tin này dùng để gắn đúng kết quả với đúng dữ liệu.

## 3. EDA và tiền xử lý QA sau khi chốt dữ liệu

- [x] Notebook 01 đã tính độ dài câu hỏi/context, thống kê mô tả và vẽ phân bố nhãn, phân bố độ dài trên tập train con.
- [x] Có ảnh `reports/images/label_distribution.png` và `length_distribution.png`.
- [x] Đã chạy lại EDA trong notebook 01 sau làm sạch; số mẫu, văn bản và nhãn giữ nguyên nên các thống kê độ dài/phân bố nhãn không đổi.
- [ ] Thêm hoặc chạy xuất lại ảnh EDA trong reports/images: notebook hiện chỉ plt.show(), không savefig; hai PNG trên đĩa vẫn mang thời điểm 20/10/2026.
- [ ] Viết rõ độ dài `len(text.split())` là số đơn vị ngăn bằng khoảng trắng; không đồng nhất với từ tiếng Việt đã tách hay subword của tokenizer.
- [x] Có `src/data_preprocessing.py`: tokenization, sliding window, offset mapping, nhãn start/end và CLS cho cửa sổ không chứa đáp án.
- [x] Có notebook `check_data_preprocessing.ipynb`; người dùng đã xác nhận chạy kiểm tra trước đây.
- [ ] Bổ sung/kiểm tra các điều kiện an toàn trước khi token hóa: câu hỏi/context hợp lệ; cả `answers.text` và `answers.answer_start` có phần tử trước khi truy cập `[0]`; mặc định nhãn dựa trên số mẫu gốc, không dựa trên số cửa sổ.
- [ ] Tạo danh sách câu hỏi đã `lstrip()` riêng thay vì gán ngược vào batch đầu vào; chú thích đúng rằng `lstrip()` chỉ bỏ khoảng trắng đầu chuỗi.
- [ ] Sửa ghi chú còn sai về `max_length=384` tính cả question và token đặc biệt, context không mặc định được đủ 384 token; stride là số token chồng lặp; padding phục vụ gom batch cả CPU/GPU; `only_second` không bảo đảm xử lý mọi câu hỏi dài tùy ý.
- [ ] Kiểm tra lại các trường hợp QA sau sửa: câu có đáp án, không có đáp án, nhiều đáp án chuẩn, đáp án gần ranh giới cửa sổ và đoạn văn dài. Với PhoBERT, kiểm tra thêm tách từ và ánh xạ về văn bản gốc.

## 4. Thống nhất cách chấm điểm trước khi chốt thí nghiệm

- [ ] Sửa decoder Zero-shot notebook 03: chỉ chọn token thuộc context bằng sequence_ids == 1; bộ lọc hiện dựa vào offset/kích thước context nên lọt token câu hỏi. Kiểm tra span ký tự hợp lệ trước khi cập nhật best_score/best_text. Sau sửa cần chạy lại Zero-shot, chưa chốt metrics hiện tại.
- [ ] Chốt/ghi rõ chính sách no-answer Zero-shot: hiện lấy CLS max qua cửa sổ và DELTA=2.0, khác decoder fine-tune. Lưu DELTA, cách gộp CLS, n_best và max answer length vào config; nếu thay đổi chính sách phải chạy lại inference/chấm điểm tương ứng.
- [x] Có mã giải mã span và chấm EM/F1 trong `src/evaluation.py`.
- [ ] Dùng cùng cách chuẩn hóa và chấm EM/F1 cho baseline, Zero-shot và cả ba kiến trúc. Hiện notebook 03/06 dùng chuẩn hóa khác `src/evaluation.py`; đối chiếu yêu cầu SQuAD trong rubric và ghi rõ nếu điều chỉnh cho tiếng Việt.
- [ ] Giải quyết chênh lệch Zero-shot: `zeroshot_metrics.json` có HasAns F1 = 58,75446%, nhưng file phân tích dev chấm lại thành 58,92969%. Chốt một quy tắc và tính lại từ predictions, không chọn con số thuận lợi hơn.
- [ ] Thống nhất định nghĩa no-answer khi đánh giá phân loại: `prediction_text == ""` so với `is_impossible`. Nếu dùng `.strip()` hoặc chuẩn hóa khác thì phải công bố và áp dụng đồng nhất.
- [ ] Phân biệt **accuracy có/không có đáp án** = `(TN + TP) / N` với **recall NoAns** = `TP / (TP + FN)`; không lấy NoAns EM rồi gọi đó là accuracy trên toàn bộ dữ liệu.
- [ ] Giải thích trường hợp mDeBERTa: seed 42 có 103 và seed 123 có 114 câu NoAns được dự đoán chuỗi không rỗng nhưng rỗng sau chuẩn hóa dấu câu/khoảng trắng. Vì vậy NoAns EM có thể khác recall dựa trên chuỗi rỗng nguyên bản.
- [ ] Ghi rõ cách tính std và dùng thống nhất. Các summary hiện dùng độ lệch chuẩn quần thể, `ddof=0`; báo cáo theo dạng mean +- std qua ba seed.
- [ ] Lưu cấu hình chấm điểm và decoder: max answer length, số token tốt nhất được xét, ngưỡng CLS/no-answer, chuẩn hóa và quy tắc ghép nhiều cửa sổ. PhoBERT đang có giới hạn span khác XLM-R/mDeBERTa; cần giải thích khác biệt.

## 5. Đối chiếu baseline, Zero-shot và chín lượt fine-tune

### 5.1. Những kết quả đã có

- [x] Baseline Jaccard đã chạy trên 3.814 mẫu dev; có metrics, dự đoán và ví dụ lỗi.
- [x] Baseline TF-IDF + Logistic Regression bằng scikit-learn đã chạy; có `baseline_tfidf_logreg_metrics.json`, predictions và `baseline_comparison_metrics.json`.
- [x] Notebook 02 đã có accuracy/confusion matrix cho cả hai baseline. Jaccard accuracy = 69,56% nhưng recall NoAns = 0%; TF-IDF + Logistic Regression accuracy = 58,36%, recall NoAns = 23,43%.
- [x] Zero-shot có dự đoán, metrics và `zeroshot_config.json`; đã có xác nhận trước đây về chạy kernel sạch trên đủ 3.814 mẫu và không lỗi inference.
- [ ] Hoàn thiện việc ghi nhãn/tách phần thử nghiệm trong notebook 03: hiện tiêu đề ghi PhoBERT, checkpoint thực là bert-base-multilingual-cased; Markdown nói không huấn luyện nhưng cell trainer.train() đã thực thi. Phần này không phải Zero-shot hoặc kiến trúc PhoBERT chính thức.
- [x] XLM-RoBERTa-base đủ seed 42, 123, 1024; có checkpoint, metrics, predictions và summary.
- [x] mDeBERTa-v3-base đủ seed 42, 123, 1024; có checkpoint, metrics, predictions và summary.
- [x] PhoBERT-base-v2 đủ seed 42, 123, 1024; có checkpoint, metrics, predictions và summary. Không còn thiếu ba lượt chạy như checklist cũ; còn phải xử lý tính nhất quán dữ liệu/metrics ở các mục trên.
- [x] Kiểm tra chín file dự đoán fine-tune: mỗi file đủ 3.814 ID dev, không lệch tập ID; `ground_truths` khớp đáp án trong dev hiện tại.

### 5.2. Công việc cần hoàn thiện

- [ ] Gắn phiên bản dữ liệu và cấu hình vào từng run để xác nhận kết quả hiện có thuộc đúng dữ liệu đã chốt. Hiện metrics chưa lưu mã băm dữ liệu, nên chưa đủ chứng cứ cho việc train trên cùng một phiên bản.
- [ ] Sau các mục 2–4, xác định run nào cần train lại. Chỉ đổi cách tổng hợp/chấm điểm thì ưu tiên chấm lại predictions đã lưu; thay đổi dữ liệu/nhãn/tiền xử lý huấn luyện thì chạy lại các run bị ảnh hưởng.
- [ ] Nếu chạy lại với dữ liệu/cấu hình mới, dùng thư mục kết quả phiên bản mới hoặc cơ chế nhận biết phiên bản. `train_qa.py` hiện bỏ qua run khi đã có `metrics.json` và `best_model`, nên có nguy cơ tái dùng kết quả cũ sau khi thay dữ liệu.
- [x] Đối chiếu notebook 04 (không tính Test): GPU/BF16 sẵn sàng; ba seed đúng; Trainer chọn theo F1 và decoder lọc offset không thuộc context. Tuy nhiên đầu ra cell chạy hiện ghi giữ nguyên checkpoint cho cả ba seed; metrics từng seed từ 20/10/2026, trước báo cáo làm sạch 09/11/2026. Chưa có bằng chứng đã train lại bản cleaned; không dùng các số này như kết quả huấn luyện mới.
- [ ] Kiểm tra/checkpoint-selection khi thay cách chấm F1: không gọi một checkpoint là tốt nhất theo metric mới nếu trước đây nó được chọn bằng metric khác mà chưa đối chiếu.
- [ ] Bổ sung cấu hình cần cho báo cáo: learning rate, batch vật lý, gradient accumulation, batch hiệu dụng, epoch yêu cầu, epoch/checkpoint tốt nhất, seed, tokenizer, max length, stride và môi trường. PhoBERT metrics hiện thiếu trường best epoch/batch hiệu dụng; XLM-R seed 42 thiếu thời gian train nếu cần báo cáo thời gian.
- [ ] Phân tích mDeBERTa: seed 1024 trả chuỗi rỗng ở 3.794/3.814 câu dev, còn HasAns EM gần 0. Kiểm tra nhãn/features, loss, decoder, điểm CLS và ngưỡng; tách chứng cứ khỏi giả thuyết. Điểm NoAns cao không tự chứng minh mô hình tốt.
- [ ] Chốt bảng ba seed và mean +- std sau khi thống nhất cách chấm. Summary hiện tại là kết quả ban đầu: XLM-R F1 49,66 +- 0,98; mDeBERTa F1 21,81 +- 6,01; PhoBERT F1 56,20 +- 0,23. Không chép số cũ khác các file hiện tại.

## 6. Đánh giá tổng hợp trên dev và phân tích lỗi

- [x] Notebook 07 có mã đọc dự đoán, kiểm tra ID, tính HasAns/NoAns EM/F1 và accuracy/confusion matrix.
- [x] File `results/dev_answerability_and_group_metrics.json` hiện có Jaccard, TF-IDF + Logistic Regression, Zero-shot và ba seed mỗi mô hình XLM-R/mDeBERTa.
- [ ] Chạy lại notebook 07 sau khi chốt scorer để bổ sung **PhoBERT ba seed** vào file kết quả. Mã đã tự tìm PhoBERT nhưng file tổng hợp hiện chưa chứa phương pháp này.
- [ ] Có bảng Overall EM/F1, HasAns EM/F1, NoAns EM/F1, accuracy và confusion matrix cho tất cả phương pháp trên cùng dev; fine-tune trình bày từng seed và mean +- std phù hợp.
- [ ] Đọc ma trận nhầm lẫn theo cùng thứ tự: hàng nhãn thật `[HasAns, NoAns]`, cột dự đoán `[HasAns, NoAns]`; khi NoAns là lớp dương thì ma trận là `[[TN, FP], [FN, TP]]`.
- [ ] Phân loại ít nhất ba nhóm lỗi có định nghĩa rõ, ví dụ: trả lời khi thực tế không có đáp án; trả rỗng khi thực tế có đáp án; chọn sai đoạn hoặc sai biên đáp án. Nếu phân tích riêng lỗi tách từ/offset thì ghi thành nhóm bổ sung.
- [ ] Lưu ví dụ thực tế cho từng nhóm: ID, câu hỏi, context, đáp án chuẩn, dự đoán, seed/mô hình và giải thích. So sánh cùng câu hỏi giữa **cả ba kiến trúc**, không chỉ viết lý thuyết chung.
- [ ] Thống kê số lượng/tỷ lệ lỗi và nhận xét kiến trúc tốt/kém, độ ổn định qua seed. Mọi kết luận như chưa hội tụ, learning rate chưa tối ưu hoặc quên kiến thức cần bằng chứng; nếu chưa có thì ghi là giả thuyết.

### Trạng thái riêng của Test

- [x] XLM-R đã tạo dự đoán cho cả ba seed trên 7.301 mẫu Test; có `results/test_evaluation/xlm-roberta/seed-*/predictions.json` và summary.
- [x] Summary Test ghi `metric_samples_per_seed=[0,0,0]`, EM/F1/accuracy là `null`; đây là chưa có nhãn để chấm, không phải điểm 0%.
- [ ] Sửa mô tả còn nói đã tính metrics trên Test trong notebook 04, tài liệu và báo cáo. Ghi đây là bước tạo dự đoán khi Test thiếu nhãn.
- [ ] Chỉ khi nhận được nhãn chuẩn hoặc hoàn thành phương án tập giữ lại ở mục 1 mới đánh giá test độc lập; dùng cùng tập và scorer cho mọi phương pháp. Không coi việc đánh giá Test có nhãn là đã hoàn thành.

## 7. Chọn mô hình và hoàn thiện demo

- [x] Có mã giao diện Streamlit ở `demo/app.py` và `demo/testStreamlit.py`, tải mô hình và hỗ trợ dự đoán span/no-answer.
- [ ] Chọn mô hình demo tốt nhất **trong ba kiến trúc fine-tune** bằng metric dev đã thống nhất và độ ổn định. Hiện summary PhoBERT cao nhất nhưng chưa chốt vì còn khác biệt tiền xử lý/scorer; cả hai app hiện ưu tiên XLM-R seed 42.
- [ ] Tích hợp đúng checkpoint đã chọn. Nếu chọn PhoBERT, phải dùng tách từ, tokenizer và cách ánh xạ/giải mã tương ứng; không chỉ đổi đường dẫn trong pipeline XLM-R hiện tại.
- [ ] Hiển thị rõ mô hình thực sự đang dùng; nếu rơi về Zero-shot dự phòng thì ghi rõ, tránh trình diễn fallback như mô hình fine-tune tốt nhất.
- [ ] Người dùng chạy thử demo với câu có đáp án, không có đáp án, context dài và câu dễ gây nhầm; lưu ví dụ minh họa thành công và thất bại.
- [ ] Lưu ảnh/video demo đang chạy; chuẩn bị hướng dẫn chạy và đường link demo theo yêu cầu nộp bài.

## 8. Báo cáo tiến độ, báo cáo cuối và bảo vệ

- [ ] Xác nhận đã nộp báo cáo tiến độ lần 1: tóm tắt bài báo/mục tiêu bằng lời nhóm; EDA; số liệu baseline + Zero-shot; tiến độ ba kiến trúc; phân công bốn vai trò.
- [ ] Xác nhận đã nộp báo cáo tiến độ lần 2: đủ ba kiến trúc x ba seed và mean +- std; ít nhất ba nhóm lỗi có ví dụ; ảnh/video/link demo; kết quả phần unanswerable nâng cao.
- [ ] Hoàn thiện báo cáo cuối với cấu trúc: (1) Giới thiệu; (2) Dataset; (3) Quy trình xử lý dữ liệu; (4) Kết quả thực nghiệm; (5) Phân tích và thảo luận; (6) Kết luận.
- [ ] Mô tả rõ nguồn dữ liệu, bài báo nền tảng, khác biệt phiên bản UIT-ViQuAD/UIT-ViQuAD 2.0, số mẫu thực tế, chọn tập con, kiểm tra/làm sạch, offset mapping và quy trình tách từ PhoBERT.
- [ ] Đưa bảng kết quả cuối cùng đã thống nhất: baseline, Zero-shot, ba kiến trúc x ba seed, mean +- std; ghi rõ split, đơn vị %, cách chấm và định nghĩa accuracy/recall NoAns.
- [ ] Báo cáo phần nâng cao bằng số liệu cụ thể: mô hình học trả lời rỗng, accuracy phân loại có/không có đáp án, confusion matrix và thảo luận lỗi.
- [ ] Thảo luận kiến trúc tốt/kém và std; nêu ảnh hưởng dữ liệu lỗi, khác biệt số mẫu PhoBERT, khác biệt scorer và việc Test chưa có nhãn. Không gán nguyên nhân chưa được chứng minh.
- [ ] Đặt phần hạn chế/hướng phát triển ở phần thảo luận hoặc mục riêng để đáp ứng rubric; phần kết luận chỉ tổng kết kết quả đạt được theo mong muốn người dùng.
- [ ] Bổ sung bảng đóng góp của từng thành viên: phần việc, sản phẩm bàn giao, tỷ lệ công sức; kèm mã nguồn và link/hướng dẫn demo.
- [ ] Chuẩn bị slide và tổng duyệt bảo vệ: giải thích baseline, ba kiến trúc, seed/mean/std, offset mapping, EM/F1, no-answer, ba nhóm lỗi và khác biệt dev/test.
- [ ] Đối chiếu báo cáo DOCX, bảng số liệu, slide và các file JSON để loại số liệu/mô tả cũ. Chưa đánh dấu đã nộp chỉ vì có file DOCX trong thư mục.

## 9. Đóng gói và tài liệu chạy lại

- [ ] Chọn và ghi đúng môi trường QA/kernel dùng để chạy; tránh nhầm kernel `rmit-c1` của dự án phụ. Lưu phiên bản thư viện hoạt động và cấu hình GPU/CUDA phục vụ tái lập.
- [ ] Bổ sung dependencies thực sự dùng vào tài liệu cài đặt: hiện `requirements.txt` thiếu `streamlit` và `matplotlib`; kiểm tra tương thích Transformers, PyTorch và datasets/fsspec.
- [ ] Sửa README còn dấu merge conflict (`<<<<<<< HEAD`, `=======`); thống nhất seed 42/123/1024, checkpoint `vinai/phobert-base-v2`, Streamlit, đường dẫn/lệnh chạy thật và cách chuẩn hóa metric. Lệnh smoke test đang trỏ đến file notebook `.py` không có trong dự án cần sửa.
- [ ] Chuẩn bị hướng dẫn từ môi trường sạch: tạo dữ liệu, kiểm tra, EDA, baseline/Zero-shot, fine-tune, phân tích dev và demo; ghi bước nào tốn thời gian và người dùng tự chạy.
- [ ] Kiểm tra bản nộp chứa mã nguồn, cấu hình, bảng metrics, dự đoán/ví dụ lỗi, ảnh EDA/demo, báo cáo và slide; hướng dẫn lấy checkpoint nếu không đưa trọng số lên GitHub.
- [x] Cập nhật checklist và `AGENT.md` theo lần rà soát ngày 09/11/2026. Các mục chưa có bằng chứng hoàn thành vẫn để trống.

## Trình tự công việc tiếp theo

1. Sửa decoder notebook 03 để loại token câu hỏi, lọc span trước khi cập nhật điểm; chốt/lưu chính sách CLS và DELTA. Chỉnh phần thử nghiệm mBERT đang mang tiêu đề PhoBERT và mô tả sai rằng không train.
2. Thống nhất cách chấm giữa notebook 02/03, src/evaluation.py và PhoBERT. Chạy lại Zero-shot sau sửa decoder; baseline có thể chấm lại dự đoán đã lưu nếu chỉ đổi scorer.
3. Hoàn thiện kiểm tra tiền xử lý và xử lý khác biệt PhoBERT; gắn hash bản cleaned vào thí nghiệm. Không dùng nhầm các checkpoint cũ vốn học từ offset trước sửa; xác định run bị ảnh hưởng cần train lại.
4. Tổng hợp đủ baseline/Zero-shot và ba kiến trúc x ba seed trong notebook 07; ghi đúng dev, EM/F1, accuracy, recall và confusion matrix.
5. Phân tích ít nhất ba nhóm lỗi, giải thích mDeBERTa và độ ổn định qua seed.
6. Chọn mô hình fine-tune tốt nhất theo quy tắc đã chốt; hoàn thiện demo và chụp bằng chứng.
7. Hoàn thiện báo cáo/slide, sửa README/dependencies và bản nộp. Xuất ảnh EDA từ notebook, xác nhận hai báo cáo tiến độ và phân công. Các việc tài liệu có thể làm song song.
