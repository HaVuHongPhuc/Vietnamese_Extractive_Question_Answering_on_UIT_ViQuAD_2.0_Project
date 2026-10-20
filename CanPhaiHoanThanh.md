5. Cấu trúc báo cáo
• 1. Giới thiệu bài toán.
• 2. Mô tả dataset.
• 3. Quy trình xử lý dữ liệu.
• 4. Kết quả thực nghiệm.
• 5. Phân tích và thảo luận.
• 6. Kết luận.


Yêu cầu:
(A) Baseline cổ điển (TF-IDF/luật + mô hình học máy truyền thống của scikit-learn) — 1 SV phụ trách.
(B) Zero-shot: mô hình pretrained dùng trực tiếp KHÔNG fine-tune (qua pipeline có sẵn của Hugging Face) — làm chung hoặc gộp vào phần 
của SV phụ trách baseline.
(C) Fine-tune BẮT BUỘC với CẢ 3 KIẾN TRÚC PRETRAINED KHÁC NHAU— 3 SV còn lại, MỖI NGƯỜI PHỤ TRÁCH TRỌN VẸN 1 
KIẾN TRÚC. Xem bảng gợi ý kiến trúc theo loại bài toán bên dưới. Mỗi kiến trúc phải chạy với ÍT NHẤT 3 RANDOM SEED khác nhau, báo 
cáo kết quả dạng mean ± std (không chỉ 1 con số duy nhất) — đây là thực hành chuẩn trong nghiên cứu để biết kết quả có ổn định hay do may 
mắn.
(D) Demo ứng dụng: xây 1 giao diện đơn giản bằng Gradio hoặc Streamlit để nhập văn bản và xem kết quả dự đoán của mô hình tốt nhất trong 3 
kiến trúc — không cần đẹp, chỉ cần chạy được, giúp cả nhóm và giảng viên kiểm tra định tính mô hình dễ dàng hơn là đọc số liệu khô khan.
Bảng gợi ý 3 kiến trúc theo loại bài toán:
• Phân loại/gán nhãn văn bản tiếng Anh: BERT-base, RoBERTa-base, DistilBERT (hoặc ALBERT-base).
• Phân loại/gán nhãn văn bản tiếng Việt: PhoBERT-base, XLM-RoBERTa-base, mBERT (hoặc ViDeBERTa nếu nhóm muốn thử).
• Sinh văn bản (tóm tắt/QA sinh câu/data-to-text) tiếng Anh: T5-small, BART-base, Pegasus-small (hoặc FLAN-T5-small).
• Sinh văn bản tiếng Việt: ViT5-base, BARTpho, mT5-small.
• NER/token classification: dùng cùng bảng phân loại văn bản ở trên (BERT-base/RoBERTa-base/DistilBERT cho tiếng Anh; PhoBERT/XLMR/mBERT cho tiếng Việt).
Thang điểm gợi ý (để phân hóa rõ nhóm khá/giỏi):
• Mức 5-6 điểm: chỉ hoàn thành baseline + 1 kiến trúc fine-tune (chạy 1 seed), không có demo, không phân tích lỗi.
2
• Mức 7-8 điểm: hoàn thành (A)+(B)+(C) với ĐỦ 3 kiến trúc × 3 seed (mean±std đầy đủ); có demo (D) chạy được; phân tích lỗi với ít nhất 3 loại 
lỗi cụ thể.
• Mức 9-10 điểm: đạt mức 7-8, CỘNG THÊM hoàn thành đầy đủ "Yêu cầu nâng cao" ghi trong từng đề tài, giải thích được vì sao kiến trúc nào 
tốt nhất/kém nhất và độ lệch chuẩn (std) giữa các seed nói lên điều gì về độ ổn định của mô hình, và trả lời tốt câu hỏi phản biện của giảng viên.
Lịch trình gợi ý cho 9 tuần:
• Tuần 1: đọc bài báo nền tảng + tìm hiểu dữ liệu; chia việc theo 4 vai trò (baseline+zero-shot / kiến trúc 1 / kiến trúc 2 / kiến trúc 3+demo).
• Tuần 2: cài đặt baseline + zero-shot; mỗi SV setup môi trường Colab/Kaggle riêng cho kiến trúc của mình.
• Tuần 3-4: mỗi SV fine-tune kiến trúc được giao, chạy 3 seed song song trên tài khoản riêng. → Cuối Tuần 4: nộp BÁO CÁO TIẾN ĐỘ LẦN 
1.
• Tuần 5: gộp kết quả 3 kiến trúc, tính mean±std; bắt đầu xây demo app song song.
• Tuần 6: phân tích lỗi có hệ thống (≥3 nhóm lỗi cụ thể); hoàn thiện demo app.
• Tuần 7: thực hiện phần "Yêu cầu nâng cao" riêng của đề tài (bắt buộc cho điểm 9-10). → Cuối Tuần 7: nộp BÁO CÁO TIẾN ĐỘ LẦN 2.
• Tuần 8: viết báo cáo, hoàn thiện slide.
• Tuần 9: rà soát, tổng duyệt nội bộ. → Cuối Tuần 9: nộp BÁO CÁO CUỐI CÙNG và báo cáo/phản biện.
BÁO CÁO TIẾN ĐỘ (2 lần) VÀ BÁO CÁO CUỐI CÙNG:
Mục đích: giúp giảng viên phát hiện sớm nhóm bị chậm tiến độ hoặc hiểu sai đề tài, đồng thời là căn cứ chấm điểm quá trình (không chỉ chấm 
sản phẩm cuối). Mỗi lần báo cáo nộp 1 file ngắn (không cần trình bày đẹp) qua kênh do giảng viên chỉ định.
• BÁO CÁO TIẾN ĐỘ LẦN 1 — nộp cuối Tuần 4: 
(1) Tóm tắt bài báo nền tảng và mục tiêu đề tài bằng lời của nhóm (không copy nguyên văn); 
(2) Thống kê dữ liệu đã khám phá (số mẫu, phân bố nhãn/lớp, ví dụ mẫu dữ liệu); 
(3) Kết quả baseline (A) và zero-shot (B) — có số liệu cụ thể, không phải "đang làm"; 
(4) Tiến độ fine-tune 3 kiến trúc: mỗi SV báo cáo môi trường đã setup, đã chạy được bao nhiêu seed, gặp lỗi/vướng mắc gì; 
(5) Phân công công việc rõ ràng theo 4 vai trò và mức độ hoàn thành của từng người tính đến thời điểm này.
• BÁO CÁO TIẾN ĐỘ LẦN 2 — nộp cuối Tuần 7: 
3
(1) Kết quả fine-tune ĐẦY ĐỦ cả 3 kiến trúc × 3 seed, trình bày bảng mean±std so sánh với baseline/zero-shot; 
(2) Phân tích lỗi có hệ thống với ít nhất 3 nhóm lỗi cụ thể kèm ví dụ; 
(3) Trạng thái demo app (đường link hoặc video/ảnh chụp màn hình minh họa demo đang chạy); 
(4) Kế hoạch/kết quả bước đầu của phần "Yêu cầu nâng cao" riêng của đề tài; 
• BÁO CÁO CUỐI CÙNG — nộp cuối Tuần 9: báo cáo đầy đủ dạng văn bản (docx/pdf) gồm tất cả nội dung trên đã hoàn thiện, CỘNG THÊM: 
(a) phần "Yêu cầu nâng cao" đã hoàn thành với số liệu cụ thể; 
(b) thảo luận/nhận xét cuối: kiến trúc nào tốt nhất và tại sao, độ lệch chuẩn (std) giữa các seed nói lên điều gì về độ ổn định; 
(c) bảng phân chia đóng góp cụ thể của từng thành viên (% công sức, phần việc); 
(d) mã nguồn đầy đủ (GitHub repo hoặc file nén) và đường link demo app; 
(e) hạn chế của đề tài và hướng phát triển nếu có thêm thời gian. 
Nhóm cũng cần chuẩn bị buổi báo cáo/phản biện trực tiếp trước lớp và giảng viên.
YÊU CẦU CHUNG CHO TẤT CẢ ĐỀ TÀI:
• Tìm hiểu bộ dữ liệu và bài báo nền tảng: nguồn gốc, số lượng mẫu, các nhãn/lớp, cách chia train/test, đóng góp chính của bài báo.
• Cài đặt (A) baseline, (B) zero-shot, và (C) fine-tune với ĐỦ 3 kiến trúc pretrained khác nhau × 3 seed/kiến trúc, báo cáo mean±std cho mỗi 
kiến trúc.
• Xây dựng (D) demo app (Gradio/Streamlit) chạy được với mô hình tốt nhất trong 3 kiến trúc.
• Phân tích lỗi có hệ thống: phân loại lỗi thành ít nhất 3 nhóm cụ thể, có so sánh xem 3 kiến trúc có mắc cùng loại lỗi hay không.
• Hoàn thành phần "Yêu cầu nâng cao" ghi cụ thể trong từng đề tài — bắt buộc để đạt điểm 9-10.
• Báo cáo rõ: ai phụ trách phần nào (baseline/kiến trúc 1/kiến trúc 2/kiến trúc 3/demo), các bước tiền xử lý, tham số đã dùng, và kết quả đánh giá 
bằng số liệu cụ thể


TÊN ĐỀ TÀI: Hỏi đáp trích xuất tiếng Việt (UIT-ViQuAD 2.0)
Nguồn: Nguyen, Nguyen, Nguyen, Nguyen, "A Vietnamese 
Dataset for Evaluating Machine Reading Comprehension," 
COLING, 2020 [27].
Tập dữ liệu: UIT-ViQuAD 2.0 — hơn 23.000 câu hỏi tiếng 
Việt dựa trên 174 bài Wikipedia, câu trả lời là đoạn trích trực 
tiếp trong văn bản (một phần câu hỏi không có câu trả lời —
unanswerable). Bản gốc yêu cầu ký thỏa thuận với UIT, nhưng 
có bản mirror công khai không cần đăng ký qua Hugging Face 
(`taidng/UIT-ViQuAD2.0`).
Phiên bản tiếng Việt của bài toán QA trích xuất (tương tự 
SQuAD tiếng Anh) — cho phép nhóm thực hành đúng quy 
trình "Question Answering" chính thức của Hugging Face 
nhưng bằng tiếng Việt.
Yêu cầu: lấy tập con 5.000-8.000 cặp câu hỏi-trả lời (bỏ bớt 
phần câu hỏi unanswerable để đơn giản hóa, hoặc giữ lại như 
phần nâng cao); (1) baseline = khớp từ khóa tìm câu chứa nhiều 
từ trùng câu hỏi nhất; (2) zero-shot = model QA đa ngữ 
pretrained có sẵn (`deepset/xlm-roberta-base-squad2`) áp trực tiếp không fine-tune thêm; (3) fine-tune PhoBERT hoặc XLMR-base cho extractive QA trên UIT-ViQuAD. So sánh cả 3.
Đánh giá: Exact Match, F1 (theo cách tính chính thức của 
SQuAD, có thể tái sử dụng script); phân tích lỗi.
Yêu cầu nâng cao (BẮT BUỘC để đạt điểm giỏi 8-10): bổ sung 
xử lý câu hỏi unanswerable (mô hình phải học cách trả lời 
"không có đáp án" thay vì luôn cố trích một đoạn), báo cáo 
Accuracy riêng cho việc phát hiện câu hỏi không trả lời được.

NGÔN NGỮ VÀ TOOL:
Python; Google Colab; 
underthesea/VnCoreNLP; 
Hugging Face 
Transformers/Datasets; 
Gradio/Streamlit (demo 
app)

TÀI LIỆU NGUỒN CHO DANH SÁCH ĐỀ TÀI:
[27] K. Van Nguyen, D.-V. Nguyen, A. G.-T. Nguyen, N. L.-T. Nguyen, "A Vietnamese Dataset for Evaluating Machine Reading 
Comprehension," COLING, 2020. https://aclanthology.org/2020.coling-main.233/ — Dataset công khai (bản mirror không cần đăng ký): 
https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0

