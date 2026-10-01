# Dự án: Hỏi đáp trích xuất tiếng Việt (UIT-ViQuAD 2.0)

Dự án thực hiện giải quyết bài toán trích xuất câu trả lời (Extractive Question Answering) trên tập dữ liệu tiếng Việt UIT-ViQuAD 2.0.

## Lộ trình thực hiện (Roadmap)
1. **Giai đoạn 1: Khảo sát & Tiền xử lý**: Phân tích dữ liệu (EDA), làm sạch dữ liệu và xây dựng pipeline tiền xử lý trong `src/data_preprocessing.py`.
2. **Giai đoạn 2: Baseline & Zero-shot**: Xây dựng thuật toán dựa trên từ khóa và thử nghiệm với các mô hình đa ngôn ngữ đã huấn luyện sẵn để lấy kết quả đối chứng.
3. **Giai đoạn 3: Fine-tuning**: Huấn luyện và tinh chỉnh 3 mô hình (XLM-RoBERTa, MDeBERTa, PhoBERT) trên tập dữ liệu đã chọn.
4. **Giai đoạn 4: Đánh giá & Phân tích**: So sánh kết quả giữa 3 phương pháp, phân tích lỗi và hoàn thiện báo cáo.
5. **Giai đoạn 5: Demo**: Xây dựng giao diện Gradio để trình diễn khả năng của mô hình tốt nhất.

<<<<<<< HEAD
## Cài đặt & Thiết lập
=======
##Cài đặt & Thiết lập
>>>>>>> 36a04389045fed49fa99b143d018149047918d97
Để đảm bảo tính đồng nhất và tránh xung đột thư viện, tất cả thành viên bắt buộc phải sử dụng môi trường ảo.

```bash
# 1. Clone repository
git clone https://github.com/<your-org>/uit-viquad2-extractive-qa.git
cd uit-viquad2-extractive-qa

# 2. Tạo môi trường ảo (Virtual Environment)
python -m venv venv

# Kích hoạt trên Linux/macOS:
source venv/bin/activate

# Kích hoạt trên Windows:
 .\\venv\\Scripts\\activate

# 3. Nâng cấp pip và cài đặt thư viện
pip install --upgrade pip
pip install -r requirements.txt
```

## Thiết lập Dữ liệu (Dataset Setup)
Dữ liệu gốc được lấy từ Hugging Face: `taidng/UIT-ViQuAD2.0`.
*   **Raw data**: Nằm trong `data/raw/`.
*   **Processed data**: Tập con 5.000 - 8.000 mẫu nằm trong `data/processed/`.

Để kiểm tra dữ liệu ngay sau khi cài đặt, chạy đoạn mã sau trong một file python hoặc notebook:
```python
from datasets import load_dataset
ds = load_dataset("taidng/UIT-ViQuAD2.0")
print("Train samples:", len(ds["train"]))
print("Validation samples:", len(ds["validation"]))
```

## Quy ước Kỹ thuật (Team Conventions)
*   **Seed**: Luôn sử dụng `seed: [42, 123, 2024]` khi chia dữ liệu và huấn luyện để đảm bảo kết quả đồng nhất giữa các thành viên.
*   **Định dạng Output**: Kết quả dự đoán phải lưu vào file `.json` hoặc `.csv` với các trường:
    *   `id`: Mã định danh câu hỏi.
    *   `prediction_text`: Nội dung trích xuất của mô hình.
    *   `no_answer_probability`: Xác suất không có đáp án (dùng cho mục nâng cao).
*   **Mô hình mục tiêu**:
    *   Baseline: Thuật toán Overlap từ khóa.
    *   Zero-shot: `deepset/xlm-roberta-base-squad2`.
    *   Fine-tuning: `xlm-roberta-base`, `microsoft/mdeberta-v3-base`, `vinai/phobert-base-v2.5`.

## Kiểm tra nhanh (Smoke Test)
Sử dụng lệnh sau để kiểm tra xem GPU và quy trình dự đoán cơ bản có hoạt động bình thường hay không:

```bash
# Chạy thử nghiệm suy luận nhanh Zero-shot với 10 mẫu để test môi trường
python notebooks/03_zeroshot_xlmr_squad2.py --sample_size 10
```

## Thước đo Đánh giá (Evaluation Metrics)

Hệ thống đánh giá hiệu năng dựa trên 2 độ đo tiêu chuẩn của bài toán SQuAD:
1. **Exact Match (EM):** Tỷ lệ phần trăm chuỗi dự đoán trùng khớp tuyệt đối 100% từng ký tự với câu trả lời gốc (`Exact Match = 1` khi `pred == ground_truth`, ngược lại `0`).
2. **F1-Score (Token-level):** Đo lường độ trùng lặp mức token giữa chuỗi dự đoán và đáp án chuẩn:
   * **Precision:** Tỷ lệ token trong dự đoán xuất hiện trong đáp án.
   * **Recall:** Tỷ lệ token trong đáp án được mô hình trích xuất thành công.
   * $F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
3. **Phần nâng cao (Unanswerable Questions):** Tính thêm **Accuracy** và **No-Answer F1** riêng cho các mẫu không có câu trả lời trong ngữ cảnh.

## CƠ CHẾ HOẠT ĐỘNG TOÀN THƯ MỤC(có thể chỉnh sửa nếu thấy chưa đúng, đây mới là structure gợi í)
### 1. Thư mục Module Cốt lõi (`src/`)
Chứa toàn bộ mã nguồn dùng chung cho toàn bộ dự án, bảo đảm tính tái sử dụng cao, hạn chế trùng lặp mã và đồng nhất kết quả giữa các thành viên.

#### `src/data_preprocessing.py`
* **Mục đích:** Xử lý và chuyển đổi dữ liệu thô (chuỗi ký tự, vị trí index gốc) thành cấu trúc Tensor đầu vào tương thích với các mô hình Transformer.
* **Chức năng chính:**
  * Chuẩn hóa bảng mã tiếng Việt (Unicode NFC, loại bỏ ký tự điều khiển ẩn).
  * Xử lý đoạn văn bản dài vượt quá độ dài tối đa (`max_seq_length = 384`) bằng kỹ thuật trượt cửa sổ (`doc_stride = 128`).
  * Ánh xạ vị trí đáp án (`offset_mapping`): Chuyển đổi vị trí ký tự gốc (`answer_start` dạng character index) sang vị trí token (`start_positions`, `end_positions` dạng token index).
  * Xử lý câu hỏi không có đáp án (`unanswerable`): Gán vị trí `start_positions = 0` và `end_positions = 0` (trỏ trực tiếp vào token `<s>` hoặc `[CLS]`).
* **Thiết kế mã nguồn & Logic:**
  * Khai báo Tokenizer với thuộc tính `return_offsets_mapping=True` và `return_overflowing_tokens=True`.
  * Duyệt qua từng đoạn trượt (`features`). Nếu vùng bao của câu trả lời nằm trọn vẹn trong đoạn trượt đó, tính token index tương ứng. Nếu nằm ngoài hoặc là câu hỏi unanswerable, gán nhãn về `(0, 0)`.
* **Luồng dữ liệu (Data Flow):**
  $$\text{Input: } \{\text{context}, \text{question}, \text{answers}\} \xrightarrow{\text{Tokenizer + Offset Map}} \text{Output: PyTorch Dataset } (\text{input\_ids}, \text{attention\_mask}, \text{start\_positions}, \text{end\_positions})$$

---

#### `src/evaluation.py`
* **Mục đích:** Cung cấp bộ công cụ đo lường độ chính xác chuẩn mực theo tiêu chuẩn đánh giá của Stanford SQuAD 2.0 và UIT-ViQuAD 2.0.
* **Chức năng chính:**
  * **Exact Match (EM):** Xác định tỷ lệ phần trăm câu trả lời dự đoán trùng khớp tuyệt đối 100% từng ký tự với câu trả lời gốc.
  * **F1-Score (Token-level):** Đánh giá độ chồng lấp từ vựng giữa chuỗi dự đoán và nhãn thực tế.
  * Tiền xử lý chuỗi đánh giá: Đưa về chữ thường, loại bỏ toàn bộ dấu câu và khoảng trắng thừa trước khi so sánh.
  * Phân tách chỉ số cho hai tập: Câu hỏi có đáp án (`HasAns F1/EM`) và câu hỏi không có đáp án (`NoAns F1/EM`).
* **Thiết kế mã nguồn & Logic:**
  * `compute_exact_match(prediction, ground_truth)`: Trả về `1.0` nếu `prediction.strip() == ground_truth.strip()`, ngược lại `0.0`.
  * `compute_f1(prediction, ground_truth)`: Biến đổi chuỗi thành danh sách token, sau đó tính:
    $$\text{Precision} = \frac{\vert{}\text{Tokens}_{pred} \cap \text{Tokens}_{true}\vert{}}{\vert{}\text{Tokens}_{pred}\vert{}}, \quad \text{Recall} = \frac{\vert{}\text{Tokens}_{pred} \cap \text{Tokens}_{true}\vert{}}{\vert{}\text{Tokens}_{true}\vert{}}$$
    $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
  * Trường hợp đặc biệt: Nếu cả dự đoán và đáp án thực tế đều rỗng (dự đoán đúng câu hỏi unanswerable), gán trực tiếp $EM = 1.0, F_1 = 1.0$.
* **Luồng dữ liệu (Data Flow):**
  $$\text{Input: } [\text{predictions}], [\text{ground\_truths}] \xrightarrow{\text{Normalize \& Match}} \text{Output: } \{\text{"exact\_match"}: float, \text{"f1"}: float\}$$

---

#### `src/utils.py`
* **Mục đích:** Cung cấp các tiện ích hệ thống nhằm đảm bảo tính tái lập kết quả thực nghiệm (reproducibility) và theo dõi log.
* **Chức năng chính:**
  * Thiết lập cố định hạt giống ngẫu nhiên (Random Seed) cho toàn bộ môi trường phần cứng và phần mềm.
  * Cung cấp các hàm đọc/ghi dữ liệu an toàn dạng `.json` và `.csv`.
  * Cấu hình hệ thống Logging xuất tiến trình huấn luyện đồng thời ra màn hình terminal và file văn bản.
* **Thiết kế mã nguồn & Logic:**
  * Hàm `set_seed(seed)`: Thiết lập đồng thời `random.seed(seed)`, `np.random.seed(seed)`, `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)` và khóa tối ưu hóa ngẫu nhiên bằng `torch.backends.cudnn.deterministic = True`.

---

### 2. Thư mục Thử nghiệm & Huấn luyện (`notebooks/`)

#### `notebooks/01_eda_data_exploration.ipynb`
* **Mục đích:** Phân tích khám phá dữ liệu (EDA), phục vụ các số liệu bắt buộc trong Báo cáo Tiến độ Lần 1.
* **Chức năng chính:**
  * Thống kê kích thước phân chia các tập `train`, `validation`, `test`.
  * Định lượng tỷ lệ phân bố giữa câu hỏi có câu trả lời (`Answerable`) và không có câu trả lời (`Unanswerable`).
  * Trích xuất độ dài (min, max, median, mean) của `context`, `question` và `answer` theo số lượng từ và subword token.
  * Phân tích các dạng từ để hỏi phổ biến nhất (Ai, Ở đâu, Khi nào, Tại sao, Là gì).
* **Luồng hoạt động:** Nạp dữ liệu từ Hugging Face $\rightarrow$ Chuyển thành `pandas.DataFrame` $\rightarrow$ Trích xuất bảng số liệu và vẽ biểu đồ $\rightarrow$ Xuất ảnh trực quan vào `reports/images/`.

---

#### `notebooks/02_baseline_keyword_overlap.ipynb`
* **Mục đích:** Triển khai phương pháp Baseline A (Heuristic/Khớp từ khóa) làm mốc so sánh tối thiểu cho đề tài.
* **Chức năng chính:**
  * Tách đoạn ngữ cảnh (`context`) thành danh sách các câu đơn lẻ dựa trên dấu ngắt câu hoặc thư viện `underthesea`.
  * So sánh câu hỏi với từng câu trong đoạn văn bằng độ đo tương đồng từ vựng:
    $$\text{Jaccard}(Q, S) = \frac{\vert{}\text{Tokens}_Q \cap \text{Tokens}_S\vert{}}{\vert{}\text{Tokens}_Q \cup \text{Tokens}_S\vert{}}$$
  * Lựa chọn câu có điểm số trùng lặp cao nhất làm câu trả lời dự đoán.
* **Luồng hoạt động:** Đọc tập validation $\rightarrow$ Thực thi Heuristic Match $\rightarrow$ Gọi `src/evaluation.py` tính EM/F1 $\rightarrow$ Lưu kết quả vào `results/baseline_results.csv`.

---

#### `notebooks/03_zeroshot_xlmr_squad2.ipynb`
* **Mục đích:** Triển khai phương pháp Baseline B (Zero-shot) sử dụng checkpoint đa ngữ tiền huấn luyện `deepset/xlm-roberta-base-squad2`.
* **Chức năng chính:**
  * Đánh giá khả năng chuyển giao tri thức liên ngôn ngữ (Cross-lingual Transfer) của mô hình pre-trained trên tiếng Anh sang văn bản tiếng Việt mà không qua bước cập nhật trọng số.
* **Thiết kế mã nguồn:** Nạp pipeline `question-answering` từ Hugging Face, truyền dữ liệu theo batch để tối ưu hóa năng lực tính toán của GPU.
* **Luồng hoạt động:** Nạp model weights $\rightarrow$ Feed-forward trên tập Validation $\rightarrow$ Xuất file dự đoán và ghi nhận chỉ số vào `results/zeroshot_results.csv`.

---

#### `notebooks/04_finetune_xlm_roberta.ipynb`
* **Mục đích:** Huấn luyện tinh chỉnh mô hình đa ngữ nền tảng `xlm-roberta-base` trên tập dữ liệu UIT-ViQuAD 2.0 (Kiến trúc 1).
* **Chức năng chính:**
  * Sử dụng SentencePiece BPE Tokenizer chuẩn hóa trực tiếp ký tự chuỗi thô.
  * Khởi tạo kiến trúc `AutoModelForQuestionAnswering` (Transformer Encoder + Phân loại Span Tuyến tính ở đầu ra).
  * Huấn luyện mô hình, đánh giá sau mỗi epoch và lưu checkpoint có điểm F1 cao nhất.
  * Lặp lại quy trình với 3 random seed (`42`, `123`, `2024`) để tính giá trị trung bình và độ lệch chuẩn.
* **Thiết lập tham số:** `learning_rate = 2e-5`, `batch_size = 16`, `epochs = 3`, `fp16 = True`.

---

#### `notebooks/05_finetune_mdeberta.ipynb`
* **Mục đích:** Huấn luyện mô hình `microsoft/mdeberta-v3-base` (Kiến trúc 2) để so sánh hiệu năng kiến trúc với RoBERTa.
* **Chức năng chính:**
  * Khai thác cơ chế **Disentangled Attention** (biểu diễn phân tách giữa nội dung ngữ nghĩa và vị trí tương đối) để đánh giá khả năng nhận diện ranh giới span câu trả lời.
* **Thiết lập tham số:** Sử dụng `DebertaV2TokenizerFast`, hạ `learning_rate = 1.5e-5` để đảm bảo sự ổn định khi hội tụ loss.

---

#### `notebooks/06_finetune_phobert.ipynb`
* **Mục đích:** Huấn luyện mô hình đơn ngữ chuyên biệt cho tiếng Việt `vinai/phobert-base-v2` (Kiến trúc 3).
* **Chức năng chính:**
  * Đánh giá hiệu năng của mô hình chỉ tập trung vào ngữ liệu tiếng Việt so với các mô hình đa ngôn ngữ.
* **Điểm lưu ý kỹ thuật (Word Segmentation):**
  * PhoBERT hoạt động ở mức từ/từ ghép, bắt buộc phải phân đoạn từ (bằng `underthesea` hoặc `rdrsegmenter`) và nối từ bằng dấu gạch dưới (ví dụ: `Hà Nội` $\rightarrow$ `Hà_Nội`).
  * Phải thực hiện tái định vị lại chỉ số `answer_start` trong chuỗi ký tự mới trước khi chuyển vào tokenizer để tránh hiện tượng lệch nhãn ground truth.

---

### 3. Thư mục Dữ liệu (`data/`)

* **`data/raw/`**: Lưu trữ dữ liệu nguyên bản tải về từ Hugging Face (`train.json`, `dev.json`). Chế độ: **Chỉ đọc (Read-only)**, tuyệt đối không can thiệp để giữ nguyên trạng mốc đối chứng.
* **`data/processed/`**: Chứa tập con 5.000 – 8.000 mẫu đã qua tinh lọc, đồng bộ hóa cấu trúc nhãn, tối ưu hóa bộ nhớ cho các phiên huấn luyện trên Google Colab / GPU.
* **`data/samples/`**: Tập hợp 20–50 mẫu câu đại diện cho các ca kiểm thử điển hình (câu đơn giản, câu phức hợp, câu không có lời giải) dùng để chạy kiểm thử quy trình nhanh (Smoke test) trong vài giây.

---

### 4. Thư mục Giao diện Demo (`demo/`)

#### `demo/app.py`
* **Mục đích:** Cung cấp giao diện trực quan cho người dùng tương tác với hệ thống hỏi đáp (phục vụ nghiệm thu Giai đoạn 5).
* **Chức năng chính:**
  * Tiếp nhận đoạn văn bản ngữ cảnh (`context`) và câu hỏi (`question`) từ giao diện web.
  * Tải checkpoint mô hình có hiệu năng cao nhất trong 3 kiến trúc đã thử nghiệm.
  * Xuất câu trả lời được trích xuất trực tiếp, hiển thị điểm tin cậy (Confidence Score) hoặc cảnh báo không có câu trả lời trong văn bản.
* **Công nghệ sử dụng:** Xây dựng trên thư viện `Gradio`, hỗ trợ triển khai link chia sẻ trực tiếp (`share=True`).

---

### 5. Thư mục Kết quả & Báo cáo (`results/`, `reports/`)

* **`results/baseline_results.csv`**: Bảng tổng kết số liệu EM/F1 của phương pháp Khớp từ khóa[cite: 7].
* **`results/zeroshot_results.csv`**: Bảng tổng kết số liệu EM/F1 của mô hình Zero-shot XLM-R[cite: 7].
* **`results/finetune_logs/`**: Lưu nhật ký training loss, validation loss theo từng epoch và checkpoint đánh giá qua 3 seed của các mô hình fine-tuning[cite: 7].
* **`reports/images/`**: Lưu trữ biểu đồ phân bố EDA, biểu đồ so sánh cột giữa các mô hình để chèn vào báo cáo[cite: 7].
* **`reports/TienDo_Lan1_NhomXX.pdf`**: Bản báo cáo tiến độ chính thức nộp vào cuối Tuần 4 theo yêu cầu môn học[cite: 7].

---

### 6. Cấu hình Môi trường Gốc

* **`requirements.txt`**: Khai báo phiên bản của các thư viện bắt buộc (`torch`, `transformers`, `datasets`, `accelerate`, `evaluate`, `underthesea`, `gradio`) nhằm khóa môi trường, bảo đảm mã nguồn chạy đồng nhất trên mọi máy của các thành viên[cite: 1, 7].
* **`README.md`**: Bản đặc tả hệ thống và cẩm nang hướng dẫn gia nhập dự án (Onboarding Manual), đóng vai trò xương sống điều phối toàn bộ luồng làm việc của nhóm[cite: 7].   


## Tài liệu Tham khảo (References)
* Nguyen, K. V., Nguyen, D.-V., Nguyen, A. G.-T., & Nguyen, N. L.-T. (2020). *A Vietnamese Dataset for Evaluating Machine Reading Comprehension*. Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020), pages 2595–2608.
* Hugging Face Dataset Mirror: `taidng/UIT-ViQuAD2.0`
* Pre-trained Zero-shot Checkpoint: `deepset/xlm-roberta-base-squad2`
