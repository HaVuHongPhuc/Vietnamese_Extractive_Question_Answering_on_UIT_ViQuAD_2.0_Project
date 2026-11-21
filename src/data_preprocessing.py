"""Tiền xử lý Extractive QA cho XLM-RoBERTa và mDeBERTa.

Dữ liệu gốc lưu vị trí đáp án theo ký tự. Tokenizer chia câu hỏi và đoạn
văn thành token; module này dùng offset của token để tạo nhãn start/end.
Khi chấm điểm, module giữ ID và offset context để trích lại văn bản gốc.

Ví dụ: "Phạm Văn Đồng sinh năm 1906." có đáp án "1906" ở [23, 27).
Chỉ số token của "1906" phụ thuộc tokenizer và vị trí câu hỏi trong feature.
"""

from typing import Any

MAX_SEQ_LENGTH = 384
DOC_STRIDE = 128


def _tokenize_questions_and_contexts(
    examples: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
    doc_stride: int,
) -> Any:
    """Tạo các cửa sổ cho cặp như "Ở đâu?" và "Hà Nội rất đẹp."."""
    # Tạo list mới: lstrip bỏ khoảng trắng đầu câu hỏi. Context giữ nguyên
    # từng ký tự vì answer_start được đo trên chính chuỗi đó.
    questions = [question.lstrip() for question in examples["question"]]

    # 384 là số token tối đa của cả câu hỏi, context và token đặc biệt.
    # only_second yêu cầu cắt context nếu quá dài. Câu hỏi phải đủ ngắn để
    # còn chỗ cho context và phần chồng lặp; tokenizer có thể báo lỗi nếu không đủ.
    # Stride 128 là số token context dùng chung giữa hai cửa sổ kề nhau.
    # Output input_ids/attention_mask có shape [số feature, max_seq_length].
    return tokenizer(
        questions,
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",  # Cố định độ dài để gom các feature thành batch.
    )


def _first_answer_spans(examples: dict[str, Any]) -> list[tuple[int, int] | None]:
    """Lấy khoảng ký tự của đáp án đầu tiên. Ví dụ "Hà Nội" -> [0, 6)."""
    # Nhãn gốc tính theo số câu hỏi. Một câu hỏi có thể sinh nhiều feature
    # nhưng tất cả feature của nó đều tra cùng một nhãn qua sample_mapping.
    sample_count = len(examples["question"])
    labels = examples.get("is_impossible", [False] * sample_count)
    if len(labels) != sample_count:
        raise ValueError("Số nhãn is_impossible phải bằng số câu hỏi trong batch.")
    spans: list[tuple[int, int] | None] = []

    for sample_index, is_impossible in enumerate(labels):
        if is_impossible:
            spans.append(None)
            continue

        answers = examples["answers"][sample_index]
        # Đọc cả hai danh sách trước khi lấy [0]. Nếu dữ liệu chỉ có text
        # hoặc chỉ có offset thì không thể tạo nhãn token đáng tin cậy.
        if not isinstance(answers, dict):
            raise ValueError(f"Mẫu {sample_index}: answers phải là một dictionary.")
        texts = answers.get("text")
        starts = answers.get("answer_start")
        if not isinstance(texts, list) or not isinstance(starts, list):
            raise ValueError(f"Mẫu {sample_index}: thiếu answers.text hoặc answers.answer_start.")
        if len(texts) != len(starts):
            raise ValueError(f"Mẫu {sample_index}: text và answer_start phải cùng số phần tử.")

        # Với dữ liệu SQuAD không có cột is_impossible, hai list rỗng
        # biểu thị câu không có đáp án. Nhãn False ghi rõ thì phải có gold.
        if not texts:
            if "is_impossible" in examples:
                raise ValueError(f"Mẫu {sample_index}: được đánh dấu có đáp án nhưng thiếu đáp án.")
            spans.append(None)
            continue

        answer_text = texts[0]
        start_char = starts[0]
        if not isinstance(answer_text, str) or not answer_text.strip():
            raise ValueError(f"Mẫu {sample_index}: text của đáp án đầu tiên bị rỗng.")
        if type(start_char) is not int or start_char < 0:
            raise ValueError(f"Mẫu {sample_index}: answer_start phải là số nguyên không âm.")

        # Python cắt theo [start, end), nên end_char nằm ngay sau ký tự
        # cuối cùng. Vẫn giữ toàn bộ đáp án chuẩn cho bộ chấm EM/F1 riêng.
        end_char = start_char + len(answer_text)
        context = examples["context"][sample_index]
        if context[start_char:end_char] != answer_text:
            raise ValueError(f"Mẫu {sample_index}: answer_start không khớp văn bản đáp án.")
        spans.append((start_char, end_char))

    return spans


def prepare_train_features(
    examples: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int = MAX_SEQ_LENGTH,
    doc_stride: int = DOC_STRIDE,
) -> dict[str, Any]:
    """Tạo nhãn token start/end từ đáp án gốc, ví dụ "Hà Nội" ở đầu context."""
    # Bước 1–2: Kiểm tra đáp án rồi token hóa câu hỏi và context.
    answer_spans = _first_answer_spans(examples)
    tokenized_examples = _tokenize_questions_and_contexts(
        examples, tokenizer, max_seq_length, doc_stride
    )

    # Bước 3–4: Lấy thông tin ánh xạ ra khỏi feature đưa vào mô hình.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    start_positions = []
    end_positions = []

    # Bước 5–10: Mỗi cửa sổ nhận một cặp chỉ số, nên shape của mỗi list
    # nhãn là [số feature]. Mặc định trỏ về CLS nếu thiếu đáp án trọn vẹn.
    for feature_index, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][feature_index]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        start_position = cls_index
        end_position = cls_index
        answer_span = answer_spans[sample_mapping[feature_index]]

        if answer_span is not None:
            start_char, end_char = answer_span
            sequence_ids = tokenized_examples.sequence_ids(feature_index)
            context_positions = []
            for token_index, sequence_id in enumerate(sequence_ids):
                offset_start, offset_end = offsets[token_index]
                if sequence_id == 1 and offset_start < offset_end:
                    context_positions.append(token_index)

            if context_positions:
                first_context = context_positions[0]
                last_context = context_positions[-1]

                # Cửa sổ phải chứa cả hai biên ký tự của đáp án.
                if offsets[first_context][0] <= start_char and offsets[last_context][1] >= end_char:
                    for token_index in context_positions:
                        offset_start, offset_end = offsets[token_index]
                        if offset_start <= start_char < offset_end:
                            start_position = token_index
                        if offset_start < end_char <= offset_end:
                            end_position = token_index

                    # Một biên không rơi vào token (ví dụ bị tokenizer bỏ qua)
                    # thì cả feature vẫn mang nhãn CLS thay vì nhãn nửa vời.
                    if start_position == cls_index or end_position == cls_index or start_position > end_position:
                        start_position = cls_index
                        end_position = cls_index

        start_positions.append(start_position)
        end_positions.append(end_position)

    # Bước 11: Trả về đúng một nhãn bắt đầu và kết thúc cho mỗi feature.
    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples


def prepare_validation_features(
    examples: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int = MAX_SEQ_LENGTH,
    doc_stride: int = DOC_STRIDE,
) -> dict[str, Any]:
    """Giữ ID và offset context để trích lại đáp án, ví dụ "Hà Nội"."""
    # Bước 1–2: Tạo cửa sổ giống train và giữ ID của câu hỏi gốc.
    tokenized_examples = _tokenize_questions_and_contexts(
        examples, tokenizer, max_seq_length, doc_stride
    )
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    # Bước 3: Chỉ token context có offset để decoder trích văn bản.
    # offset_mapping có shape [số feature, max_seq_length].
    for feature_index, sample_index in enumerate(sample_mapping):
        sequence_ids = tokenized_examples.sequence_ids(feature_index)
        tokenized_examples["example_id"].append(examples["id"][sample_index])
        offsets = tokenized_examples["offset_mapping"][feature_index]
        for token_index, sequence_id in enumerate(sequence_ids):
            if sequence_id != 1:
                offsets[token_index] = None

    # Bước 4: Trả feature và metadata cho bộ giải mã.
    return tokenized_examples
