"""
Module: data_preprocessing.py
Chức năng: Tiền xử lý dữ liệu và ánh xạ vị trí đáp án (Offset Mapping)
cho bài toán Extractive QA (Hỗ trợ câu hỏi unanswerable).
"""
from typing import Dict, Any
def prepare_train_features(
    examples: Dict[str, Any],
    tokenizer: Any,
    max_seq_length: int = 384,
    doc_stride: int = 128
) -> Dict[str, Any]:
    """
    Tiền xử lý tập Train: Tokenize, trượt cửa sổ và ánh xạ span từ ký tự sang token.
    """
    # Làm sạch khoảng trắng đầu/cuối của câu hỏi
    examples["question"] = [q.lstrip() for q in examples["question"]]

    # Tokenize đồng thời Question và Context
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
       truncation="only_second",          # Chỉ cắt bớt context, giữ nguyên 100% câu hỏi
        max_length=max_seq_length,         # Độ dài tối đa (ví dụ: 384 tokens)
        stride=doc_stride,                 # Khoảng gối đầu giữa 2 lát cắt (ví dụ: 128 tokens)
        return_overflowing_tokens=True,    # Context dài quá 384 thì tự động cắt thành nhiều lát
        return_offsets_mapping=True,       # Trả về tọa độ ký tự tương ứng của từng token
        padding="max_length",              # Chèn thêm token padding cho đủ 384 token để GPU gom batch
    )

    """
    truncation="only_second": Trong một cặp (câu hỏi, ngữ cảnh), câu hỏi là yếu tố
    bắt buộc phải giữ nguyên vẹn để mô hình hiểu đúng ý. Nếu tổng độ dài vượt quá 384,
    hệ thống chỉ cắt ngắn ngữ cảnh (context), không bao giờ cắt bớt câu hỏi.

    - stride=doc_stride kết hợp return_overflowing_tokens=True: Nếu đoạn ngữ cảnh dài tới 600 từ,
    thư viện sẽ tự động xẻ đoạn văn thành nhiều lát cắt nhỏ (gọi là các features):

       + Lát 1: Từ token 0 đến 384.

       + Lát 2: Từ token 256 đến 640 (gối đầu 128 token lên lát 1 để tránh việc câu trả lời bị
       chặt đứt đôi ở ngay điểm cắt).

    - return_offsets_mapping=True: Yêu cầu Tokenizer lưu lại tọa độ ký tự
    (start_char, end_char) của từng token trong chuỗi văn bản gốc
    để phục vụ việc dò tìm ở các bước sau.
    """


    # Tách thông tin trung gian để xử lý
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    """
    - pop("overflow_to_sample_mapping"): Khi một câu gốc bị xẻ thành 2 hoặc 3 lát cắt,
    danh sách sample_mapping sẽ ghi lại xem lát cắt hiện tại được sinh ra từ
    câu hỏi gốc thứ mấy trong tập dữ liệu. Dùng lệnh pop() để lấy dữ liệu ra xử lý và
    xóa khỏi dictionary, tránh làm PyTorch báo lỗi vì không nhận diện được
    trường này lúc nạp vào mô hình.

    - pop("offset_mapping"): Lấy bảng tọa độ ký tự ra để dò tìm vị trí token.
    
    - Hai mảng rỗng start_positions và end_positions: Đây là nhãn đầu ra
    mà mô hình Transformer cần học (dự đoán token bắt đầu và token kết thúc).
    """


    # Phân loại Token và Xác định vị trí CLS
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Xác định sequence: None (special token), 0 (question), 1 (context)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        is_impossible = examples.get("is_impossible", [False] * len(sample_mapping))[sample_index]

        # Trường hợp 1: Câu hỏi không có đáp án
        if is_impossible or len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
            continue

        # Trường hợp 2: Câu hỏi có đáp án
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])

        # Tìm phạm vi token thuộc phần Context
        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        # Kiểm tra xem đáp án có nằm trọn trong đoạn trượt này không
        if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
            # Nằm ngoài cửa sổ trượt -> đánh dấu nhãn unanswerable cho feature này
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Tìm token bắt đầu
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            start_position = token_start_index - 1

            # Tìm token kết thúc
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            end_position = token_end_index + 1

            tokenized_examples["start_positions"].append(start_position)
            tokenized_examples["end_positions"].append(end_position)

    return tokenized_examples


def prepare_validation_features(
    examples: Dict[str, Any],
    tokenizer: Any,
    max_seq_length: int = 384,
    doc_stride: int = 128
) -> Dict[str, Any]:

    # Tiền xử lý tập Validation/Test: Giữ lại offset_mapping và example_id
    # để phục vụ việc map logits dự đoán ngược lại văn bản gốc lúc đánh giá.

    examples["question"] = [q.lstrip() for q in examples["question"]]

    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_seq_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Đặt offset của các token không thuộc context về None để lúc decode bỏ qua
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

"""
1. Vấn đề gốc rễ: Sự lệch pha giữa Ký tự và TokenTrong file dữ liệu gốc (train.json),
câu trả lời được lưu theo vị trí ký tự (Character index):Đoạn văn: "Phạm Văn Đồng sinh năm 1906 tại
Quảng Ngãi..."Câu hỏi: "Phạm Văn Đồng sinh năm nào?"Đáp án: "1906", answer_start = 14 (nghĩa là
đếm từ chữ P đầu tiên đến chữ số 1 của năm 1906 vừa đúng 14 ký tự).Tuy nhiên,
mô hình AI (Transformer) không đọc từng ký tự. Khi văn bản đi vào mô hình, bộ phân tách
từ (Tokenizer) sẽ băm đoạn văn thành các Token (mảnh từ):
PlaintextChỉ số Token:   [0]     [1]     [2]     [3]     [4]    [5]      [6]     [7]     ...
         Các Token:     [<s>]  [Phạm]  [Văn]   [Đồng]  [sinh]  [năm]   [1906]   [tại]    ...
Lúc này, con số 14 ký tự hoàn toàn vô nghĩa với mô hình. Mô hình
cần biết:"Đáp án bắt đầu từ Token số 6 và kết thúc ở Token số 6."Hàm data_preprocessing.py
chính là "người phiên dịch": Nó đọc chỉ số ký tự (14), tra xem ký tự đó rơi trúng
Token số mấy trong danh sách Token, rồi gắn nhãn start_positions = 6 và end_positions = 6 để đưa
vào huấn luyện.

2. Hai nhiệm vụ chính của file data_preprocessing.pyÁnh xạ vị trí (offset_mapping):
Tokenizer lưu lại bảng tọa độ ký tự của từng token (ví dụ: Token [1906] bao phủ từ ký
tự thứ 14 đến 18).Code sẽ dò tọa độ này để tìm ra index của token bắt đầu và kết thúc.Cắt lát văn bản
dài (Sliding Window):Mô hình chỉ nhận tối đa 384 token. Nhưng một bài viết Wikipedia có thể
dài tới 1.000 token.Ta phải cắt bài viết thành nhiều đoạn nhỏ trượt gối đầu lên
nhau:Đoạn 1: từ token 0 $\rightarrow$ 384.Đoạn 2: từ token 256 $\rightarrow$ 640 (gối đầu 128token
để không làm đứt câu).Nếu đoạn cắt nào chứa câu trả lời $\rightarrow$ gán token index của câu
trả lời đó.Nếu đoạn cắt nào không chứa câu trả lời (hoặc câu hỏi unanswerable) $\rightarrow$ gán
nhãn (0, 0) (trỏ về token đặc biệt [CLS] ở vị trí đầu tiên, quy ước là "không có câu trả lời").
"""