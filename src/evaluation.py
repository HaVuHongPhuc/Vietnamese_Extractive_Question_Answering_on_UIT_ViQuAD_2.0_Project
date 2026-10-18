"""Giải mã câu trả lời và tính EM/F1 cho bài toán extractive QA."""

from __future__ import annotations

import math
import string
from collections import defaultdict
from typing import Any

import numpy as np

# Giới hạn chiều dài câu trả lời để tránh tạo span quá dài.
MAX_ANSWER_LENGTH = 30

# Chỉ xét 20 vị trí đầu/cuối có điểm cao để giảm số cặp token cần thử.
N_BEST_TOKENS = 20

# Ngưỡng 0 dùng để so sánh điểm câu trả lời với điểm no-answer (CLS).
NO_ANSWER_THRESHOLD = 0.0


def normalize_text(text: str) -> str:
    """Chuẩn hóa nhẹ tiếng Việt trước khi tính EM/F1.

    SQuAD tiếng Anh thường bỏ articles (a/an/the), nhưng tiếng Việt không có
    articles tương đương theo cách đó. Vì vậy ta chỉ viết thường, thay dấu câu
    bằng khoảng trắng và gom khoảng trắng thừa.
    """
    text = text.lower()
    text = "".join(" " if char in string.punctuation else char for char in text)
    return " ".join(text.split())


def exact_match(prediction: str, references: list[str]) -> float:
    """Trả 1 nếu dự đoán trùng hoàn toàn với ít nhất một đáp án chuẩn."""
    normalized_prediction = normalize_text(prediction)
    return float(any(normalized_prediction == normalize_text(ref) for ref in references))


def token_f1(prediction: str, reference: str) -> float:
    """F1 dựa trên các token cách nhau bởi khoảng trắng.

    Với tiếng Việt, phép tách này tính đơn vị theo âm tiết cách viết; đây là
    cách đơn giản, nhất quán để bắt đầu và dễ so sánh với baseline dự án.
    """
    predicted_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()

    # Hai chuỗi rỗng nghĩa là mô hình trả no-answer đúng.
    if not predicted_tokens and not reference_tokens:
        return 1.0
    if not predicted_tokens or not reference_tokens:
        return 0.0

    # Counter giữ số lần xuất hiện để các token lặp không bị tính quá mức.
    from collections import Counter

    common = Counter(predicted_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def score_prediction(prediction: str, references: list[str]) -> tuple[float, float]:
    """Lấy EM/F1 cao nhất trong danh sách đáp án chuẩn của một câu hỏi."""
    if not references:
        references = [""]
    em = exact_match(prediction, references)
    f1 = max(token_f1(prediction, reference) for reference in references)
    return em, f1


def decode_predictions(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    feature_ids: list[str],
    feature_offsets: list[list[Any]],
    examples_by_id: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Đổi dự đoán token thành đoạn văn bản lấy nguyên văn từ context.

    Ý tưởng: với mỗi feature (một cửa sổ context), thử ghép 20 vị trí bắt đầu
    tốt nhất với 20 vị trí kết thúc tốt nhất. Bỏ các cặp sai thứ tự, quá dài,
    hoặc nằm ngoài context. Cuối cùng giữ span có tổng logit cao nhất cho mỗi
    câu hỏi, kể cả khi câu hỏi tạo ra nhiều cửa sổ chồng lấn.
    """
    best_answer: dict[str, tuple[float, str]] = {}
    # CLS đại diện cho no-answer. Lấy score thấp nhất qua các cửa sổ vì một
    # cửa sổ khác có thể chứa bằng chứng tốt hơn cho câu hỏi có đáp án.
    best_null_score: dict[str, float] = defaultdict(lambda: math.inf)

    for feature_index, example_id in enumerate(feature_ids):
        example = examples_by_id[example_id]
        offsets = feature_offsets[feature_index]
        starts = start_logits[feature_index]
        ends = end_logits[feature_index]

        # Mã CLS của cả hai tokenizer đứng trong vị trí đầu của cặp đầu vào.
        null_score = float(starts[0] + ends[0])
        best_null_score[example_id] = min(best_null_score[example_id], null_score)

        # Chỉ khảo sát các vị trí có logit cao; giảm mạnh số cặp cần thử.
        top_starts = np.argsort(starts)[-N_BEST_TOKENS:]
        top_ends = np.argsort(ends)[-N_BEST_TOKENS:]
        context = example["context"]

        for start_index in top_starts:
            for end_index in top_ends:
                if end_index < start_index:
                    continue
                if end_index - start_index + 1 > MAX_ANSWER_LENGTH:
                    continue
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue

                char_start = int(offsets[start_index][0])
                char_end = int(offsets[end_index][1])
                if char_end <= char_start:
                    continue

                answer = context[char_start:char_end]
                score = float(starts[start_index] + ends[end_index])
                if score > best_answer.get(example_id, (-math.inf, ""))[0]:
                    best_answer[example_id] = (score, answer)

    predictions: dict[str, str] = {}
    for example_id in examples_by_id:
        answer_score, answer_text = best_answer.get(example_id, (-math.inf, ""))
        null_score = best_null_score.get(example_id, math.inf)

        # score difference dương nghĩa là CLS/no-answer có điểm cao hơn span.
        # Ngưỡng cố định 0 để so sánh công bằng giữa các seed và kiến trúc.
        score_difference = null_score - answer_score
        predictions[example_id] = (
            "" if score_difference > NO_ANSWER_THRESHOLD else answer_text
        )
    return predictions


def make_compute_metrics(
    feature_ids: list[str],
    feature_offsets: list[list[Any]],
    examples_by_id: dict[str, dict[str, Any]],
):
    """Tạo hàm metric cho Trainer; metadata offsets được đóng gói trong hàm."""

    def compute_metrics(evaluation_prediction: Any) -> dict[str, float]:
        # Với mô hình hỏi đáp, predictions là cặp (start_logits, end_logits).
        predictions = evaluation_prediction.predictions
        start_logits, end_logits = predictions[0], predictions[1]

        decoded = decode_predictions(
            start_logits,
            end_logits,
            feature_ids,
            feature_offsets,
            examples_by_id,
        )

        em_scores: list[float] = []
        f1_scores: list[float] = []
        for example_id, example in examples_by_id.items():
            references = example["answers"].get("text", [])
            em, f1 = score_prediction(decoded[example_id], references)
            em_scores.append(em)
            f1_scores.append(f1)

        # Nhân 100 để kết quả dễ đọc theo phần trăm như các báo cáo QA.
        return {
            "exact_match": 100 * float(np.mean(em_scores)),
            "f1": 100 * float(np.mean(f1_scores)),
        }

    return compute_metrics
