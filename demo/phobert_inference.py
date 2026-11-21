"""Tiền xử lý suy luận theo notebook 06 của Quy."""

import re
from functools import lru_cache
from typing import Any

from underthesea import word_tokenize

MAX_LENGTH = 256
DOC_STRIDE = 64
MAX_QUESTION_TOKENS = 64


def segment_context(context: str) -> str:
    """Ví dụ phát triển có thể thành phát_triển sau tách từ."""
    words = word_tokenize(context)
    if isinstance(words, str):
        words = words.split(" ")

    segmented_words = []
    for token in words:
        # Notebook 06 bỏ token chỉ có khoảng trắng hoặc dấu gạch dưới.
        if any(not char.isspace() and char != "_" for char in token):
            segmented_words.append(token.replace(" ", "_"))
    return " ".join(segmented_words)


def prepare_phobert_windows(
    question: str, context: str, tokenizer: Any,
) -> tuple[dict[str, list[list[int]]], list[list[tuple[int, int] | None]], str]:
    """Tạo cửa sổ và offset trên context đã tách từ như notebook 06."""
    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    if cls_id != 0 or sep_id is None or pad_id is None:
        raise ValueError("Tokenizer của checkpoint cần có CLS ở vị trí 0 cùng SEP và PAD.")

    @lru_cache(maxsize=4096)
    def encode_word(word: str) -> tuple[int, ...]:
        # Một từ có thể tách thành nhiều mảnh BPE. Tất cả dùng cùng offset của từ.
        ids = tokenizer.encode(word, add_special_tokens=False)
        return tuple(ids or [tokenizer.unk_token_id])

    question_ids = []
    for word in word_tokenize(question.lstrip(), format="text").split():
        question_ids.extend(encode_word(word))
        if len(question_ids) >= MAX_QUESTION_TOKENS:
            break
    question_ids = question_ids[:MAX_QUESTION_TOKENS]

    segmented = segment_context(context)
    context_ids = []
    context_offsets = []
    for match in re.finditer(r"\S+", segmented):
        pieces = encode_word(match.group())
        context_ids.extend(pieces)
        context_offsets.extend([match.span()] * len(pieces))
    if not context_ids:
        raise ValueError("Không tìm thấy token trong đoạn văn sau khi tách từ.")

    prefix = [cls_id] + question_ids + [sep_id, sep_id]
    # 256 tính cả hai văn bản và 4 token đặc biệt. Stride là phần chồng lặp.
    window_length = MAX_LENGTH - len(prefix) - 1
    step = window_length - DOC_STRIDE
    all_ids = []
    all_masks = []
    all_offsets = []

    for window_start in range(0, len(context_ids), step):
        window_end = min(window_start + window_length, len(context_ids))
        ids = prefix + context_ids[window_start:window_end] + [sep_id]
        pad_count = MAX_LENGTH - len(ids)
        all_ids.append(ids + [pad_id] * pad_count)
        all_masks.append([1] * len(ids) + [0] * pad_count)
        # Chỉ token context có offset. Decoder loại câu hỏi, SEP và padding.
        all_offsets.append(
            [None] * len(prefix) + context_offsets[window_start:window_end] + [None] * (1 + pad_count)
        )
        if window_end == len(context_ids):
            break

    # input_ids/attention_mask/offsets: [số cửa sổ, 256].
    return {"input_ids": all_ids, "attention_mask": all_masks}, all_offsets, segmented
