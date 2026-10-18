"""Các hàm tiện ích chung có thể được nhiều module trong project dùng lại."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(file_path: Path, data: Any) -> None:
    """Ghi dữ liệu Python thành file JSON dễ đọc.

    JSON là định dạng văn bản để lưu dữ liệu có cấu trúc như dict và list.
    Ví dụ: metrics Python -> metrics.json -> có thể mở bằng VS Code hoặc đọc lại
    bằng json.load().

    file_path: vị trí và tên file cần ghi, ví dụ results/seed-42/metrics.json.
    data: dữ liệu Python cần lưu, ví dụ dict điểm số hoặc list dự đoán.
    """
    # Tạo các thư mục cha nếu chưa có; exist_ok=True tránh lỗi nếu chúng đã tồn tại.
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # ensure_ascii=False giữ nguyên chữ tiếng Việt thay vì đổi thành dạng \uXXXX.
    # indent=2 xuống dòng và thụt lề để con người dễ đọc, encoding=UTF-8 lưu đúng dấu.
    json_text = json.dumps(data, ensure_ascii=False, indent=2)

    # Ghi chuỗi JSON ra đĩa; nếu file đã tồn tại thì nội dung cũ được thay bằng nội dung mới.
    file_path.write_text(json_text, encoding="utf-8")
