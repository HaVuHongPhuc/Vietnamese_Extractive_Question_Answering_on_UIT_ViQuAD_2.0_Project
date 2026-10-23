# Môi trường local cho RMIT Hackathon — Challenge 1

Môi trường này dành cho encoder baseline DistilBERT multilingual/XLM-R và K-fold.
Không sửa `requirements.txt` hoặc `.venv` của dự án UIT-ViQuAD.
Chưa cài LoRA/QLoRA; chưa tải dữ liệu hoặc pretrained model của cuộc thi.
Venv nằm tại `C:\Users\havuh\venvs\rmit-c1`, ngoài OneDrive để tránh lỗi đường dẫn dài.

## Cài đặt trên Windows

Yêu cầu: Python 3.12, NVIDIA driver tương thích CUDA 13.0 và Internet.
`scripts/setup_c1.ps1` tự xác định thư mục gốc từ vị trí của script.
Chạy script bằng đường dẫn tuyệt đối tới file đó:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\havuh\OneDrive\Desktop\HocTap\Vietnamese_Extractive_Question_Answering_on_UIT_ViQuAD_2.0_Project\scripts\setup_c1.ps1"
```

Script sẽ tạo `C:\Users\havuh\venvs\rmit-c1`, cài PyTorch CUDA và các dependencies, chạy `pip check`,
đăng ký kernel trong chính venv và ở cấp user, kiểm tra import/GPU và xuất
`requirements-c1-lock.txt`. Script không tự train và không tải pretrained model.
Execution policy chỉ được áp dụng cho tiến trình script này.

Kiểm tra lại hoặc mở JupyterLab bằng các lệnh PowerShell sau:

```powershell
$Project = "C:\Users\havuh\OneDrive\Desktop\HocTap\Vietnamese_Extractive_Question_Answering_on_UIT_ViQuAD_2.0_Project"
& "C:\Users\havuh\venvs\rmit-c1\Scripts\python.exe" -X utf8 "$Project\scripts\check_c1_environment.py" --require-cuda
& "C:\Users\havuh\venvs\rmit-c1\Scripts\python.exe" -m jupyterlab --notebook-dir "$Project"
```

Activate (không bắt buộc nếu gọi trực tiếp Python như trên):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& "C:\Users\havuh\venvs\rmit-c1\Scripts\Activate.ps1"
```

## Dùng trong VS Code

1. Mở notebook, chọn **Select Kernel → Python Environments**.
2. Chọn hoặc nhập interpreter `C:\Users\havuh\venvs\rmit-c1\Scripts\python.exe`.
3. Kernel trong JupyterLab có tên **Python (RMIT C1)**.
4. Kiểm tra cell đầu: `import sys; print(sys.executable)`.

Nếu danh sách Python Environments chưa thấy venv, lưu notebook rồi chạy
`Developer: Reload Window` trong Command Palette. Trong notebook, chọn
**Select Kernel → Select Another Kernel → Jupyter Kernels → Python (RMIT C1)**.
Kernel cũng được đăng ký ở cấp user tại
`C:\Users\havuh\AppData\Roaming\jupyter\kernels\rmit-c1` để ứng dụng ngoài venv
có thể tìm thấy. Không cần cài lại dependencies hoặc tạo môi trường mới.

Có thể gọi Python trong venv bằng đường dẫn tuyệt đối mà không cần activate.
Nếu muốn activate trong PowerShell, dùng đường dẫn tuyệt đối tới
`C:\Users\havuh\venvs\rmit-c1\Scripts\Activate.ps1`. Khi bị chặn script, chỉ thay đổi policy cho phiên
hiện tại bằng `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

## Dùng dữ liệu local

Notebook đã gửi trước đó dùng `/kaggle/input` và `/kaggle/working`, chỉ dành cho
Kaggle. Trên Windows cần đổi chúng thành đường dẫn local tới dữ liệu/output và
đổi `MODEL_DIR` thành thư mục chứa backbone được ban tổ chức duyệt.
Chỉ cài môi trường không làm notebook tự có dữ liệu hoặc model.

## Tái lập dependencies

- `requirements-c1.txt`: dependencies chính; Transformers được cố định ở 4.57.6.
- `requirements-c1-lock.txt`: toàn bộ phiên bản thực tế sau khi cài thành công.
- Để tái lập bản GPU: cài `torch==2.14.1+cu130` từ index CUDA 13.0 chính thức
  như trong script trước, rồi cài file lock bằng Python của venv.
- Không đưa venv lên Git hoặc chuyển nguyên venv sang máy khác; tạo lại nó.

Môi trường local này không thay đổi thư viện/kernel trong notebook Kaggle.