# Chạy từ bất kỳ thư mục nào; mọi đường dẫn được tính từ vị trí script.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
# Đường dẫn ngắn, nằm ngoài OneDrive để tránh WinError 206 khi cài PyTorch.
$VenvDirectory = Join-Path $env:USERPROFILE 'venvs\rmit-c1'
$Python = Join-Path $VenvDirectory 'Scripts\python.exe'
$Requirements = Join-Path $ProjectRoot 'requirements-c1.txt'
$CheckScript = Join-Path $PSScriptRoot 'check_c1_environment.py'
$LockFile = Join-Path $ProjectRoot 'requirements-c1-lock.txt'

function Invoke-C1Python {
    param([string[]]$Arguments)

    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($Arguments -join ' ')"
    }
}

# Không sửa hoặc xóa môi trường .venv đang phục vụ dự án QA.
if (-not (Test-Path -LiteralPath $Python)) {
    if (Test-Path -LiteralPath $VenvDirectory) {
        throw "An incomplete environment already exists at $VenvDirectory. Inspect it first."
    }
    & py -3.12 -m venv $VenvDirectory
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not create the environment. Install Python 3.12 first.'
    }
}

Invoke-C1Python -Arguments @(
    '-c', 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
)
Invoke-C1Python -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip')

# Cài bản CUDA từ kho PyTorch chính thức, không lấy torch CPU mặc định.
Invoke-C1Python -Arguments @(
    '-m', 'pip', 'install', 'torch==2.14.1+cu130',
    '--index-url', 'https://download.pytorch.org/whl/cu130'
)
Invoke-C1Python -Arguments @('-m', 'pip', 'install', '-r', $Requirements)
Invoke-C1Python -Arguments @('-m', 'pip', 'check')

# Dùng prefix bên trong venv, không thay thế kernel của môi trường khác.
Invoke-C1Python -Arguments @(
    '-m', 'ipykernel', 'install', '--prefix', $VenvDirectory,
    '--name', 'rmit-c1', '--display-name', 'Python (RMIT C1)'
)

# Đăng ký thêm ở cấp user để VS Code/Jupyter ngoài venv có thể tìm thấy.
# Tên riêng rmit-c1 không thay đổi kernel python3 hoặc kernel QA.
Invoke-C1Python -Arguments @(
    '-m', 'ipykernel', 'install', '--user',
    '--name', 'rmit-c1', '--display-name', 'Python (RMIT C1)'
)

# Kiểm tra import và một lần inference rất nhỏ; không train hoặc tải model.
Invoke-C1Python -Arguments @('-X', 'utf8', $CheckScript, '--require-cuda')

# Lưu toàn bộ phiên bản thực tế để có thể tái lập môi trường sau này.
$FrozenPackages = & $Python -m pip freeze
if ($LASTEXITCODE -ne 0) {
    throw 'Could not export installed package versions.'
}
$FrozenPackages | Set-Content -LiteralPath $LockFile -Encoding UTF8

Write-Output "SUCCESS: environment ready at $VenvDirectory"
Write-Output "Lock file: $LockFile"
Write-Output 'No competition training has been started.'