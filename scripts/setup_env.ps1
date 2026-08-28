# skin_detector 개발 환경 구성 (Windows).
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#
# 이미 구성돼 있으면 건너뛴다 (멱등).

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "=== skin_detector 환경 구성 ===" -ForegroundColor Cyan

# ── 1. Python 3.12 ────────────────────────────────────────────────────
#
# 주의: `python.exe` 가 0바이트 Microsoft Store 스텁인 경우가 흔하다.
# 스텁은 버전을 출력하지 않고 pip install 이 조용히 실패한다.
# 아래는 실제 설치본 경로를 직접 찾으므로 스텁에 걸리지 않는다.

$candidates = @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:ProgramFiles\Python312\python.exe",
    "C:\Python312\python.exe"
)
$py = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $py) {
    Write-Host "Python 3.12 를 찾지 못했다. winget 으로 설치한다..." -ForegroundColor Yellow
    winget install --id Python.Python.3.12 --scope user --architecture x64 --silent `
        --accept-package-agreements --accept-source-agreements
    $py = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $py) {
    Write-Host "Python 설치 실패. python.org 에서 3.12 x64 를 직접 설치한다." -ForegroundColor Red
    exit 1
}
Write-Host "Python: $py" -ForegroundColor Green
& $py --version

# ── 2. venv ───────────────────────────────────────────────────────────
#
# venv 를 쓰면 Store 별칭 문제를 **완전히 우회한다** — venv 안의 python.exe 는
# 실제 인터프리터의 복사본이라 별칭이 끼어들 여지가 없다.
# 다만 터미널에서 전역 `python` 을 쓰고 싶다면 별칭을 꺼야 한다:
#   설정 -> 앱 -> 고급 앱 설정 -> 앱 실행 별칭 -> python.exe / python3.exe 끄기

if (-not (Test-Path ".venv")) {
    Write-Host "venv 생성..." -ForegroundColor Cyan
    & $py -m venv .venv
}
$venvPy = Join-Path $root ".venv\Scripts\python.exe"

# ── 3. 의존성 ─────────────────────────────────────────────────────────
Write-Host "의존성 설치..." -ForegroundColor Cyan
& $venvPy -m pip install --quiet --upgrade pip
& $venvPy -m pip install --quiet `
    numpy opencv-python "mediapipe<0.11" pillow pillow-heif `
    pytest pytest-cov httpx2 `
    fastapi "uvicorn[standard]" python-multipart `
    matplotlib

# ── 4. 랜드마커 모델 ──────────────────────────────────────────────────
#
# 모델 번들은 mediapipe 휠에 들어 있지 않다. 별도로 받아야 한다.
$model = "models\face_landmarker.task"
if (-not (Test-Path $model)) {
    Write-Host "FaceLandmarker 모델 다운로드 (~3.7MB)..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force models | Out-Null
    $url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    Invoke-WebRequest -Uri $url -OutFile $model
}
Write-Host "모델: $model ($([math]::Round((Get-Item $model).Length/1MB,1)) MB)" -ForegroundColor Green

# ── 5. 검증 ───────────────────────────────────────────────────────────
Write-Host "`n=== 자체 검증 ===" -ForegroundColor Cyan
$env:PYTHONIOENCODING = "utf-8"
& $venvPy -m pytest test\ -q

Write-Host "`n다음 단계:" -ForegroundColor Cyan
Write-Host "  1) 수동 카메라 앱 설정 확인 (docs\VALIDATION.md 의 촬영 규약)"
Write-Host "  2) 사진을 data\incoming\ 에 넣고:"
Write-Host "       .venv\Scripts\python.exe scripts\analyze.py --ingest data\incoming"
Write-Host "  3) 웹 UI:"
Write-Host "       .venv\Scripts\python.exe -m uvicorn web.server:app --host 127.0.0.1 --port 8000"
