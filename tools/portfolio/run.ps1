# Разбор портфолио Максима.
# Ставит всё нужное в _служебное\venv (рядом с этим файлом, без кэша pip в профиле) и запускает portfolio_sort.py.
#
# Положить папку со скриптами в «фото мебель\_служебное» и запустить:
#   powershell -ExecutionPolicy Bypass -File ".\run.ps1"            # всё сразу (analyze + apply)
#   powershell -ExecutionPolicy Bypass -File ".\run.ps1" analyze    # только анализ
#   powershell -ExecutionPolicy Bypass -File ".\run.ps1" apply      # раскладка по разметке
#   powershell -ExecutionPolicy Bypass -File ".\run.ps1" undo       # вернуть дубли на место

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$venv = Join-Path $here "venv"
$py = Join-Path $venv "Scripts\python.exe"

function Check($what) {
    if ($LASTEXITCODE -ne 0) { Write-Host "Ошибка: $what (код $LASTEXITCODE)" -ForegroundColor Red; exit 1 }
}

if (-not (Test-Path $py)) {
    Write-Host "Первый запуск: создаю окружение и ставлю библиотеки (~1 ГБ, 5-15 минут)..."
    if (Get-Command py -ErrorAction SilentlyContinue) { & py -3 -m venv $venv }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { & python -m venv $venv }
    else { Write-Host "Не найден Python. Установите: winget install Python.Python.3.12" -ForegroundColor Red; exit 1 }
    Check "создание venv"
    & $py -m pip install --no-cache-dir --upgrade pip; Check "pip"
    & $py -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; Check "torch"
    & $py -m pip install --no-cache-dir -r (Join-Path $here "requirements.txt"); Check "библиотеки"
}

$env:PYTHONUTF8 = "1"
$cmd = if ($args.Count -gt 0) { $args } else { @("run") }
& $py (Join-Path $here "portfolio_sort.py") @cmd
Check "portfolio_sort.py"
