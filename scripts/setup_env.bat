@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."

echo ========================================
echo   Environment setup
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 goto nopython
python --version

if exist "venv" goto havevenv
echo [1/3] Creating venv...
python -m venv venv
if errorlevel 1 goto venvfail
goto deps

:havevenv
echo [1/3] venv already exists

:deps
echo [2/3] Installing core dependencies...
call venv\Scripts\python.exe -m pip install --upgrade pip -q
call venv\Scripts\python.exe -m pip install -r requirements-core.txt
if errorlevel 1 goto depfail

echo [3/3] Self check...
call venv\Scripts\python.exe -c "import numpy,pandas,scipy,sklearn;print('core OK')"
if errorlevel 1 goto depfail

echo.
echo ========================================
echo   Done.
echo   Next: open the 01_xxx source folder, then run:
echo     ..\venv\Scripts\python.exe run_benchmark.py --model all-baselines --dataset mit --split test
echo   Models: pip install -r requirements-models.txt
echo ========================================
exit /b 0

:nopython
echo [ERROR] Python not found in PATH
exit /b 1

:venvfail
echo [ERROR] venv creation failed
exit /b 1

:depfail
echo [ERROR] dependency installation failed
exit /b 1
