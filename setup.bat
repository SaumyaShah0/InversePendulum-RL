@echo off
REM ================================
REM  InversePendulum-RL Setup Script
REM  Uses Python 3.10 + virtual env
REM ================================

echo.
echo [1/3] Creating virtual environment with Python 3.10...

REM Use the Python launcher to force 3.10
py -3.10 -m venv venv

if errorlevel 1 (
    echo Failed to create venv. Make sure Python 3.10 is installed and "py -3.10" works.
    pause
    exit /b 1
)

echo [2/3] Activating virtual environment...
call venv\Scripts\activate

echo [3/3] Installing required Python packages...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete!
echo To start working, run:
echo.
echo   venv\Scripts\activate
echo   python src\train_ppo.py
echo.
pause