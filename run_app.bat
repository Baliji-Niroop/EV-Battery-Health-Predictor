@echo off
cd /d %~dp0

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python is not installed or not added to PATH.
    echo Please install Python and try again.
    pause
    exit /b
)

for /f "tokens=2 delims= " %%v in ('python -c "import sys; print(sys.version.split()[0])"') do set PYV=%%v
echo Detected Python %PYV%

echo Installing dependencies...
python -m pip install -r requirements.txt --quiet --disable-pip-version-check

python -c "import streamlit" >nul 2>nul
if %errorlevel% neq 0 (
    echo Failed to install Streamlit. Check your internet or run:
    echo python -m pip install -r requirements.txt
    pause
    exit /b
)

echo.
echo Starting EV Battery Health Dashboard...
echo If the browser didn't open automatically, visit: http://localhost:8501
echo.

python -m streamlit run dashboard/app.py

pause
