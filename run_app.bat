@echo off
cd /d %~dp0

set PYTHON_EXEC=python
if exist venv\Scripts\python.exe (
    set PYTHON_EXEC=venv\Scripts\python.exe
    echo Using python from virtual environment...
)

where %PYTHON_EXEC% >nul 2>nul
if %errorlevel% neq 0 (
    if not exist venv\Scripts\python.exe (
        echo Python is not installed or not found.
        echo Please install Python and try again.
        pause
        exit /b
    )
)

for /f "delims=" %%v in ('%PYTHON_EXEC% -c "import sys; print(sys.version.split()[0])"') do set PYV=%%v
echo Detected Python %PYV%

echo Installing dependencies...
%PYTHON_EXEC% -m pip install -r requirements.txt --disable-pip-version-check

%PYTHON_EXEC% -c "import streamlit" >nul 2>nul
if %errorlevel% neq 0 (
    echo Failed to install Streamlit. Check your internet or run:
    echo %PYTHON_EXEC% -m pip install -r requirements.txt
    pause
    exit /b
)

echo.
echo Starting EV Battery Health Dashboard...
echo If the browser didn't open automatically, visit: http://localhost:8501
echo.

%PYTHON_EXEC% -m streamlit run dashboard/app.py

pause
