#!/bin/bash
cd "$(dirname "$0")"

if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not added to PATH."
    echo "Please install Python 3 and try again."
    exit 1
fi

PYV=$(python3 -c "import sys; print(sys.version.split()[0])")
echo "Detected Python $PYV"

echo "Installing dependencies..."
python3 -m pip install -r requirements.txt --quiet --disable-pip-version-check

python3 -c "import streamlit" >/dev/null 2>&1 || {
    echo "Failed to install Streamlit. Check your internet or run:"
    echo "python3 -m pip install -r requirements.txt"
    exit 1
}

echo ""
echo "Starting EV Battery Health Dashboard..."
echo "If the browser didn't open automatically, visit: http://localhost:8501"
echo ""

python3 -m streamlit run dashboard/app.py
