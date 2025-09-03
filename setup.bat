@echo off
echo 🧠 DataMind Setup Script
echo ========================

echo 📦 Installing dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ❌ Error installing dependencies!
    echo Please check your Python installation and try again.
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully!
echo.
echo 🚀 Starting DataMind application...
echo.
echo 📝 The application will open in your default browser at http://localhost:8501
echo.
echo 💡 To stop the application, press Ctrl+C in this terminal
echo.

streamlit run app.py

pause