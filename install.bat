@echo off
REM LibUIPC Auto-Install Script for Windows

echo 🚀 LibUIPC Auto-Installer for Windows
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python is required but not found
    echo Please install Python 3.10+ and add it to PATH
    pause
    exit /b 1
)

REM Run the auto installer
echo Starting automatic installation...
python auto_install.py %*

echo.
echo Installation completed!
echo.
echo To test the installation:
echo   python -c "import uipc; print('✅ LibUIPC imported successfully!')"
pause