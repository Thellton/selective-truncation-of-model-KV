@echo off
setlocal

REM Get the directory where this batch script is located
set SCRIPT_DIR=%~dp0

REM --- 1. Create Model_files folder and clone the model ---
echo Creating Model_files directory...
mkdir "%SCRIPT_DIR%Model_files"
if errorlevel 1 (
    echo Failed to create Model_files directory. It might already exist.
)

echo Changing to Model_files directory...
cd /d "%SCRIPT_DIR%Model_files"
if errorlevel 1 (
    echo Failed to change to Model_files directory.
    goto end_script
)

echo Cloning Qwen/Qwen3-0.6B model...
REM Check if Git is installed and in PATH
git --version >nul 2>&1
if errorlevel 1 (
    echo Git is not installed or not found in PATH. Please install Git and ensure it's in your PATH.
    echo You can download Git from https://git-scm.com/
    cd /d "%SCRIPT_DIR%"
    goto end_script
)

git clone https://huggingface.co/Qwen/Qwen3-0.6B
if errorlevel 1 (
    echo Failed to clone the model. Check your internet connection or if the folder already contains git data.
    cd /d "%SCRIPT_DIR%"
    goto end_script
)
echo Model cloned successfully.

REM --- 2. Go back, create VENV, and install requirements ---
echo Changing back to the script directory...
cd /d "%SCRIPT_DIR%"
if errorlevel 1 (
    echo Failed to change back to the script directory.
    goto end_script
)


echo Creating Python virtual environment (qwen_env)...
REM Check if python is installed and in PATH
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not found in PATH. Please install Python.
    echo You can download Python from https://www.python.org/
    goto end_script
)
python -m venv qwen_env
if errorlevel 1 (
    echo Failed to create virtual environment.
    goto end_script
)
echo Virtual environment created.

echo Activating virtual environment and installing packages...
call "%SCRIPT_DIR%qwen_env\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment.
    goto end_script
)

echo Installing packages from requirements.txt...
pip install -r "%SCRIPT_DIR%requirements.txt"
if errorlevel 1 (
    echo Failed to install packages. Check requirements.txt and your internet connection.
    goto end_script
)
echo Packages installed successfully.

echo Deactivating virtual environment (setup complete).
call deactivate

echo.
echo --- Setup Complete ---
echo You can now run the 'run_test.bat' script to execute testing_hypothesis.py.

:end_script
endlocal
pause