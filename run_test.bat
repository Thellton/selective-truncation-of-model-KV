@echo off
setlocal enabledelayedexpansion

REM Get the directory where this batch script is located
set "SCRIPT_DIR=%~dp0"

REM Change to the script's directory.
echo Changing to script directory: %SCRIPT_DIR%
cd /d "%SCRIPT_DIR%"
if errorlevel 1 (
    echo ERROR: Could not change to script directory: %SCRIPT_DIR%
    pause
    exit /b 1
)
echo Current directory is: %CD%

REM --- Activate VENV and run the Python script ---
echo Activating virtual environment (qwen_env)...

set "VENV_ACTIVATE_PATH=qwen_env\Scripts\activate.bat"
echo DEBUG: Checking for VENV activate script at: [%CD%\%VENV_ACTIVATE_PATH%]

if exist "%VENV_ACTIVATE_PATH%" (
    echo Found activate script. Calling it...
    call "%VENV_ACTIVATE_PATH%"
    if errorlevel 1 (
        echo Failed to activate virtual environment.
        goto end_script
    )
) else (
    echo Virtual environment activation script NOT found at %CD%\%VENV_ACTIVATE_PATH%.
    echo Please run setup_environment.bat first.
    goto end_script
)

echo Running testing_hypothesis.py...
set "PYTHON_SCRIPT_PATH=testing_hypothesis.py"
if exist "%PYTHON_SCRIPT_PATH%" (
    echo Found Python script. Running it...
    python "%PYTHON_SCRIPT_PATH%"
    if errorlevel 1 (
        echo testing_hypothesis.py encountered an error.
    )
) else (
    echo %PYTHON_SCRIPT_PATH% not found in %CD%.
    echo Please ensure the Python script is present.
    goto end_script
)

echo Deactivating virtual environment.
call deactivate

:end_script
echo.
echo --- Test Run Complete ---
endlocal
pause