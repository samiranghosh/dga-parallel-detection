@echo off
REM ============================================================================
REM reproduce.bat — Reproduce all experiments and generate the HTML report
REM DGA Parallel Detection | Group 09 | AMLCCZG516
REM ============================================================================
REM Usage:
REM   reproduce.bat          Full pipeline (preprocess + benchmark + report)
REM   reproduce.bat --quick  Skip preprocessing (uses existing data\)
REM ============================================================================

echo ============================================
echo  DGA Parallel Detection — Full Reproduction
echo  Group 09 ^| AMLCCZG516 ^| BITS Pilani
echo ============================================
echo.

REM ── Check Python ──
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found. Install Python 3.11.x first.
    exit /b 1
)
echo Using: & python --version

REM ── Install dependencies ──
echo.
echo [1/5] Installing dependencies...
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install jupyter nbconvert -q
echo       Done.

REM ── Preprocess (skip with --quick) ──
if "%1"=="--quick" (
    echo.
    echo [2/5] Skipping preprocessing [--quick mode, using existing data\]
) else (
    echo.
    echo [2/5] Running preprocessing pipeline...
    python main.py --mode preprocess
    if %ERRORLEVEL% neq 0 ( echo FAILED & exit /b 1 )
    echo       Done.
)

REM ── Run full benchmark suite ──
echo.
echo [3/5] Running benchmark suite (E1-E8, 3 repetitions)...
echo       This may take 30-60 minutes depending on hardware.
python main.py --mode benchmark --output-dir results/ --repetitions 3
if %ERRORLEVEL% neq 0 ( echo FAILED & exit /b 1 )
echo       Done. Results saved to results\metrics.json

REM ── Run tests ──
echo.
echo [4/5] Running test suite...
python -m pytest tests/ -v --tb=short
if %ERRORLEVEL% neq 0 ( echo WARNING: Some tests failed. & echo. )
echo       Done.

REM ── Generate HTML report ──
echo.
echo [5/5] Executing notebook and exporting HTML report...
jupyter nbconvert --execute P3_Experimental_Report.ipynb --to html --ExecutePreprocessor.timeout=600
if %ERRORLEVEL% neq 0 ( echo FAILED & exit /b 1 )
echo       Done. Report: P3_Experimental_Report.html

echo.
echo ============================================
echo  Reproduction complete.
echo.
echo  Outputs:
echo    results\metrics.json          — Raw benchmark data
echo    results\plots\                — Benchmark visualizations
echo    P3_Experimental_Report.html   — Full report with outputs
echo ============================================
