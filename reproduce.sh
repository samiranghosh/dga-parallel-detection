#!/bin/bash
# ============================================================================
# reproduce.sh — Reproduce all experiments and generate the HTML report
# DGA Parallel Detection | Group 09 | AMLCCZG516
# ============================================================================
# Usage:
#   bash reproduce.sh          # Full pipeline (preprocess + benchmark + report)
#   bash reproduce.sh --quick  # Skip preprocessing (uses existing data/)
# ============================================================================

set -e  # Exit on first error

echo "============================================"
echo " DGA Parallel Detection — Full Reproduction"
echo " Group 09 | AMLCCZG516 | BITS Pilani"
echo "============================================"
echo ""

# ── Check Python ──
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "ERROR: Python not found. Install Python 3.11.x first."
    exit 1
fi
PYTHON=$(command -v python3 || command -v python)
echo "Using: $PYTHON ($($PYTHON --version))"

# ── Check virtual environment ──
if [ -z "$VIRTUAL_ENV" ]; then
    echo "WARNING: No virtual environment detected. Consider activating one first."
    echo "         python3 -m venv venv && source venv/bin/activate"
    echo ""
    read -p "Continue without venv? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ── Install dependencies ──
echo ""
echo "[1/5] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install jupyter nbconvert -q
echo "      Done."

# ── Preprocess (skip with --quick) ──
if [ "$1" != "--quick" ]; then
    echo ""
    echo "[2/5] Running preprocessing pipeline..."
    echo "      (Data loading, TLD stripping, train/test split)"
    $PYTHON main.py --mode preprocess
    echo "      Done."
else
    echo ""
    echo "[2/5] Skipping preprocessing (--quick mode, using existing data/)"
fi

# ── Run full benchmark suite ──
echo ""
echo "[3/5] Running benchmark suite (E1–E8, 3 repetitions)..."
echo "      This may take 30–60 minutes depending on hardware."
$PYTHON main.py --mode benchmark --output-dir results/ --repetitions 3
echo "      Done. Results saved to results/metrics.json"

# ── Run tests ──
echo ""
echo "[4/5] Running test suite..."
$PYTHON -m pytest tests/ -v --tb=short
echo "      Done."

# ── Generate HTML report ──
echo ""
echo "[5/5] Executing notebook and exporting HTML report..."
jupyter nbconvert --execute P3_Experimental_Report.ipynb --to html \
    --ExecutePreprocessor.timeout=600
echo "      Done. Report: P3_Experimental_Report.html"

echo ""
echo "============================================"
echo " Reproduction complete."
echo ""
echo " Outputs:"
echo "   results/metrics.json          — Raw benchmark data"
echo "   results/plots/                — Benchmark visualizations"
echo "   P3_Experimental_Report.html   — Full report with outputs"
echo "============================================"
