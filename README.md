# DGA Parallel Detection

**Parallel Feature Extraction for Domain Generation Algorithm (DGA) Based Malware Detection**

AMLCCZG516 — ML System Optimization — Group 09, BITS Pilani

---

## Overview

This project implements a data-parallel pipeline for detecting malware that uses Domain Generation Algorithms (DGA) to communicate with Command & Control servers. Based on the framework proposed by [Li et al. (2019)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8631171), we optimize the feature extraction and classification stages using Python multiprocessing on a standard multi-core CPU.

**Key Contributions:**
- **Layer 1 — Data Parallelism:** Parallel linguistic feature extraction across K CPU cores using `multiprocessing.Pool`
- **Layer 2 — Model Parallelism:** Random Forest training with `n_jobs=-1` via scikit-learn's joblib backend
- **Overlapping chunk strategy** to handle Levenshtein distance sequential dependencies at chunk boundaries
- **Comprehensive benchmarking** with strong/weak scaling analysis, Amdahl's Law validation, and IPC overhead measurement
- **5-feature production configuration** — feature ablation revealed that dropping Levenshtein improves both accuracy (93.18%) and throughput
- **Statistical validation** — 5-fold cross-validation with paired t-tests for model and feature-set comparisons
- **Real-time inference API** — FastAPI endpoint for single-domain and batch classification (~2–4ms per domain)

## Target Platform

| Component | Specification |
|-----------|--------------|
| Machine | HP OMEN 16 Gaming Laptop (16-xd0xxx) |
| CPU | AMD Ryzen 7 7840HS — 8 cores / 16 threads @ 3.8 GHz |
| RAM | 32 GB DDR5 |
| OS | Windows 11 Home (Build 26200) |
| Python | 3.11.x (CPython) |

## Project Structure

```
dga-parallel-detection/
├── data/
│   ├── dga_training_data.json.gz # ExtraHop dataset (via git lfs clone)
│   ├── english_dictionary.txt   # NLTK English word corpus (auto-generated)
│   ├── ngram_table.pkl          # Trigram frequency table (auto-generated)
│   ├── train.csv                # 80% stratified split (auto-generated)
│   └── test.csv                 # 20% stratified split (auto-generated)
├── src/
│   ├── __init__.py
│   ├── features.py              # 6 linguistic feature extractors
│   ├── parallel_engine.py       # Pool-based parallel orchestration
│   ├── chunker.py               # Overlapping chunk creation + adaptive sizing
│   ├── shared_resources.py      # Shared memory setup for dict/n-grams
│   ├── classifier.py            # RF training (n_jobs), DT comparison, CV
│   ├── benchmark.py             # Timing, profiling, and metrics collection
│   ├── fault_handler.py         # Retry logic, validation, timeout wrapper
│   └── preprocess.py            # Data loading, cleaning, TLD stripping
├── tests/
│   ├── __init__.py
│   ├── test_features.py         # Unit tests for each feature function
│   ├── test_parallel.py         # Correctness: parallel == sequential output
│   └── test_boundary.py         # Levenshtein overlap verification
├── results/
│   ├── plots/                   # Generated benchmark visualizations
│   └── metrics.json             # Raw benchmark data
├── P3_Experimental_Report.ipynb # Final experimental report (81 cells)
├── P3_Experimental_Report.html # Executed notebook with all outputs and plots
├── api.py                       # FastAPI inference endpoint
├── main.py                      # Entry point (sequential / parallel / benchmark)
├── requirements.txt             # Pinned Python dependencies
├── .gitattributes               # Git LFS tracking rules
├── .gitignore
└── README.md                    # This file
```

## Setup Instructions

### 1. Prerequisites

Ensure you have **Python 3.11.x** installed. Verify with:

```bash
python --version
```

On Windows, you may need to use `python` instead of `python3`. If you have multiple Python versions, use `py -3.11` to target the correct one.

### 2. Clone the Repository

```bash
git clone https://github.com/samiranghosh/dga-parallel-detection.git
cd dga-parallel-detection
```

### 3. Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note on `python-Levenshtein`:** This package requires a C compiler. On Windows, install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) if the install fails. Alternatively, the project falls back to a pure-Python DP implementation if the C extension is unavailable.

### 5. Download NLTK Data

Run the following once to download the English word corpus:

```bash
python -c "import nltk; nltk.download('words')"
```

### 6. Prepare Datasets

#### Option A — ExtraHop DGA Training Dataset (Recommended)

This is the simplest option. A single dataset with both DGA and benign domains, pre-labeled.

- URL: https://github.com/ExtraHop/DGA-Detection-Training-Dataset
- License: MIT (free for all use)
- Size: ~16.2 million domains (JSON, gzipped)
- Format: `{"domain": "example", "threat": "benign"}` or `{"domain": "xyzabc", "threat": "dga"}`
- TLDs already stripped — no extra preprocessing needed

**Important:** The dataset file uses **Git LFS** (Large File Storage), so a direct
browser download or `wget` on the raw URL will NOT work. You must clone with LFS:

```bash
# Install Git LFS (one-time setup)
git lfs install

# Clone the repo (this downloads the actual data file via LFS)
git clone https://github.com/ExtraHop/DGA-Detection-Training-Dataset.git

# Copy the data file into your project
cp DGA-Detection-Training-Dataset/dga-training-data-encoded.json.gz data/

# Clean up the cloned repo (optional)
rm -rf DGA-Detection-Training-Dataset
```

On Windows, Git LFS is included with [Git for Windows](https://gitforwindows.org/).
Verify with `git lfs version`.

The preprocessing script will sample 500K DGA + 500K benign from this file.

#### Option B — Separate DGA + Benign Sources

If you prefer per-family DGA labels for clustering experiments:

**DGA Domains (25 families):**
- URL: https://github.com/chrmor/DGA_domains_dataset
- Size: 337,500 DGA domains across 25 malware families (CryptoLocker, Locky, etc.)
- Free for research use

**Benign Domains:**
- URL: https://www.kaggle.com/datasets/cheedcheed/top1m (Alexa Top 1M)
- Requires free Kaggle account to download

#### Auto-Preprocessing

Once raw data files are in place, run the preprocessing pipeline:

```bash
python main.py --mode preprocess
```

This will:
- Clean and deduplicate domains
- Strip TLD suffixes using `tldextract`
- Generate the English dictionary file from NLTK
- Compute the trigram frequency table
- Create stratified 80/20 train/test splits
- Save all derived files to `data/`

## Usage

### Sequential Mode (Baseline)

Run feature extraction and classification in single-threaded mode:

```bash
python main.py --mode sequential
```

### Parallel Mode

Run with automatic core detection (uses all physical cores):

```bash
python main.py --mode parallel
```

Specify worker count manually:

```bash
python main.py --mode parallel --workers 8
```

### Benchmark Mode

Run the complete experiment suite (E1–E8) as defined in the P1 design document:

```bash
python main.py --mode benchmark --output-dir results/
```

This runs all experiments with multiple repetitions and saves results to `results/metrics.json` and plots to `results/plots/`.

Run a specific experiment only:

```bash
python main.py --mode benchmark --experiment E1    # Strong scaling
python main.py --mode benchmark --experiment E6    # DT vs RF comparison
python main.py --mode benchmark --experiment E7    # Feature ablation
```

### P3 Experimental Report (Jupyter Notebook)

The full experimental report with all 8 experiments, 6 enhancements, inline plots, and analysis:

**View without running:** The pre-executed HTML report with all outputs and plots is available at [`P3_Experimental_Report.html`](P3_Experimental_Report.html). Download and open in any browser — no Python or Jupyter required.

**Re-run from scratch:**

```bash
pip install jupyter
jupyter notebook P3_Experimental_Report.ipynb
```

The notebook runs all experiments end-to-end — data loading → feature extraction → parallel benchmarks → classification → acceptance criteria validation. Results are saved to `results/metrics.json` and plots to `results/plots/`.

### Real-Time Inference API

Start the FastAPI inference server (uses the 5-feature configuration and n_estimators=50):

```bash
pip install fastapi uvicorn
python api.py
```

Classify a single domain:

```bash
curl -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"domain": "xjk29df"}'
```

Batch classification (up to 1000 domains):

```bash
curl -X POST http://localhost:8000/classify/batch \
  -H 'Content-Type: application/json' \
  -d '{"domains": ["google", "xjk29df", "facebook", "a1b2c3d4"]}'
```

Health check:

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "domain": "xjk29df",
  "label": "DGA",
  "probability": 0.94,
  "features": {
    "length": 7.0,
    "numerical_ratio": 0.2857,
    "meaningful_word_ratio": 0.0,
    "pronounceability": 0.0001,
    "lms_percentage": 0.0
  },
  "latency_ms": 2.1
}
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | `preprocess`, `sequential`, `parallel`, or `benchmark` | `parallel` |
| `--workers` | Number of worker processes (0 = auto-detect) | `0` |
| `--data-path` | Path to the data directory | `data/` |
| `--output-dir` | Directory for benchmark results | `results/` |
| `--experiment` | Specific experiment to run (E1–E8, or `all`) | `all` |
| `--n-estimators` | Number of Random Forest trees | `100` |
| `--repetitions` | Number of repetitions per benchmark config | `5` |
| `--verbose` | Enable detailed logging | `False` |

## Running Tests

Run the full test suite:

```bash
pytest tests/ -v
```

Run with coverage report:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

Run specific test files:

```bash
pytest tests/test_features.py -v       # Unit tests for feature functions
pytest tests/test_parallel.py -v       # Correctness: parallel == sequential
pytest tests/test_boundary.py -v       # Levenshtein overlap verification
```

### Critical Test: Parallel Correctness

The most important test verifies that the parallel pipeline produces identical results to the sequential baseline:

```bash
pytest tests/test_parallel.py::test_parallel_equals_sequential -v
```

This extracts features from a 10K-domain subset using both modes and asserts element-wise equality (within floating-point tolerance).

## Experiments Overview

| ID | Description | Variable | Fixed |
|----|-------------|----------|-------|
| E1 | Strong scaling | K = 1, 2, 4, 8, 16 | N = 1M |
| E2 | High-core efficiency | K = 8, 12, 16 | N = 1M |
| E3 | Dataset scaling | N = 10K–1M | K = 8 |
| E4 | Chunk size sweep | Chunk granularity | N = 1M, K = 8 |
| E5 | RF hyperparameter | Trees = 50–500 | N = 1M, K = 8 |
| E6 | DT vs RF comparison | Model type | N = 1M, K = 8 |
| E7 | Feature ablation | 1–6 features | N = 1M, K = 8 |
| E8 | Weak scaling | N/K = 125K constant | K = 1, 2, 4, 8 |

## Performance Targets & Measured Results

| Metric | Target | Minimum | Measured | Status |
|--------|--------|---------|----------|--------|
| Speedup (8 cores) | ≥ 5.5x | ≥ 4.0x | **5.75×** | ✅ Met |
| CPU Utilization | > 85% all cores | > 70% | 51.8% system-wide (≈104% physical-core) | ✅ Explainable |
| IPC Overhead | < 5% | < 10% | **1.30%** | ✅ Met |
| Classification Accuracy | > 93% | > 90% | **93.18%** (5-feature) / 92.60% (6-feature) | ✅ Met |
| Throughput (8 cores) | > 50K domains/sec | > 30K domains/sec | **63,048 dom/s** | ✅ Met |

**Key findings from E1–E8 and enhancements:**

- Amdahl's Law (P = 0.949) validated within 3.2% up to K=8; SMT (K=16) diverges by −28.2% due to CPU-bound contention
- IPC overhead 3.7× lower than predicted — NumPy arrays serialize via pickle's zero-copy buffer protocol
- Pronounceability is the single most important feature (+25.8% accuracy jump)
- Levenshtein distance *hurts* accuracy in shuffled datasets — the 5-feature configuration is strictly better
- K=12 is the practical sweet spot (6.94× speedup, 57.9% efficiency)
- n_estimators=50 is sufficient (92.64% accuracy, 9× faster than 500 trees)

## 6 Linguistic Features

| # | Feature | Description | Complexity |
|---|---------|-------------|------------|
| 1 | Length | Character count of domain string | O(1) |
| 2 | Numerical Character % | Ratio of digit characters to total length | O(m) |
| 3 | Meaningful Word Ratio | Fraction of domain covered by English dictionary words | O(m²) |
| 4 | Pronounceability Score | Average trigram probability across character trigrams | O(m) |
| 5 | LMS Percentage | Longest meaningful substring length / total length | O(m² × D) |
| 6 | Levenshtein Edit Distance | Edit distance to the previous domain in sorted order | O(m²) |

## Team — Group 09

| Member | ID | Role |
|--------|----|------|
| Tushar Gajanan Lokhande | 2024CT05001 | Data pipeline & preprocessing |
| Pradyumna Ray | 2024CT05003 | Feature extraction implementation |
| Rohit Kumar Dubey | 2024CT05050 | Parallel engine & chunker |
| Samiran Ghosh | 2023CT05033 | Classifier & ML evaluation |
| Maliga Jaswanth | 2024CT05041 | Benchmarking & report |

## References

1. Li, Y., Xiong, K., Chin, T., & Hu, C. (2019). *A Machine Learning Framework for Domain Generation Algorithm-Based Malware Detection.* IEEE Access, 7, 32765–32782.
2. Woodbridge, J., et al. (2016). *Predicting Domain Generation Algorithms with Long Short-Term Memory Networks.* arXiv:1611.00791.
3. Catania, C., & Bromberg, S. (2021). *Real-Time Detection of Dictionary DGA Network Traffic Using Deep Learning.* SN Computer Science, 2(4), 1–17.
4. Polepally, V., et al. (2024). *DeepMaxout LSTM: A Spark Framework-Driven Deep Learning Architecture for Intrusion Detection.* Microsystem Technologies.

## License

This project is developed for academic purposes as part of AMLCCZG516 coursework at BITS Pilani. Not intended for production use.
