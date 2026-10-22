# Setup Guide

## Prerequisites

- Python 3.10, 3.11, or 3.12
- 4GB RAM (8GB recommended)
- 5GB free disk space
- Windows 10/11, macOS 10.15+, or Linux

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Piyush-yadav03/Sentiment-Intelligence.git
cd Sentiment-Intelligence
```

### 2. Create Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Training Models

### Quick Test (2-5 minutes)
```bash
python src/train.py --train-size 200 --test-size 100 --epochs 1 --batch-size 4
```

### Standard Training (15-20 minutes)
```bash
python src/train.py --train-size 800 --test-size 300 --epochs 3 --batch-size 8
```

### Best Accuracy (30-45 minutes)
```bash
python src/train.py --train-size 2000 --test-size 500 --epochs 5 --batch-size 16
```

## Launch Application

```bash
streamlit run sentimind.py
```

Open browser at: http://localhost:8501

## Running Tests

```bash
pytest tests/ -v
```

## Troubleshooting

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

### Out of Memory
- Reduce `--batch-size` to 2 or 4
- Reduce `--train-size` and `--test-size`
- Close other applications

### Slow Training
- Reduce `--max-length` to 64
- Reduce `--train-size`
- Use Quick Test configuration

## Next Steps

- View [PROJECT_STATUS_REPORT.md](../PROJECT_STATUS_REPORT.md) for detailed documentation
- Check [report/methodology.md](../report/methodology.md) for technical details
- Explore the web application pages for interactive analysis
