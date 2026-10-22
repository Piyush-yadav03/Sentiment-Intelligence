# ===================================
# 🧠 SentiMind AI - Launch Script  
# Neural Sentiment Intelligence
# ===================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "  🧠 SentiMind AI" -ForegroundColor Magenta
Write-Host "  Neural Sentiment Intelligence" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"
$streamlit = Join-Path $root ".venv\Scripts\streamlit.exe"

if (-not (Test-Path $python)) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run setup first:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv" -ForegroundColor White
    Write-Host "  .\.venv\Scripts\activate" -ForegroundColor White
    Write-Host "  pip install -r requirements.txt" -ForegroundColor White
    Write-Host ""
    exit 1
}

Write-Host "🚀 Starting SentiMind AI..." -ForegroundColor Yellow
Write-Host "📱 Opening in browser: http://localhost:8501" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Gray
Write-Host ""

& $streamlit run (Join-Path $root "sentimind.py") @args
