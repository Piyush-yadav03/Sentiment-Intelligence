# Improved Training Configuration for Better Accuracy
# Expected: 70-80% accuracy, 10-15 minutes on CPU

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "SentimentLab AI - Improved Training" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Training samples: 800" -ForegroundColor White
Write-Host "  Test samples: 300" -ForegroundColor White
Write-Host "  Epochs: 3" -ForegroundColor White
Write-Host "  Batch size: 8" -ForegroundColor White
Write-Host ""
Write-Host "Expected Results:" -ForegroundColor Green
Write-Host "  Accuracy: 75-85%" -ForegroundColor White
Write-Host "  Training time: 10-15 minutes" -ForegroundColor White
Write-Host ""
Write-Host "Starting training..." -ForegroundColor Cyan
Write-Host ""

.\.venv\Scripts\python.exe src/train.py `
    --train-size 800 `
    --test-size 300 `
    --epochs 3 `
    --batch-size 8 `
    --max-length 96 `
    --temperature 2.0 `
    --alpha 0.5

Write-Host ""
Write-Host "==================================" -ForegroundColor Green
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green
Write-Host ""
Write-Host "Check results/metrics.json for your actual accuracy" -ForegroundColor Yellow
Write-Host "Refresh the web app to see updated results" -ForegroundColor Yellow
