# Best Accuracy Training Configuration
# Expected: 80-88% accuracy, 30-45 minutes on CPU

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "SentimentLab AI - Maximum Accuracy" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Training samples: 2000" -ForegroundColor White
Write-Host "  Test samples: 500" -ForegroundColor White
Write-Host "  Epochs: 5" -ForegroundColor White
Write-Host "  Batch size: 16" -ForegroundColor White
Write-Host "  Max length: 128 (more context)" -ForegroundColor White
Write-Host ""
Write-Host "Expected Results:" -ForegroundColor Green
Write-Host "  Accuracy: 80-88%" -ForegroundColor White
Write-Host "  Training time: 30-45 minutes" -ForegroundColor White
Write-Host ""
Write-Host "⚠️  This will take longer but give much better results!" -ForegroundColor Yellow
Write-Host ""
Write-Host "Starting training..." -ForegroundColor Cyan
Write-Host ""

.\.venv\Scripts\python.exe src/train.py `
    --train-size 2000 `
    --test-size 500 `
    --epochs 5 `
    --batch-size 16 `
    --max-length 128 `
    --temperature 2.0 `
    --alpha 0.5

Write-Host ""
Write-Host "==================================" -ForegroundColor Green
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green
Write-Host ""
Write-Host "Check results/metrics.json for your actual accuracy" -ForegroundColor Yellow
Write-Host "Refresh the web app to see updated results" -ForegroundColor Yellow
