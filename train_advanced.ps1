# Advanced Training with Optimized Hyperparameters
# Expected: 82-90% accuracy, 25-40 minutes on CPU

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "SentimentLab AI - Advanced Training" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "🎯 Optimized Configuration:" -ForegroundColor Yellow
Write-Host "  Training samples: 1500" -ForegroundColor White
Write-Host "  Test samples: 400" -ForegroundColor White
Write-Host "  Epochs: 4" -ForegroundColor White
Write-Host "  Batch size: 12" -ForegroundColor White
Write-Host "  Max length: 128 (more context)" -ForegroundColor White
Write-Host "  Temperature: 3.0 (softer targets)" -ForegroundColor White
Write-Host "  Alpha: 0.4 (more teacher influence)" -ForegroundColor White
Write-Host ""
Write-Host "✨ Why These Settings:" -ForegroundColor Green
Write-Host "  • Larger dataset for better learning" -ForegroundColor White
Write-Host "  • Higher temperature reveals more knowledge" -ForegroundColor White
Write-Host "  • Lower alpha gives more weight to teacher" -ForegroundColor White
Write-Host "  • Longer sequences capture more context" -ForegroundColor White
Write-Host ""
Write-Host "Expected Results:" -ForegroundColor Green
Write-Host "  Baseline: 78-84% accuracy" -ForegroundColor White
Write-Host "  Distilled: 82-90% accuracy" -ForegroundColor White
Write-Host "  Training time: 25-40 minutes" -ForegroundColor White
Write-Host ""
Write-Host "Starting training..." -ForegroundColor Cyan
Write-Host ""

.\.venv\Scripts\python.exe src/train.py `
    --train-size 1500 `
    --test-size 400 `
    --epochs 4 `
    --batch-size 12 `
    --max-length 128 `
    --temperature 3.0 `
    --alpha 0.4

Write-Host ""
Write-Host "==================================" -ForegroundColor Green
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Results saved to results/metrics.json" -ForegroundColor Yellow
Write-Host "🌐 Refresh web app to see updated results" -ForegroundColor Yellow
Write-Host ""
Write-Host "💡 Compare with previous runs in experiment log!" -ForegroundColor Cyan
