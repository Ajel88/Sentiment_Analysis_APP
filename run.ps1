Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🚀 Starting Sentiment Analysis Docker" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Stop existing
docker-compose down 2>$null

# Start
Write-Host "Starting containers..." -ForegroundColor Yellow
docker-compose up

Write-Host "`nIf containers start successfully:" -ForegroundColor Green
Write-Host "🌐 Open: http://localhost:7860" -ForegroundColor White