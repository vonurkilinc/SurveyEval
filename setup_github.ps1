# PowerShell script to set up GitHub repository
# Run this after creating the repository on GitHub

Write-Host "GitHub Repository Setup Script" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "Step 1: Create a new repository on GitHub:" -ForegroundColor Yellow
Write-Host "   1. Go to https://github.com/new" -ForegroundColor Cyan
Write-Host "   2. Repository name: SurveyEval" -ForegroundColor Cyan
Write-Host "   3. Choose Public or Private" -ForegroundColor Cyan
Write-Host "   4. DO NOT initialize with README, .gitignore, or license" -ForegroundColor Cyan
Write-Host "   5. Click 'Create repository'" -ForegroundColor Cyan
Write-Host ""
Write-Host "Step 2: After creating the repository, run:" -ForegroundColor Yellow
Write-Host "   git remote add origin https://github.com/YOUR_USERNAME/SurveyEval.git" -ForegroundColor Cyan
Write-Host "   git branch -M main" -ForegroundColor Cyan
Write-Host "   git push -u origin main" -ForegroundColor Cyan
Write-Host ""
Write-Host "Or if you prefer SSH:" -ForegroundColor Yellow
Write-Host "   git remote add origin git@github.com:YOUR_USERNAME/SurveyEval.git" -ForegroundColor Cyan
Write-Host "   git branch -M main" -ForegroundColor Cyan
Write-Host "   git push -u origin main" -ForegroundColor Cyan
Write-Host ""

