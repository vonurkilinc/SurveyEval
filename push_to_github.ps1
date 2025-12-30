# PowerShell script to push SurveyEval to GitHub
# Run this AFTER creating the repository on GitHub

param(
    [Parameter(Mandatory=$true)]
    [string]$GitHubUsername,
    
    [Parameter(Mandatory=$false)]
    [string]$RepositoryName = "SurveyEval"
)

Write-Host "Setting up GitHub remote and pushing..." -ForegroundColor Green

# Add remote origin
$remoteUrl = "https://github.com/$GitHubUsername/$RepositoryName.git"
Write-Host "Adding remote: $remoteUrl" -ForegroundColor Yellow
git remote add origin $remoteUrl

# Verify remote was added
Write-Host "`nVerifying remote configuration..." -ForegroundColor Yellow
git remote -v

# Push to GitHub
Write-Host "`nPushing to GitHub..." -ForegroundColor Yellow
git push -u origin main

Write-Host "`nDone! Your repository is now on GitHub." -ForegroundColor Green
Write-Host "Visit: https://github.com/$GitHubUsername/$RepositoryName" -ForegroundColor Cyan

