# One-click verification & simulation pipeline for BTL2 Topic 1 (Windows PowerShell)
# Usage:
#   .\run_all.ps1          # Run all 21 pytest checks + full 100-trial Monte-Carlo + 8 figures
#   .\run_all.ps1 -Quick   # Run all 21 pytest checks + fast 15-trial smoke simulation + 8 figures
param(
    [switch]$Quick
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:MPLBACKEND = "Agg"

Write-Host "=== [1/2] Running pytest verification suite (21 tests + Hypothesis) ===" -ForegroundColor Cyan
python -m pytest (Join-Path $Root "tests") -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if ($Quick) {
    Write-Host "=== [2/2] Running QUICK Monte-Carlo (15 trials) & generating 8 figures ===" -ForegroundColor Cyan
    python (Join-Path $Root "main.py") --quick
} else {
    Write-Host "=== [2/2] Running FULL Monte-Carlo (100 trials) & generating 8 figures ===" -ForegroundColor Cyan
    python (Join-Path $Root "main.py")
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Pipeline completed successfully! Results saved in results/data & results/figures ===" -ForegroundColor Green
