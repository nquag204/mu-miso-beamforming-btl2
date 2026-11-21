# One-click reproducibility pipeline for BTL2 Topic 1 (Windows PowerShell)
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:MPLBACKEND = "Agg"

Write-Host "=== [1/4] Running pytest verification suite ===" -ForegroundColor Cyan
python -m pytest (Join-Path $Root "tests") -v

Write-Host "=== [2/4] Running Monte-Carlo simulations and 300-DPI figure generation ===" -ForegroundColor Cyan
python (Join-Path $Root "main.py")

Write-Host "=== [3/4] Generating LaTeX report (baocao_btl2.tex) ===" -ForegroundColor Cyan
python (Join-Path $Root "report\build_tex.py")

Write-Host "=== [4/4] Generating Word report (Bao_cao_BTL02_DeTai1.docx) ===" -ForegroundColor Cyan
python (Join-Path $Root "report\build_docx.py")

Write-Host "=== Pipeline completed successfully! ===" -ForegroundColor Green
