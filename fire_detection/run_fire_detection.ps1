# Fire Detection Project Main Script
# UTF-8 Encoding

$ErrorActionPreference = "Stop"

function Show-Menu {
    Clear-Host
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "                                                            " -ForegroundColor Cyan
    Write-Host "         Fire Detection Project - Auto Execution            " -ForegroundColor Yellow
    Write-Host "                                                            " -ForegroundColor Cyan
    Write-Host "    YOLO-based Fire Detection Model Training & Processing  " -ForegroundColor White
    Write-Host "                                                            " -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Check-Location {
    if (-not (Test-Path "fire_dataset.yaml")) {
        Write-Host "Error: Please run from monoculus folder!" -ForegroundColor Red
        Write-Host ""
        Write-Host "Current location: $(Get-Location)" -ForegroundColor Yellow
        Write-Host "Required location: monoculus folder" -ForegroundColor Yellow
        Write-Host ""
        Read-Host "Press Enter to continue"
        exit 1
    }
    
    Write-Host "Current location: $(Get-Location)" -ForegroundColor Green
    Write-Host ""
}

function Show-Results {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Results" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Check training results
    if (Test-Path "fire_detection_runs") {
        Write-Host "Trained Models:" -ForegroundColor Yellow
        $models = Get-ChildItem -Path "fire_detection_runs" -Recurse -Filter "best.pt" -ErrorAction SilentlyContinue | 
            Where-Object { $_.FullName -like "*\weights\best.pt" }
        
        foreach ($model in $models) {
            Write-Host "   - $($model.FullName)" -ForegroundColor Green
        }
        Write-Host ""
    } else {
        Write-Host "No trained models found." -ForegroundColor Yellow
        Write-Host ""
    }
    
    # Check processed videos
    if (Test-Path "fire_detected_videos") {
        Write-Host "Processed Videos:" -ForegroundColor Yellow
        $videos = Get-ChildItem -Path "fire_detected_videos" -Filter "*.mp4" -ErrorAction SilentlyContinue
        
        foreach ($video in $videos) {
            Write-Host "   - $($video.Name)" -ForegroundColor Green
        }
        Write-Host ""
        
        $open = Read-Host "Open video folder? (Y/N)"
        if ($open -eq "Y" -or $open -eq "y") {
            Start-Process "fire_detected_videos"
        }
    } else {
        Write-Host "No processed videos found." -ForegroundColor Yellow
        Write-Host ""
    }
}

# Main logic
Show-Menu
Check-Location

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "                       Menu" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Train Model" -ForegroundColor White
Write-Host "2. Process Videos" -ForegroundColor White
Write-Host "3. Full Process (Train + Process)" -ForegroundColor Green
Write-Host "4. View Results" -ForegroundColor White
Write-Host "5. Exit" -ForegroundColor White
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
$choice = Read-Host "Select (1-5)"

switch ($choice) {
    "1" {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "Starting Model Training" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Cyan
        & ".\train_fire.ps1"
    }
    "2" {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "Starting Video Processing" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Cyan
        & ".\process_videos.ps1"
    }
    "3" {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "Full Process Execution" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Step 1/2: Model Training" -ForegroundColor Yellow
        Write-Host "------------------------------------------------------------" -ForegroundColor Cyan
        & ".\train_fire.ps1"
        
        Write-Host ""
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "Step 2/2: Video Processing" -ForegroundColor Yellow
        Write-Host "------------------------------------------------------------" -ForegroundColor Cyan
        & ".\process_videos.ps1"
        
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Green
        Write-Host "Full Process Complete!" -ForegroundColor Green
        Write-Host "============================================================" -ForegroundColor Green
    }
    "4" {
        Show-Results
    }
    "5" {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "Goodbye!" -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host ""
        exit 0
    }
    default {
        Write-Host ""
        Write-Host "Invalid selection." -ForegroundColor Red
    }
}

Write-Host ""
Read-Host "Press Enter to continue"
