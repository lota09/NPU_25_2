# Fire Detection Model Training Script
# UTF-8 Encoding

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Fire Detection YOLO Model Training" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Activate Python virtual environment if exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
}

# Check required packages
Write-Host "Checking required packages..." -ForegroundColor Yellow
$packagesInstalled = $true
try {
    python -c "import torch, ultralytics, cv2" 2>$null
    if ($LASTEXITCODE -ne 0) {
        $packagesInstalled = $false
    }
} catch {
    $packagesInstalled = $false
}

if (-not $packagesInstalled) {
    Write-Host ""
    Write-Host "Required packages are not installed." -ForegroundColor Red
    $install = Read-Host "Do you want to install packages? (Y/N)"
    if ($install -eq "Y" -or $install -eq "y") {
        Write-Host "Installing packages..." -ForegroundColor Yellow
        pip install ultralytics torch torchvision opencv-python numpy
    } else {
        Write-Host "Installation cancelled. Please install required packages first:" -ForegroundColor Red
        Write-Host "pip install ultralytics torch torchvision opencv-python numpy" -ForegroundColor Yellow
        Read-Host "Press Enter to continue"
        exit 1
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Training Options" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "1. Quick Training (yolov8n, 50 epochs) - ~30 min" -ForegroundColor White
Write-Host "2. Standard Training (yolov8n, 100 epochs) - ~1 hour (Recommended)" -ForegroundColor Green
Write-Host "3. Advanced Training (yolov8s, 150 epochs) - ~3 hours" -ForegroundColor White
Write-Host "4. Custom" -ForegroundColor White
Write-Host ""
$choice = Read-Host "Select (1-4)"

switch ($choice) {
    "1" {
        $MODEL = "yolov8n.pt"
        $EPOCHS = "50"
        $BATCH = "16"
        $NAME = "fire_quick"
    }
    "2" {
        $MODEL = "yolov8n.pt"
        $EPOCHS = "100"
        $BATCH = "16"
        $NAME = "fire_standard"
    }
    "3" {
        $MODEL = "yolov8s.pt"
        $EPOCHS = "150"
        $BATCH = "16"
        $NAME = "fire_advanced"
    }
    "4" {
        Write-Host ""
        $MODEL = Read-Host "Model (yolov8n.pt/yolov8s.pt/yolov8m.pt)"
        $EPOCHS = Read-Host "Epochs"
        $BATCH = Read-Host "Batch size"
        $NAME = Read-Host "Experiment name"
    }
    default {
        Write-Host "Invalid selection." -ForegroundColor Red
        Read-Host "Press Enter to continue"
        exit 1
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Training" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Model: $MODEL" -ForegroundColor Yellow
Write-Host "Epochs: $EPOCHS" -ForegroundColor Yellow
Write-Host "Batch Size: $BATCH" -ForegroundColor Yellow
Write-Host "Experiment Name: $NAME" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Run training
python train_fire_detection.py --model $MODEL --epochs $EPOCHS --batch $BATCH --name $NAME

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Training Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results: fire_detection_runs\$NAME\" -ForegroundColor Cyan
Write-Host "Best Model: fire_detection_runs\$NAME\weights\best.pt" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next Step: Video Processing" -ForegroundColor Yellow
Write-Host "python process_fire_videos.py --model fire_detection_runs\$NAME\weights\best.pt --video-dir assets" -ForegroundColor White
Write-Host ""

Read-Host "Press Enter to continue"
