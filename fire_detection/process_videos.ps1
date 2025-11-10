# Process videos with trained fire detection model
# UTF-8 Encoding

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Fire Detection Video Processing" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Activate Python virtual environment if exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
}

# Find latest trained model
Write-Host "Searching for trained models..." -ForegroundColor Yellow
$LATEST_MODEL = Get-ChildItem -Path "fire_detection_runs" -Recurse -Filter "best.pt" -ErrorAction SilentlyContinue | 
    Where-Object { $_.FullName -like "*\weights\best.pt" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -eq $LATEST_MODEL) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Error: No trained model found!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "You need to train a model first:" -ForegroundColor Yellow
    Write-Host "  1. Run .\train_fire.ps1" -ForegroundColor White
    Write-Host "  or" -ForegroundColor White
    Write-Host "  2. python train_fire_detection.py" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host "Found model: $($LATEST_MODEL.FullName)" -ForegroundColor Green
Write-Host ""

# Check videos in assets folder
Write-Host "Checking videos in assets folder..." -ForegroundColor Yellow
$videos = Get-ChildItem -Path "assets" -Filter "*.mp4" -ErrorAction SilentlyContinue

if ($videos.Count -eq 0) {
    Write-Host ""
    Write-Host "Error: No videos found in assets folder!" -ForegroundColor Red
    Read-Host "Press Enter to continue"
    exit 1
}

foreach ($video in $videos) {
    Write-Host "  - $($video.Name)" -ForegroundColor White
}

Write-Host ""
Write-Host "Total $($videos.Count) videos to process." -ForegroundColor Green
Write-Host ""

# Select confidence threshold
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Select Confidence Threshold" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "1. Low (0.3) - Detect all possibilities" -ForegroundColor White
Write-Host "2. Medium (0.5) - Balanced detection (Recommended)" -ForegroundColor Green
Write-Host "3. High (0.7) - Only confident detections" -ForegroundColor White
Write-Host "4. Custom" -ForegroundColor White
Write-Host ""
$conf_choice = Read-Host "Select (1-4)"

switch ($conf_choice) {
    "1" { $CONFIDENCE = "0.3" }
    "2" { $CONFIDENCE = "0.5" }
    "3" { $CONFIDENCE = "0.7" }
    "4" { $CONFIDENCE = Read-Host "Confidence threshold (0.0-1.0)" }
    default { $CONFIDENCE = "0.5" }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting Video Processing" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Model: $($LATEST_MODEL.FullName)" -ForegroundColor Yellow
Write-Host "Confidence: $CONFIDENCE" -ForegroundColor Yellow
Write-Host "Video Count: $($videos.Count)" -ForegroundColor Yellow
Write-Host "Output Folder: fire_detected_videos\" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Run video processing
python process_fire_videos.py --model "$($LATEST_MODEL.FullName)" --video-dir assets --confidence $CONFIDENCE

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Processing Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results: fire_detected_videos\" -ForegroundColor Cyan

if (Test-Path "fire_detected_videos") {
    $result_videos = Get-ChildItem -Path "fire_detected_videos" -Filter "*.mp4" -ErrorAction SilentlyContinue
    foreach ($rv in $result_videos) {
        Write-Host "  - $($rv.Name)" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Double-click video files to play them." -ForegroundColor Yellow
Write-Host ""

Read-Host "Press Enter to continue"
