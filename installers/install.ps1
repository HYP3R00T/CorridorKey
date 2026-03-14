# CorridorKey Installer for Windows
#
# Usage:
#   irm https://corridorkey.dev/install.ps1 | iex
# Or locally:
#   powershell -ExecutionPolicy Bypass -File install.ps1

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$host.UI.RawUI.WindowTitle = "CorridorKey Installer"

function Write-Step([string]$msg) { Write-Host ""; Write-Host ">>> $msg" -ForegroundColor Cyan }
function Write-Ok([string]$msg)   { Write-Host "    [OK] $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "    [WARN] $msg" -ForegroundColor Yellow }
function Write-Fail([string]$msg) { Write-Host ""; Write-Host "    [ERROR] $msg" -ForegroundColor Red }

# -------------------------------------------------------
# Banner
# -------------------------------------------------------
Write-Host ""
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "    CorridorKey - AI Green Screen Keyer"           -ForegroundColor Cyan
Write-Host "    Windows Installer"                             -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

# -------------------------------------------------------
# Step 1: Pick GPU variant
# -------------------------------------------------------
Write-Host ""
Write-Host "Which GPU do you have?" -ForegroundColor White
Write-Host ""
Write-Host "  [1] NVIDIA GPU (CUDA)"
Write-Host "  [2] No GPU / CPU only"
Write-Host ""

$choice = ""
while ($choice -notin @("1", "2")) {
    $choice = Read-Host "Enter choice [1/2]"
}

switch ($choice) {
    "1" { $package = "corridorkey-cli[cuda]"; $backend = "NVIDIA (CUDA)" }
    "2" { $package = "corridorkey-cli";       $backend = "CPU" }
}

Write-Host ""
Write-Ok "Selected: $backend"

# -------------------------------------------------------
# Step 2: Ensure uv is installed
# -------------------------------------------------------
Write-Step "Checking for uv package manager..."

$uvCmd = Get-Command "uv" -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    Write-Host "    uv not found. Installing..."
    try {
        Invoke-RestMethod "https://astral.sh/uv/install.ps1" | Invoke-Expression
    } catch {
        Write-Fail "Failed to install uv: $_"
        Write-Host "    Install manually: https://docs.astral.sh/uv/" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }

    # Refresh PATH for this session
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "User") + ";$env:PATH"

    $uvCmd = Get-Command "uv" -ErrorAction SilentlyContinue
    if (-not $uvCmd) {
        Write-Fail "uv installed but not found on PATH. Restart PowerShell and run this script again."
        Read-Host "Press Enter to exit"
        exit 1
    }
}
Write-Ok "uv is ready."

# -------------------------------------------------------
# Step 3: Install corridorkey-cli
# -------------------------------------------------------
Write-Step "Installing $package..."

try {
    & uv tool install $package
    if ($LASTEXITCODE -ne 0) { throw "uv exited with code $LASTEXITCODE" }
} catch {
    Write-Fail "Installation failed: $_"
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Ok "corridorkey-cli installed."

# -------------------------------------------------------
# Step 4: First-time setup
# -------------------------------------------------------
Write-Step "Running first-time setup..."
Write-Host "    You will be asked whether to download the inference model (~400 MB)."
Write-Host ""

& corridorkey init

# -------------------------------------------------------
# Step 5: Create Desktop shortcut
# -------------------------------------------------------
Write-Step "Creating Desktop shortcut..."

$desktopPath = [System.Environment]::GetFolderPath("Desktop")
$launcherPath = Join-Path $desktopPath "CorridorKey_launcher.bat"
$shortcutPath = Join-Path $desktopPath "CorridorKey.lnk"

$launcherContent = @'
@echo off
if "%~1"=="" (
    echo [ERROR] No folder provided.
    echo.
    echo USAGE: Drag and drop a clips folder onto the CorridorKey shortcut.
    echo.
    pause
    exit /b 1
)
set "TARGET=%~1"
if "%TARGET:~-1%"=="\" set "TARGET=%TARGET:~0,-1%"
echo Starting CorridorKey...
echo Target: "%TARGET%"
echo.
corridorkey wizard "%TARGET%"
pause
'@

Set-Content -Path $launcherPath -Value $launcherContent -Encoding ASCII

$wsh = New-Object -ComObject WScript.Shell
$shortcut = $wsh.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $launcherPath
$shortcut.Description = "CorridorKey - AI Green Screen Keyer"
$shortcut.WorkingDirectory = $desktopPath
$shortcut.Save()

Write-Ok "Shortcut created on Desktop: CorridorKey.lnk"
Write-Host "    Drag a clips folder onto 'CorridorKey' on your Desktop to start." -ForegroundColor DarkGray

# -------------------------------------------------------
# Done
# -------------------------------------------------------
Write-Host ""
Write-Host "===================================================" -ForegroundColor Green
Write-Host "    Setup complete! You are ready to key."         -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to close"
