<#
PowerShell setup script for creating a Python 3.11 virtual environment
and installing project dependencies. Designed for Windows PowerShell.

Usage:
  1) Ensure Python 3.11 is installed (available via `py -3.11`).
  2) From project root run: `.\ackup\setup_venv.ps1` (or `./setup_venv.ps1`).

This script will:
  - create a venv at `.venv311` (configurable)
  - upgrade pip inside the venv
  - attempt to install CUDA wheels for PyTorch (cu130 by default)
    and fall back to CPU builds if CUDA wheel install fails
  - install the other dependencies from `requirements.txt`
#>

param(
    [string]$PythonCmd = "py -3.11",
    [string]$VenvDir = ".venv311",
    [string]$TorchCudaTag = "cu130"
)

function Run-Proc($commandLine) {
  # Run command via cmd.exe /c so we can accept complex commands like "py -3.11 -m venv ..."
  # Capture combined stdout/stderr in PowerShell and return exit code
  try {
    $output = & cmd.exe /c $commandLine 2>&1
    $exit = $LASTEXITCODE
    return @{ ExitCode = $exit; StdOut = ($output -join "`n"); StdErr = "" }
  } catch {
    return @{ ExitCode = 1; StdOut = ""; StdErr = $_.Exception.Message }
  }
}

Write-Host "Using Python command: $PythonCmd"

# Verify python command exists; try fallbacks
Write-Host "Checking Python availability for: $PythonCmd"
$check = Run-Proc("$PythonCmd --version")
if ($check.ExitCode -ne 0) {
  Write-Warning "Primary python command failed: $PythonCmd --version\nOutput:\n$($check.StdOut)\n$($check.StdErr)"
  $candidates = @('py -3.11','python3.11','python3','python')
  $found = $null
  foreach ($cand in $candidates) {
    $ch = Run-Proc("$cand --version")
    if ($ch.ExitCode -eq 0) {
      Write-Host "Found python candidate: $cand -> $($ch.StdOut)"
      $found = $cand
      break
    }
  }
  if ($found -ne $null) {
    $PythonCmd = $found
    Write-Host "Using fallback python command: $PythonCmd"
  } else {
    Write-Error "No suitable Python interpreter found. Please install Python 3.11 and ensure it's on PATH (or pass -PythonCmd).";
    exit 1
  }
}

# Create virtual environment
Write-Host "Creating virtual environment in '$VenvDir'..."
$res = Run-Proc("$PythonCmd -m venv `"$VenvDir`"")
if ($res.ExitCode -ne 0) {
  Write-Error "Failed to create venv. Exit=$($res.ExitCode)\nStdOut:\n$($res.StdOut)\nStdErr:\n$($res.StdErr)"
  exit 1
}

$venvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Could not find venv python at $venvPython. Activate path or install Python 3.11 and retry."
    exit 1
}

Write-Host "Upgrading pip inside venv..."
Run-Proc("`"$venvPython`" -m pip install --upgrade pip") | Out-Null

Write-Host "Attempting to install PyTorch with CUDA tag: $TorchCudaTag"
$installTorchCmd = "-m pip install --index-url https://download.pytorch.org/whl/$TorchCudaTag torch torchvision --upgrade"
# build full install command using the python executable from the venv
$r = Run-Proc("`"$venvPython`" $installTorchCmd")
if ($r.ExitCode -ne 0 -or $r.StdErr -match "No matching distribution") {
    Write-Warning "Failed to install CUDA wheel for PyTorch ($TorchCudaTag). Falling back to CPU build or best available wheel.\n$r.StdOut\n$r.StdErr"
    Run-Proc("`"$venvPython`" -m pip install --upgrade torch torchvision") | Out-Null
} else {
    Write-Host "Installed PyTorch wheels for $TorchCudaTag"
}

Write-Host "Installing other requirements from requirements.txt..."
Run-Proc("`"$venvPython`" -m pip install -r requirements.txt") | Out-Null

Write-Host "Setup finished. To activate the venv in PowerShell run:" -ForegroundColor Green
Write-Host "  .\$VenvDir\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "Then check GPU availability with:" -ForegroundColor Green
Write-Host '  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"' -ForegroundColor Cyan

Write-Host "If you want a different Python executable or CUDA tag, re-run this script with parameters." -ForegroundColor Yellow
Write-Host "Example: .\setup_venv.ps1 -PythonCmd 'C:\\Python311\\python.exe' -TorchCudaTag cu130"