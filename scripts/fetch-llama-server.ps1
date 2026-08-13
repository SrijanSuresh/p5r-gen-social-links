<#
.SYNOPSIS
    Downloads prebuilt llama.cpp Windows CUDA binaries into server/vendor/.

.DESCRIPTION
    The inference backend is llama-server.exe, an upstream ggml-org binary, rather
    than the llama-cpp-python binding. The binding is a C extension pinned to a
    CPython ABI: no cp313 wheel exists, and PyPI ships source only, so installing it
    would require the CUDA Toolkit and MSVC on every machine that runs the mod.
    A prebuilt .exe has no opinion about the Python interpreter talking to it.
    See learning.md Chapter 62.

    Two archives are needed:
      llama-<build>-bin-win-cuda-<ver>-x64.zip   the executables
      cudart-llama-bin-win-cuda-<ver>-x64.zip    the CUDA runtime DLLs

    Shipping cudart is what removes the "install the CUDA Toolkit" step: the runtime
    is user-mode and redistributable, unlike the kernel-mode driver.

.PARAMETER Build
    llama.cpp release tag (e.g. b10408). Defaults to the pinned known-good build.
    Pass 'latest' to resolve the newest release instead.

.PARAMETER CudaVersion
    CUDA build variant. Defaults to 12.4.

    12.4 is deliberate on a CUDA 13.x driver. NVIDIA guarantees *backward*
    compatibility — a newer driver runs binaries built against older toolkits — so a
    12.4 build is safe on any 12.x-or-later driver. The 13.3 build would instead rely
    on minor-version compatibility within CUDA 13, which holds only while the binary
    avoids driver APIs newer than the one installed. Backward compatibility is a
    guarantee; forward compatibility is a hope.

.PARAMETER Force
    Re-download and overwrite even if llama-server.exe is already present.

.EXAMPLE
    .\scripts\fetch-llama-server.ps1
    .\scripts\fetch-llama-server.ps1 -Build latest -Force
#>
[CmdletBinding()]
param(
    [string] $Build       = 'b10408',
    [string] $CudaVersion = '12.4',
    [switch] $Force
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Repo root is the parent of scripts/, regardless of the caller's working directory.
$RepoRoot  = Split-Path -Parent $PSScriptRoot
$VendorDir = Join-Path $RepoRoot 'server\vendor'
$ExePath   = Join-Path $VendorDir 'llama-server.exe'

if ((Test-Path $ExePath) -and -not $Force) {
    Write-Host "llama-server.exe already present at $ExePath" -ForegroundColor Green
    Write-Host "Re-run with -Force to replace it."
    exit 0
}

if ($Build -eq 'latest') {
    Write-Host 'Resolving latest llama.cpp release...'
    $rel   = Invoke-RestMethod -Uri 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest' `
                               -Headers @{ 'User-Agent' = 'p5r-gen-social-links' }
    $Build = $rel.tag_name
    Write-Host "  -> $Build"
}

$base    = "https://github.com/ggml-org/llama.cpp/releases/download/$Build"
$archives = @(
    "llama-$Build-bin-win-cuda-$CudaVersion-x64.zip",
    "cudart-llama-bin-win-cuda-$CudaVersion-x64.zip"
)

New-Item -ItemType Directory -Force -Path $VendorDir | Out-Null
$tmp = Join-Path ([System.IO.Path]::GetTempPath()) "llamacpp-$Build"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

foreach ($name in $archives) {
    $url = "$base/$name"
    $zip = Join-Path $tmp $name

    if (-not (Test-Path $zip)) {
        Write-Host "Downloading $name ..." -ForegroundColor Cyan
        # Invoke-WebRequest's progress bar costs more wall-clock than the transfer on
        # large files; suppressing it is a substantial speedup, not cosmetics.
        $prevProgress = $ProgressPreference
        $ProgressPreference = 'SilentlyContinue'
        try {
            Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
        } finally {
            $ProgressPreference = $prevProgress
        }
    } else {
        Write-Host "Using cached $name" -ForegroundColor DarkGray
    }

    Write-Host "Extracting $name ..."
    # Both archives extract flat into the same directory: the binaries link against
    # the cudart DLLs at load time, so they must sit side by side.
    Expand-Archive -Path $zip -DestinationPath $VendorDir -Force
}

if (-not (Test-Path $ExePath)) {
    throw "Extraction finished but llama-server.exe is missing from $VendorDir. " +
          "The release asset layout may have changed for build $Build."
}

Write-Host ''
Write-Host "llama-server.exe ready at $ExePath" -ForegroundColor Green
& $ExePath --version
