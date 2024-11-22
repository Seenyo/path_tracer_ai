# Build script for path tracer

param(
    [switch]$Clean,
    [switch]$Rebuild,
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release"
)

# Check if running as administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
$isAdmin = $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "Please run this script as Administrator" -ForegroundColor Red
    exit 1
}

# Configuration
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3"
$optixPath = "C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.7.0"
$vcpkgPath = "C:\vcpkg"
$vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community"

# Check dependencies
function Check-Dependency {
    param (
        [string]$path,
        [string]$name,
        [switch]$required
    )
    if (-not (Test-Path $path)) {
        if ($required) {
            Write-Host "Error: Required dependency $name not found at $path" -ForegroundColor Red
            Write-Host "Please install $name and update the path in this script" -ForegroundColor Yellow
            exit 1
        }
        else {
            Write-Host "Warning: Optional dependency $name not found at $path" -ForegroundColor Yellow
            return $false
        }
    }
    return $true
}

Write-Host "Checking dependencies..." -ForegroundColor Green

# Check required dependencies
Check-Dependency -path $cudaPath -name "CUDA Toolkit" -required
Check-Dependency -path $optixPath -name "OptiX SDK" -required
Check-Dependency -path $vcpkgPath -name "vcpkg" -required
Check-Dependency -path $vsPath -name "Visual Studio 2022" -required

# Set environment variables
$env:CUDA_PATH = $cudaPath
$env:OptiX_ROOT_DIR = $optixPath
$env:PATH = "$cudaPath\bin;$env:PATH"

# Create build directory
$buildDir = "build"
if ($Clean -or $Rebuild) {
    if (Test-Path $buildDir) {
        Write-Host "Cleaning build directory..." -ForegroundColor Yellow
        Remove-Item -Path $buildDir -Recurse -Force
    }
}

if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

# Install dependencies using vcpkg if not already installed
Write-Host "Checking vcpkg packages..." -ForegroundColor Green
$vcpkgPackages = @(
    "glm:x64-windows",
    "cxxopts:x64-windows",
    "tinyobjloader:x64-windows",
    "stb:x64-windows"
)

foreach ($package in $vcpkgPackages) {
    & "$vcpkgPath\vcpkg.exe" install $package
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to install package: $package" -ForegroundColor Red
        exit 1
    }
}

# Configure CMake
Write-Host "Configuring CMake..." -ForegroundColor Green
Push-Location $buildDir
$cmakeArgs = @(
    "-G", "Visual Studio 17 2022",
    "-A", "x64",
    "-DCMAKE_TOOLCHAIN_FILE=$vcpkgPath\scripts\buildsystems\vcpkg.cmake",
    "-DCUDA_TOOLKIT_ROOT_DIR=$cudaPath",
    "-DOptiX_ROOT_DIR=$optixPath",
    "-DCMAKE_BUILD_TYPE=$Configuration",
    ".."
)
& cmake $cmakeArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed" -ForegroundColor Red
    Pop-Location
    exit 1
}

# Build project
Write-Host "Building project..." -ForegroundColor Green
& cmake --build . --config $Configuration --parallel
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed" -ForegroundColor Red
    Pop-Location
    exit 1
}

Pop-Location

Write-Host "`nBuild completed successfully!" -ForegroundColor Green
Write-Host "`nUsage examples:" -ForegroundColor Yellow
Write-Host ".\build\bin\$Configuration\path_tracer.exe --mode gpu --width 1920 --height 1080 --samples 100" -ForegroundColor Cyan
Write-Host ".\build\bin\$Configuration\path_tracer.exe --mode cpu --width 800 --height 450 --samples 10" -ForegroundColor Cyan
Write-Host "`nFor more options, run:" -ForegroundColor Yellow
Write-Host ".\build\bin\$Configuration\path_tracer.exe --help" -ForegroundColor Cyan

# Create symbolic link to IronMan model if it exists
$modelPath = "IronMan"
if (Test-Path $modelPath) {
    $targetPath = ".\build\bin\$Configuration\$modelPath"
    if (-not (Test-Path $targetPath)) {
        Write-Host "`nCreating symbolic link to IronMan model..." -ForegroundColor Green
        New-Item -ItemType Junction -Path $targetPath -Target (Resolve-Path $modelPath)
    }
}

# Print GPU information
Write-Host "`nGPU Information:" -ForegroundColor Green
& "$cudaPath\extras\demo_suite\deviceQuery.exe"
