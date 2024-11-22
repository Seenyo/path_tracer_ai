@echo off
setlocal

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3
set OPTIX_PATH=C:\ProgramData\NVIDIA Corporation\OptiX SDK 7.7.0
set VCPKG_PATH=C:\vcpkg\installed\x64-windows\include

echo CUDA_PATH is "%CUDA_PATH%"
echo OPTIX_PATH is "%OPTIX_PATH%"
echo VCPKG_PATH is "%VCPKG_PATH%"

if not exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo Error: nvcc.exe not found at "%CUDA_PATH%\bin\nvcc.exe"
    exit /b 1
)

"%CUDA_PATH%\bin\nvcc.exe" ^
    -ptx ^
    -arch=sm_86 ^
    -m64 ^
    -std=c++17 ^
    --extended-lambda ^
    -O3 ^
    --use_fast_math ^
    --relocatable-device-code=true ^
    -I "%OPTIX_PATH%\include" ^
    -I "%CUDA_PATH%\include" ^
    -I "%~dp0include" ^
    -I "%~dp0external" ^
    -I "%VCPKG_PATH%" ^
    -I "%CUDA_PATH%/include/crt" ^
    -D OPTIX_COMPATIBILITY=7700 ^
    --keep-device-functions ^
    -rdc=true ^
    "src/gpu/ptx/optix_kernels.cu" ^
    -o "src/gpu/ptx/optix_kernels.ptx"

if %ERRORLEVEL% NEQ 0 (
    echo PTX compilation failed
    exit /b 1
)

echo PTX compilation successful
exit /b 0
