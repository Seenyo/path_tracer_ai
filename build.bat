@echo off
setlocal

:: Initialize Visual Studio environment
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

:: Create and enter build directory
if not exist cmake-build-release mkdir cmake-build-release
cd cmake-build-release

:: Configure with CMake
cmake -G "Visual Studio 17 2022" -A x64 ..

:: Build
cmake --build . --config Release

:: Return to original directory
cd ..

endlocal
