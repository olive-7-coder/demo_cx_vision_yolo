# RoboMaster Detect - Build Script (PowerShell)
# Sets up MSVC environment manually and builds with Ninja

Write-Host "=== RoboMaster Armor Plate Detector - Build ===" -ForegroundColor Cyan

$msvcRoot  = "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.51.36231"
$sdkRoot   = "C:\Program Files (x86)\Windows Kits\10"
$sdkVer    = "10.0.26100.0"
$compiler  = "$msvcRoot\bin\Hostx64\x64"
$sdkBin    = "$sdkRoot\bin\$sdkVer\x64"

$env:PATH   = "$compiler;$sdkBin;E:\OpenCV\opencv\build\x64\vc16\bin;$env:PATH"
$env:INCLUDE = "$msvcRoot\include;$sdkRoot\Include\$sdkVer\ucrt;$sdkRoot\Include\$sdkVer\um;$sdkRoot\Include\$sdkVer\shared;$sdkRoot\Include\$sdkVer\winrt"
$env:LIB     = "$msvcRoot\lib\x64;$sdkRoot\Lib\$sdkVer\ucrt\x64;$sdkRoot\Lib\$sdkVer\um\x64"

$cmake = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$cl    = "$compiler\cl.exe"
$mt    = "$sdkBin\mt.exe"
$src   = $PSScriptRoot
$bld   = Join-Path $src "build"

# Step 1: CMake configure
Write-Host "[1/2] Configuring..." -ForegroundColor Yellow
Remove-Item -Recurse -Force $bld -ErrorAction SilentlyContinue
& $cmake -B $bld -S $src -G Ninja "-DCMAKE_CXX_COMPILER=$cl" "-DCMAKE_MAKE_PROGRAM=$ninja" "-DCMAKE_MT=$mt" "-DOpenCV_DIR=E:/OpenCV/opencv/build"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED" -ForegroundColor Red; Read-Host "Press Enter"; exit 1 }

# Step 2: Build
Write-Host "[2/2] Building..." -ForegroundColor Yellow
& $cmake --build $bld
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED" -ForegroundColor Red; Read-Host "Press Enter"; exit 1 }

Write-Host "`n=== BUILD SUCCESS ===" -ForegroundColor Green
Write-Host "Executable: $bld\detect.exe" -ForegroundColor Green
Write-Host "`nRun with: .\build\detect.exe --model best.onnx --input test.jpg" -ForegroundColor White
Read-Host "Press Enter"
