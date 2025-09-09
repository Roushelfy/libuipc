@echo off
REM LibUIPC Installation Test Runner

echo 🚀 LibUIPC Installation Test Suite
echo.

REM Check if Docker is available
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Docker is required but not found
    echo Please install Docker and try again
    exit /b 1
)

REM Default values
set "METHOD=auto"
set "DISTRO=ubuntu"

REM Parse arguments
:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--method" (
    set "METHOD=%~2"
    shift /1
    shift /1
    goto parse_args
)
if "%~1"=="--distro" (
    set "DISTRO=%~2"
    shift /1
    shift /1
    goto parse_args
)
if "%~1"=="--help" (
    echo Usage: %~n0 [--method auto^|pip] [--distro ubuntu^|centos^|all]
    echo.
    echo Options:
    echo   --method    Installation method to test ^(auto^|pip^)
    echo   --distro    Distribution to test ^(ubuntu^|centos^|all^)
    echo   --help      Show this help message
    exit /b 0
)
echo Unknown option: %~1
echo Use --help for usage information
exit /b 1

:end_parse

echo Test configuration:
echo   Method: %METHOD%
echo   Distribution: %DISTRO%
echo.

REM Run tests
if "%DISTRO%"=="all" goto test_ubuntu
if "%DISTRO%"=="ubuntu" goto test_ubuntu
if "%DISTRO%"=="centos" goto test_centos
goto end_tests

:test_ubuntu
echo 🐳 Testing in ubuntu container...
echo Building ubuntu test image...
docker build -f test/Dockerfile.ubuntu -t libuipc-test-ubuntu .
if %errorlevel% neq 0 exit /b %errorlevel%

echo Running installation test...
docker run --rm --gpus all -v %cd%:/workspace/source libuipc-test-ubuntu python3 test_installation.py --method %METHOD%
if %errorlevel% neq 0 exit /b %errorlevel%

echo ✅ ubuntu test completed
echo.

if "%DISTRO%"=="ubuntu" goto end_tests

:test_centos
echo 🐳 Testing in centos container...
echo Building centos test image...
docker build -f test/Dockerfile.centos -t libuipc-test-centos .
if %errorlevel% neq 0 exit /b %errorlevel%

echo Running installation test...
docker run --rm --gpus all -v %cd%:/workspace/source libuipc-test-centos python3 test_installation.py --method %METHOD%
if %errorlevel% neq 0 exit /b %errorlevel%

echo ✅ centos test completed
echo.

:end_tests
echo 🎉 All tests completed successfully!