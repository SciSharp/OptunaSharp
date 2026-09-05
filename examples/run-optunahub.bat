@echo off
rem Launch the colorful OptunaHub x OptunaSharp examples on Windows.
rem
rem Installs the Python prerequisites (into the interpreter OptunaSharp.Interop.pythonnet will embed),
rem then runs both examples with `dotnet run`:
rem   examples\OptunaHubBenchmark.cs        - the paper's Code 1: load benchmarks/bbob, optimize live
rem   examples\OptunaHubSamplerShootout.cs  - race real hub samplers vs OptunaSharp's native TPE
rem
rem Usage:   examples\run-optunahub.bat
rem   set PYTHON=C:\path\to\python.exe  &  examples\run-optunahub.bat   (choose the interpreter)
rem   set NO_COLOR=1                    &  examples\run-optunahub.bat   (disable colors)
setlocal

rem repo root = the parent of this script's directory
cd /d "%~dp0.."

rem pick the Python interpreter (honored by the bridge via %PYTHON% when discovering libpython)
if not defined PYTHON set "PYTHON=python"
"%PYTHON%" --version >nul 2>&1
if errorlevel 1 (
  echo No python found at "%PYTHON%". Install Python 3.9+ or set PYTHON to your interpreter, then re-run.
  exit /b 1
)

for /f "delims=" %%v in ('"%PYTHON%" --version 2^>^&1') do echo Python : %%v
echo Installing prerequisites (optuna optunahub coco-experiment cmaes pandas) ...
"%PYTHON%" -m pip install --quiet optuna optunahub coco-experiment cmaes pandas
if errorlevel 1 (
  echo pip install failed. Install manually:
  echo     "%PYTHON%" -m pip install optuna optunahub coco-experiment cmaes pandas
  exit /b 1
)

if not defined FORCE_COLOR set "FORCE_COLOR=1"
if not defined OPTUNASHARP_OPTUNAHUB_QUIET set "OPTUNASHARP_OPTUNAHUB_QUIET=1"

echo.
echo ============ OptunaHub benchmark (BBOB Sphere) ============
dotnet run examples/OptunaHubBenchmark.cs

echo.
echo ============ OptunaHub sampler shootout ============
dotnet run examples/OptunaHubSamplerShootout.cs

endlocal
