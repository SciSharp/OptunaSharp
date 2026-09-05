@echo off
rem Launch the colorful OptunaHub x OptunaSharp examples on Windows.
rem
rem You can DOUBLE-CLICK this file in Explorer, or run it from a terminal. It installs the Python
rem prerequisites (into the interpreter OptunaSharp.Interop.pythonnet embeds), then runs both examples:
rem   examples\optunahub\OptunaHubBenchmark.cs        - the paper's Code 1: load benchmarks/bbob live
rem   examples\optunahub\OptunaHubSamplerShootout.cs  - race real hub samplers vs OptunaSharp's native TPE
rem
rem The window STAYS OPEN at the end (and on any error) so you can read the output.
rem
rem Usage:   double-click, or  examples\optunahub\run-optunahub.bat
rem   set PYTHON=C:\path\to\python.exe  &  examples\optunahub\run-optunahub.bat   (choose the interpreter)
rem   set NO_COLOR=1                    &  examples\optunahub\run-optunahub.bat   (disable colors)
setlocal
title OptunaHub x OptunaSharp examples
chcp 65001 >nul 2>&1

rem repo root = two levels up from this script (examples\optunahub\), so `dotnet run examples\..` resolves
cd /d "%~dp0..\.."

rem pick the Python interpreter (honored by the bridge via %PYTHON% when discovering libpython)
if not defined PYTHON set "PYTHON=python"
"%PYTHON%" --version >nul 2>&1
if errorlevel 1 (
  echo No python found at "%PYTHON%". Install Python 3.9+ or set PYTHON to your interpreter, then re-run.
  goto :done
)

for /f "delims=" %%v in ('"%PYTHON%" --version 2^>^&1') do echo Python : %%v
echo Installing prerequisites from examples/optunahub/requirements.txt ...
"%PYTHON%" -m pip install --quiet -r examples/optunahub/requirements.txt
if errorlevel 1 (
  echo.
  echo pip install failed. Install manually, then re-run:
  echo     "%PYTHON%" -m pip install -r examples/optunahub/requirements.txt
  goto :done
)

if not defined FORCE_COLOR set "FORCE_COLOR=1"
if not defined OPTUNASHARP_OPTUNAHUB_QUIET set "OPTUNASHARP_OPTUNAHUB_QUIET=1"
rem clear any stale PYTHONNET_PYDLL so the bridge discovers %PYTHON% (where we just installed the
rem prerequisites), not a previously-pinned interpreter. scoped to this window by setlocal.
set "PYTHONNET_PYDLL="

for %%E in (OptunaHubCode1 OptunaHubBenchmark OptunaHubSamplerShootout OptunaHubMultiObjective OptunaHubPruning OptunaHubLocalRegistry) do (
  echo.
  echo ============ %%E ============
  dotnet run examples/optunahub/%%E.cs
  if errorlevel 1 echo   ^(%%E exited with an error - see the output above^)
)

:done
echo.
echo ============================================================
echo   Done. This window stays open so you can read the output above.
echo   ^(Close it, or press a key.^)
pause >nul
endlocal
