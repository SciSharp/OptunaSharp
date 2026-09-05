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

rem -v q + -p:WarningLevel=0 silence the build's warning noise globally (both flags `dotnet run` honors;
rem it rejects -clp/--tl). errors still surface; the example's own colored stdout is unaffected.
for %%E in (OptunaHubCode1 OptunaHubBenchmark OptunaHubSamplerShootout OptunaHubMultiObjective OptunaHubPruning OptunaHubLocalRegistry OptunaHubDashboard) do (
  echo.
  echo ============ %%E ============
  dotnet run -v q -p:WarningLevel=0 examples/optunahub/%%E.cs
  if errorlevel 1 echo   ^(%%E exited with an error - see the output above^)
)

rem ── optuna-dashboard: open the study OptunaHubDashboard.cs just persisted, live ────────────────────
rem the .db was written by OptunaSharp.Storage.Sqlite in Optuna's exact schema, so the real Python
rem optuna-dashboard opens it directly. we launch it in its OWN window (+ your browser) so THIS window
rem stays open with the suite output above.
echo.
if exist "examples\optunahub\optunahub-dashboard.db" (
  echo ============ optuna-dashboard ============
  echo   Opening the persisted study in optuna-dashboard ^(a new window^) and your browser ...
  start "OptunaSharp dashboard" "%PYTHON%" examples/optunahub/dashboard.py
) else (
  echo   ^(no study database at examples\optunahub\optunahub-dashboard.db - skipping the dashboard^)
)

:done
echo.
echo ============================================================
echo   Done. This window stays open so you can read the output above.
echo   ^(Close it, or press a key.^)
pause >nul
endlocal
