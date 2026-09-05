#!/usr/bin/env bash
# Launch the colorful OptunaHub × OptunaSharp examples.
#
# Installs the Python prerequisites (into the interpreter OptunaSharp.Interop.pythonnet will embed),
# then runs both examples with `dotnet run`:
#   examples/OptunaHubBenchmark.cs        — the paper's Code 1: load benchmarks/bbob, optimize live
#   examples/OptunaHubSamplerShootout.cs  — race real hub samplers vs OptunaSharp's native TPE
#
# Usage:   examples/optunahub/run-optunahub.sh
#   PYTHON=/path/to/python  examples/optunahub/run-optunahub.sh     # choose the interpreter
#   NO_COLOR=1              examples/optunahub/run-optunahub.sh      # disable colors
set -euo pipefail

# repo root = two levels up from this script (examples/optunahub/) so `dotnet run examples/..` resolves
cd "$(dirname "$0")/../.."

# pick the Python interpreter (honored by the bridge via $PYTHON when discovering libpython)
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if   command -v python3 >/dev/null 2>&1; then PY=python3
  elif command -v python  >/dev/null 2>&1; then PY=python
  else echo "No python found on PATH. Install Python 3.9+ and re-run."; exit 1; fi
fi
export PYTHON="$PY"
unset PYTHONNET_PYDLL   # discover $PYTHON (where we install the deps), not a previously-pinned interpreter

echo "Python : $("$PY" --version 2>&1)"
echo "Installing prerequisites from examples/optunahub/requirements.txt …"
if ! "$PY" -m pip install --quiet -r examples/optunahub/requirements.txt; then
  echo "pip install failed. Install manually:"
  echo "    $PY -m pip install -r examples/optunahub/requirements.txt"
  exit 1
fi

export FORCE_COLOR="${FORCE_COLOR:-1}"                              # force ANSI colors on
export OPTUNASHARP_OPTUNAHUB_QUIET="${OPTUNASHARP_OPTUNAHUB_QUIET:-1}"

run() { echo; echo "════════════ $1 ════════════"; dotnet run "examples/optunahub/$2.cs"; }
run "Code 1 — a hub sampler + a hub benchmark, together" OptunaHubCode1
run "Benchmark — BBOB Sphere, live"                       OptunaHubBenchmark
run "Sampler shootout — hub samplers vs native TPE"       OptunaHubSamplerShootout
run "Multi-objective — bbob_biobj + native NSGA-II"       OptunaHubMultiObjective
run "Pruner — a hub MedianPruner cutting laggards"        OptunaHubPruning
run "Local registry + mixed Float/Int/Categorical space"  OptunaHubLocalRegistry
run "Dashboard — persist a hub-sampled study to SQLite"   OptunaHubDashboard

# ── optuna-dashboard: open the study OptunaHubDashboard.cs just persisted, live ─────────────────────
# the .db was written by OptunaSharp.Storage.Sqlite in Optuna's exact schema, so the real Python
# optuna-dashboard opens it directly. this BLOCKS (serves + opens your browser) until you Ctrl+C.
DB="examples/optunahub/optunahub-dashboard.db"
echo
if [ -f "$DB" ]; then
  echo "════════════ optuna-dashboard — the persisted study, live (Ctrl+C to stop) ════════════"
  "$PY" examples/optunahub/dashboard.py || true          # || true: Ctrl+C returns non-zero under `set -e`
else
  echo "(no study database at $DB — skipping the dashboard; OptunaHubDashboard.cs did not persist one)"
fi
