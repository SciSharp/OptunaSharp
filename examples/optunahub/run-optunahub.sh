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

echo "Python : $("$PY" --version 2>&1)"
echo "Installing prerequisites (optuna optunahub coco-experiment cmaes pandas) …"
if ! "$PY" -m pip install --quiet optuna optunahub coco-experiment cmaes pandas; then
  echo "pip install failed. Install manually:"
  echo "    $PY -m pip install optuna optunahub coco-experiment cmaes pandas"
  exit 1
fi

export FORCE_COLOR="${FORCE_COLOR:-1}"                              # force ANSI colors on
export OPTUNASHARP_OPTUNAHUB_QUIET="${OPTUNASHARP_OPTUNAHUB_QUIET:-1}"

echo
echo "════════════ OptunaHub benchmark (BBOB Sphere) ════════════"
dotnet run examples/optunahub/OptunaHubBenchmark.cs

echo
echo "════════════ OptunaHub sampler shootout ════════════"
dotnet run examples/optunahub/OptunaHubSamplerShootout.cs
