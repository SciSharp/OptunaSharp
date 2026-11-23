#!/usr/bin/env bash
# Setup of the OptunaSharp README benchmark on a Linux machine (written for the worker machine, Proxmox host `worker`; first
# used 2026-11-22, the harness folder added 2026-11-23).
#
# Everything lives in one disposable folder (BENCH; on the worker, drive 2, the worker's rule 10: data that may be lost), so
# removing BENCH undoes it completely: no apt package, no service, no change to the host outside that folder. Nothing is
# downloaded here (the worker's rule 22 caps downloads at 7 MB/s): the workstation fetched every file and copied it in through
# the link with a bandwidth limit, into $BENCH/incoming:
#   - dotnet-runtime-10.0.15-linux-x64.tar.gz   (.NET runtime for the framework-dependent probe builds)
#   - cpython-3.12.16-linux-x64.tar.gz          (python-build-standalone, install_only_stripped)
#   - wheels/                                   (optuna 5.0.0, rustuna 0.3.1, numpy 2.5.4, torch 2.12.1+cpu, ...)
#   - harness-stage.tar                         (bench/publish.sh: the probes in out/, py/, the runner, monitor, parser)
# The runtime and the venv are built once and kept (FORCE=1 rebuilds them); the harness is unpacked into its own folder
# (HARNESS, default $BENCH/v2), so a new harness never mixes files with an older one and the older results stay where they are.
# Idempotent: a second run refreshes the harness, keeping results/.
set -euo pipefail
BENCH=${BENCH:-/mnt/drive2/data/optunasharp-bench}
HARNESS=${HARNESS:-$BENCH/v2}
IN=$BENCH/incoming
cd "$BENCH"

# .NET runtime: extracted, not installed; the runner points DOTNET_ROOT at it.
if [ ! -x dotnet/dotnet ] || [ "${FORCE:-0}" = 1 ]; then
    rm -rf dotnet && mkdir dotnet && tar -xzf "$IN/dotnet-runtime-10.0.15-linux-x64.tar.gz" -C dotnet
fi
./dotnet/dotnet --list-runtimes

# CPython 3.12 (standalone build, the same minor version as the workstation's 2026-11-16 run) and a venv with the pinned wheels
# only (--no-index: pip must not reach the internet).
if [ ! -x venv/bin/python ] || [ "${FORCE:-0}" = 1 ]; then
    rm -rf python venv && tar -xzf "$IN/cpython-3.12.16-linux-x64.tar.gz"
    ./python/bin/python3 -m venv venv
    ./venv/bin/python -m pip install -q --no-index --find-links "$IN/wheels" \
        optuna==5.0.0 rustuna==0.3.1 numpy==2.5.4 scipy==1.16.3 cmaes==0.13.1 psutil==7.2.1 "torch==2.12.1+cpu"
fi
./venv/bin/python -I -c "import sys, optuna, rustuna, numpy, scipy, torch, cmaes; print('python', sys.version.split()[0], 'optuna', optuna.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'torch', torch.__version__, 'cmaes', cmaes.__version__)"
./venv/bin/python -I -m pip show rustuna | grep -i '^version'

# The harness: a clean folder, then the stage.
rm -rf "${HARNESS:?}" && mkdir -p "$HARNESS" && tar -xf "$IN/harness-stage.tar" -C "$HARNESS"
# The apphosts lose their executable bit in a tar written on Windows.
for f in "$HARNESS"/out/*; do
    case "$f" in *.* ) ;; *) chmod +x "$f" ;; esac
done
mkdir -p results
echo "SETUP DONE: harness in $HARNESS, source $(cat "$HARNESS/SOURCE")"
