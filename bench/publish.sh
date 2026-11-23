#!/usr/bin/env bash
# Builds the benchmark harness for a Linux machine: every C# probe published Release, framework-dependent, linux-x64, into ONE
# folder (out/; the probes share OptunaSharp, NumSharp and the OpenBLAS native, so one copy serves all eight apphosts), with
# the Python probes, the runner, the monitor and the parser beside it, and a SOURCE file naming the commit the probes were
# built from (the code the numbers describe). Then packs it all as bench/harness-stage.tar for bench/setup.sh.
#
# Runs wherever the .NET 10 SDK does (Git Bash on Windows included): publishing for linux-x64 from Windows is a cross-target
# of managed code, and the native files come from the NuGet packages' linux-x64 assets.
# Usage: bash bench/publish.sh [stage folder, default bench/stage]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/.." && pwd)
stage=${1:-$here/stage}

# A fresh stage, so no file from an earlier build (a probe since renamed, an older NumSharp) can end up on the worker.
rm -rf "${stage:?}"
mkdir -p "$stage/out" "$stage/py"
for p in perf wide scale proof njobs pool execprobe modeprobe; do
    dotnet publish "$here/probes/$p/$p.csproj" -c Release -r linux-x64 --self-contained false -o "$stage/out" -nologo -v:q
done
cp "$here"/py/*.py "$stage/py/"
cp "$here/monitor.py" "$here/parse.py" "$here/run_worker.sh" "$here/setup.sh" "$stage/"
# The commit of src/ the probes were built from, and whether the working tree differed from it (it is printed in every run's
# results.txt header). "-dirty" covers any change in the checkout, bench/ included; src/ is what the numbers describe.
src_state=$(git -C "$repo" status --porcelain -- src | wc -l | tr -d ' ')
echo "$(git -C "$repo" rev-parse --short HEAD) (src/ changes not committed: $src_state)" > "$stage/SOURCE"
tar -C "$stage" -cf "$here/harness-stage.tar" .
echo "staged $(find "$stage/out" -maxdepth 1 -type f | wc -l) files in out/, source $(cat "$stage/SOURCE"): $here/harness-stage.tar"
