#!/usr/bin/env bash
#
# TEMPORARY bridge — remove when NumSharp 0.70.0 ships to nuget.org.
# ---------------------------------------------------------------------------------------------------
# OptunaSharp.Core is written against a NumSharp that is AHEAD of the published NumSharp 0.60.0: it uses
# `np.linalg`, `NumSharp.RuntimeError`, `NumSharp.KeyError` (in NumSharp.Core) and the whole
# `NumSharp.Interop.OpenBLAS` package — none of which the nuget.org 0.60.0 has (OpenBLAS isn't published
# at all). So in a clean container / CI runner (no sibling checkout) Core cannot restore OR compile.
#
# This packs BOTH packages from a sibling NumSharp checkout into a local feed. Consumers add that feed
# BEFORE nuget.org (so the locally-built 0.60.0 shadows the stale nuget 0.60.0) and everything restores,
# compiles and runs — verified against OptunaSharp.Tests (scaffolding + numeric smoke, net8 + net10).
#
# When NumSharp 0.70.0 is released with these APIs + OpenBLAS on nuget.org: delete this script, drop the
# feed from the CI NuGet.config, and bump Core's NumSharp/OpenBLAS PackageReference to 0.70.0.
#
# Usage:  pack-temp-numsharp-feed.sh <numsharp-repo-root> <out-feed-dir>
# Env:    NUMSHARP_PACK_ARGS  extra args to `dotnet pack` (e.g. "-p:SkipNDScopeWeave=true" for a faster,
#                             weaver-skipped pack — passes the smoke tests, but omit it for full runtime
#                             fidelity of NumSharp's NDArray disposal weaving).
set -euo pipefail

NUMSHARP="${1:?usage: pack-temp-numsharp-feed.sh <numsharp-repo-root> <out-feed-dir>}"
FEED="${2:?usage: pack-temp-numsharp-feed.sh <numsharp-repo-root> <out-feed-dir>}"
mkdir -p "$FEED"

for proj in \
  "$NUMSHARP/src/NumSharp.Core/NumSharp.Core.csproj" \
  "$NUMSHARP/src/NumSharp.Interop.OpenBLAS/NumSharp.Interop.OpenBLAS.csproj" ; do
  echo ">> packing $(basename "$proj")"
  dotnet pack "$proj" -c Release -o "$FEED" \
    -p:GeneratePackageOnBuild=false ${NUMSHARP_PACK_ARGS:-} -v q --nologo -p:WarningLevel=0
done

echo ">> temp feed contents:"
ls -1 "$FEED"
