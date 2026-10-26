#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------
# Packs EVERY shipping project (src/*/*.csproj) into one feed, all stamped with ONE version.
#
# WHY ONE SCRIPT. The release once packed only src/OptunaSharp.Core, so a v* tag would have published
# 1 of the 20 packages. Packing the whole family here, the same way on every push, every PR and the
# release, means the release path is exercised long before a tag exists. verify_packages.cs then
# fails the run if any packable project is missing from the feed.
#
#   bash .github/scripts/pack_all.sh <output-dir> <package-version> [<assembly-version>]
#
#   <package-version>   NuGet/SemVer version for every package, e.g. 1.2.3, 1.2.3-rc.1, 0.1.0-ci.42.
#                       Also becomes AssemblyInformationalVersion, which OptunaSharpVersion.Version
#                       reports at run time.
#   <assembly-version>  Numeric AssemblyVersion/FileVersion (default: <package-version> without its
#                       prerelease label). Strong-named assemblies bind on it.
#
# Run it from the repository root, inside a git checkout: SourceLink reads the commit and the
# `origin` remote from .git, and the PDBs and the nuspec <repository commit> record them.
# -----------------------------------------------------------------------------------------------
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <output-dir> <package-version> [<assembly-version>]" >&2
  exit 2
fi
out=$1
version=$2
assembly_version=${3:-${version%%-*}}

# Packing outside a git work tree would silently drop SourceLink (Microsoft.Build.Tasks.Git warns, it
# does not fail), and the package gate would then reject every PDB. Fail here with the real cause.
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "::error::pack_all.sh must run inside a git checkout (SourceLink and <repository commit> come from it)." >&2
  exit 1
fi

mkdir -p "$out"
shopt -s nullglob
projects=(src/*/*.csproj)
if (( ${#projects[@]} == 0 )); then
  echo "::error::pack_all.sh: no src/*/*.csproj found; run it from the repository root." >&2
  exit 1
fi

echo "Packing ${#projects[@]} projects at version $version (assembly $assembly_version) into $out"
for proj in "${projects[@]}"; do
  echo "::group::pack $proj"
  # Every property is GLOBAL, so it also flows into each ProjectReference. A satellite and the Core it
  # references are therefore built in the same configuration. The first pack compiles Core; later
  # packs find it up to date and reuse it, so the family is built once, not twenty times.
  #   ContinuousIntegrationBuild=true -> deterministic build; PDBs record /_/-rooted paths instead of
  #                                       this machine's checkout path (verify_packages.cs checks it).
  #   No -p:NoWarn override: each project already suppresses its own doc-comment warnings, and a
  #   global NoWarn would replace (not extend) the SDK's and the projects' lists.
  dotnet pack "$proj" --configuration Release --output "$out" \
    -p:Version="$version" \
    -p:PackageVersion="$version" \
    -p:AssemblyVersion="$assembly_version" \
    -p:FileVersion="$assembly_version" \
    -p:ContinuousIntegrationBuild=true
  echo "::endgroup::"
done

echo "Packed:"
ls -1 "$out"/*.nupkg
