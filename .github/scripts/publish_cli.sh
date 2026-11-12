#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------
# Publishes the PORTABLE `optunasharp` command-line tool for ONE runtime identifier (RID): a single
# self-contained executable, checked, then packed as the archive a GitHub Release ships.
#
# WHY IT EXISTS. The `OptunaSharp.Cli` NuGet package is a `dotnet tool`: installing it needs the .NET
# SDK. This binary needs nothing: no .NET installation, no installer. A user extracts one file into a
# folder on PATH and runs `optunasharp`. The release used to ship only the NuGet form.
#
#   bash .github/scripts/publish_cli.sh <output-dir> <version> <rid> [<dotnet publish option>...]
#
#   <output-dir>   Receives two things:
#                    <rid>/optunasharp[.exe]     the binary, named as it is typed on a command line;
#                    optunasharp-<rid>.zip       (win-*) or
#                    optunasharp-<rid>.tar.gz    (every other RID) the release asset, holding only that
#                                                binary, so extracting it into a PATH folder adds
#                                                exactly one file there.
#                  The asset name carries no version, so the address
#                  .../releases/latest/download/optunasharp-<rid>.<ext> always serves the newest release.
#   <version>      SemVer version stamped into the binary, e.g. 1.2.3, 1.2.3-rc.1, 0.1.0-ci.42. It is
#                  what `optunasharp --version` prints. AssemblyVersion/FileVersion are this version
#                  without its prerelease label, as in pack_all.sh.
#   <rid>          One of: linux-x64 linux-arm64 linux-musl-x64 linux-musl-arm64 osx-x64 osx-arm64
#                  win-x64 win-arm64. These are the RIDs NumSharp.Interop.OpenBLAS ships a native
#                  library for; a binary for any other RID would start, then fail in the first sampler
#                  that needs LAPACK (CMA-ES, GP, NSGA-III), so other RIDs are refused here.
#   <option>...    Passed to `dotnet publish` unchanged, e.g. `--artifacts-path <dir>` to keep bin/ and
#                  obj/ out of the working tree.
#
# Run it from the repository root. Any host can publish any RID (the SDK downloads that RID's runtime
# pack), but only a host of the same OS and architecture can run the result: smoke_cli.sh does that.
# -----------------------------------------------------------------------------------------------
set -euo pipefail

# Reports a failure the way GitHub Actions annotates it, and stops.
fail() {
  echo "::error::publish_cli.sh: $*" >&2
  exit 1
}

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <output-dir> <version> <rid> [<dotnet publish option>...]" >&2
  exit 2
fi
out=$1
version=$2
rid=$3
shift 3
assembly_version=${version%%-*}

case "$rid" in
  linux-x64|linux-arm64|linux-musl-x64|linux-musl-arm64|osx-x64|osx-arm64|win-x64|win-arm64) ;;
  *) fail "unsupported RID '$rid'. NumSharp.Interop.OpenBLAS has no native library for it, so the binary could not run CMA-ES, GP or NSGA-III." ;;
esac

project=src/OptunaSharp.Cli/OptunaSharp.Cli.csproj
[[ -f "$project" ]] || fail "$project not found; run it from the repository root."

# Windows names an executable by its extension and its users expect a .zip. Everywhere else the file
# has no extension, and a .tar.gz keeps the executable bit, which a .zip extracted by most tools loses.
case "$rid" in
  win-*) binary=optunasharp.exe; published=OptunaSharp.Cli.exe; archive="optunasharp-$rid.zip" ;;
  *)     binary=optunasharp;     published=OptunaSharp.Cli;     archive="optunasharp-$rid.tar.gz" ;;
esac

mkdir -p "$out"
out=$(cd "$out" && pwd)
stage="$out/$rid"
# A file left by an earlier run would pass the "exactly one file" check below for the wrong reason,
# or fail it for no reason. The path is built from this script's own arguments, never from a glob.
rm -rf "${stage:?}"
rm -f "${out:?}/$archive"

echo "::group::publish optunasharp $version for $rid"
# Every property is GLOBAL, so it also reaches the projects the CLI references (Core and the two
# storage packages): they are built in the same configuration and their symbols are embedded too.
#   SelfContained + PublishSingleFile        the .NET runtime, the framework and every assembly in ONE
#                                            executable. This is what makes it portable. SelfContained
#                                            is given as a property, not as `--self-contained`: with
#                                            PublishSingleFile, SDK 10.0.101 evaluates
#                                            `--self-contained false` as true, so the switch is not
#                                            what decides here.
#   IncludeNativeLibrariesForSelfExtract     OpenBLAS (np.linalg) and SQLite are native libraries. The
#                                            operating system can only load them from a file, so they
#                                            travel inside the executable and are written to a per-user
#                                            folder on the first run. Without this they would be loose
#                                            files beside the executable, and it would no longer be one
#                                            file.
#   PublishReadyToRun                        Compiles the assemblies to native code ahead of time. Each
#                                            `optunasharp` command is a new process and its start is
#                                            almost all JIT. Results are bit-identical to JIT-compiled
#                                            code; only the start-up time changes.
#   EnableCompressionInSingleFile=false      A compressed bundle is decompressed into memory on EVERY
#                                            start. Measured on win-x64, `ask` + `tell` on a SQLite
#                                            study with TPE, medians of 15 interleaved cycles:
#                                              the installed .NET tool (JIT)       1,061 ms
#                                              JIT, compressed           50 MB     1,564 ms
#                                              JIT, uncompressed        107 MB     1,263 ms
#                                              ReadyToRun, compressed    68 MB     1,585 ms  (gain lost)
#                                              ReadyToRun, uncompressed 152 MB       780 ms  (this build)
#                                            The archive compresses the download instead, where it
#                                            costs nothing at run time.
#   DebugType=embedded                       Symbols live inside the assemblies, so `--debug` tracebacks
#                                            keep their line numbers and no .pdb lies beside the binary.
#   GenerateDocumentationFile=false          The XML documentation file would be a second loose file.
#   Version / AssemblyVersion / FileVersion  One version for the whole build, as pack_all.sh stamps the
#                                            packages.
#   ContinuousIntegrationBuild=true          Deterministic build: the same commit gives the same bytes.
# The CLI project itself already switches trimming and Native AOT off (`ask --sampler <Name>` builds
# samplers by reflection) and TieredPGO off (short-lived processes); those settings apply here too.
dotnet publish "$project" --configuration Release --framework net10.0 \
  --runtime "$rid" --output "$stage" \
  -p:SelfContained=true \
  -p:PublishSingleFile=true \
  -p:IncludeNativeLibrariesForSelfExtract=true \
  -p:PublishReadyToRun=true \
  -p:EnableCompressionInSingleFile=false \
  -p:DebugType=embedded \
  -p:GenerateDocumentationFile=false \
  -p:Version="$version" \
  -p:AssemblyVersion="$assembly_version" \
  -p:FileVersion="$assembly_version" \
  -p:ContinuousIntegrationBuild=true \
  "$@"
echo "::endgroup::"

# ONE file, nothing else. A second file means something was not bundled (a new native dependency, a
# content file, a .pdb), and a user who copies only the executable would get a broken tool.
shopt -s nullglob dotglob
files=("$stage"/*)
shopt -u nullglob dotglob
if (( ${#files[@]} != 1 )) || [[ "${files[0]}" != "$stage/$published" ]]; then
  ls -la "$stage" >&2 || true
  fail "expected the publish output to be the single file $published (listing above)."
fi

# The name is changed on the file, not through AssemblyName: Core grants InternalsVisibleTo to the
# assembly name `OptunaSharp.Cli`, so renaming the assembly would break the build. The bundle does not
# depend on the name of the file that holds it.
mv "$stage/$published" "$stage/$binary"
chmod 755 "$stage/$binary"

# Names the RID a native executable was built for, from its header: `win-x64`, `linux-musl-arm64`, ...
# or `unknown`. The multi-byte fields are read in the host's byte order; every host this runs on (x64,
# arm64) is little-endian, as are all three file formats.
#   $1  the executable.
describe_binary() {
  local file=$1 magic os arch
  magic=$(od -An -tx1 -N4 "$file" | tr -d ' \n')
  case "$magic" in
    4d5a*)
      # PE: the 32-bit field at 0x3C points to the "PE\0\0" signature; the COFF Machine field follows it.
      local pe
      pe=$(od -An -tu4 -j60 -N4 "$file" | tr -d ' \n')
      os=win
      case "$(od -An -tx2 -j$((pe + 4)) -N2 "$file" | tr -d ' \n')" in
        8664) arch=x64 ;;
        aa64) arch=arm64 ;;
        *) arch= ;;
      esac
      ;;
    7f454c46)
      # ELF: e_machine is the 16-bit field at offset 18. The C library is named by the program
      # interpreter, a path near the start of the file (ld-linux-* for glibc, ld-musl-* for musl).
      case "$(od -An -tu2 -j18 -N2 "$file" | tr -d ' \n')" in
        62) arch=x64 ;;
        183) arch=arm64 ;;
        *) arch= ;;
      esac
      # `|| true`: grep stops at the first match and the commands before it then die of SIGPIPE, which
      # `pipefail` reports as a failure although the match was found and printed. No match also lands
      # here, with an empty result.
      local interpreter
      interpreter=$(head -c 65536 "$file" | LC_ALL=C grep -a -o -m1 -E '/lib(64)?/ld-[A-Za-z0-9_.-]+' || true)
      case "$interpreter" in
        */ld-musl-*) os=linux-musl ;;
        */ld-linux-*) os=linux ;;
        *) os= ;;
      esac
      ;;
    cffaedfe)
      # Mach-O, 64-bit: cputype is the 32-bit field at offset 4 (0x01000007 x86_64, 0x0100000C arm64).
      os=osx
      case "$(od -An -tu4 -j4 -N4 "$file" | tr -d ' \n')" in
        16777223) arch=x64 ;;
        16777228) arch=arm64 ;;
        *) arch= ;;
      esac
      ;;
    *) os=; arch= ;;
  esac
  if [[ -n "$os" && -n "$arch" ]]; then echo "$os-$arch"; else echo unknown; fi
}

# The header must name the requested RID. `dotnet publish` without a usable RID does not fail: it
# builds for the HOST, and an arm64 Windows or macOS machine then runs the x64 result under emulation,
# so a smoke test alone would pass on a mislabelled binary.
actual=$(describe_binary "$stage/$binary")
[[ "$actual" == "$rid" ]] || fail "$stage/$binary is a $actual executable, not $rid."

echo "::group::pack $archive"
case "$archive" in
  *.zip)
    # Git for Windows has no `zip`. Windows itself ships bsdtar (System32\tar.exe, Windows 10 1803 and
    # later), which writes a .zip when asked for that format; GNU tar, the `tar` Git Bash finds first,
    # cannot. The Windows program needs Windows-style paths.
    if [[ -n "${SYSTEMROOT:-}" && -x "$(cygpath -u "$SYSTEMROOT")/System32/tar.exe" ]]; then
      "$(cygpath -u "$SYSTEMROOT")/System32/tar.exe" -c --format zip \
        -f "$(cygpath -w "$out/$archive")" -C "$(cygpath -w "$stage")" "$binary"
    elif command -v zip >/dev/null 2>&1; then
      (cd "$stage" && zip -q -9 -X "$out/$archive" "$binary")
    else
      fail "no tool to write $archive: neither Windows' System32/tar.exe nor zip is available."
    fi
    ;;
  *.tar.gz)
    # The entry must be executable for whoever extracts it. On a Windows host the file system has no
    # executable bit, so GNU tar is told the mode; bsdtar (macOS) has no such option and needs none,
    # because chmod above took effect there. Owner 0 keeps the build machine's user out of the archive.
    if tar --version 2>/dev/null | grep -q 'GNU tar'; then
      tar -C "$stage" --mode=0755 --owner=0 --group=0 --numeric-owner -czf "$out/$archive" "$binary"
    else
      tar -C "$stage" --uid 0 --gid 0 -czf "$out/$archive" "$binary"
    fi
    ;;
esac
echo "::endgroup::"

echo "Published $rid:"
ls -l "$stage/$binary" "$out/$archive"
