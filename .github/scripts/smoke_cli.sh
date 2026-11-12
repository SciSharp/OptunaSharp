#!/bin/sh
# -----------------------------------------------------------------------------------------------
# Smoke test of the PORTABLE `optunasharp` binary, the way its user meets it: extract the release
# archive, run the one file it holds, with no .NET anywhere in reach.
#
#   sh .github/scripts/smoke_cli.sh <archive-or-binary> <expected-version>
#
#   <archive-or-binary>  optunasharp-<rid>.zip / optunasharp-<rid>.tar.gz as publish_cli.sh wrote it
#                        (extracted first, which also tests the archive), or a bare binary.
#   <expected-version>   what `optunasharp --version` must print: the version given to publish_cli.sh.
#
# WHAT A PASS PROVES, each point being a way a portable build can be broken while its publish succeeds:
#   1. The archive holds exactly one file, named optunasharp[.exe], executable after extraction.
#   2. The binary is self-contained. DOTNET_ROOT* point at an empty folder, so a binary that needed an
#      installed .NET runtime would stop with "You must install .NET". (CI hosts have .NET installed,
#      so without this a framework-dependent build would pass there.)
#   3. It is stamped with the release version.
#   4. Both bundled native libraries are written out on the first run and load: SQLite (the study is
#      stored through `sqlite:///`) and OpenBLAS (every CMA-ES `ask` after the first trial runs LAPACK's
#      eigendecomposition; without OpenBLAS it throws OpenBlasMissingBackendException).
#   5. Several processes share one study: nine `ask` + `tell` pairs, each command a new process.
# It does NOT compare parameter values with Optuna's: that is the parity suites' job, and the last bits
# of a CMA-ES draw depend on the CPU's OpenBLAS kernel.
#
# POSIX sh ON PURPOSE, no bash. The same file runs in Git Bash on Windows, as bash 3.2 or zsh's sh on
# macOS, as dash on Debian and as BusyBox ash inside a bare Alpine container, where there is no bash.
# -----------------------------------------------------------------------------------------------
set -eu

# Reports a failure the way GitHub Actions annotates it, and stops. The EXIT trap then cleans up.
fail() {
  echo "::error::smoke_cli.sh: $*" >&2
  exit 1
}

if [ $# -ne 2 ]; then
  echo "usage: $0 <archive-or-binary> <expected-version>" >&2
  exit 2
fi
input=$1
expected=$2
[ -f "$input" ] || fail "$input does not exist."
# The test changes directory below, so a relative path must be anchored first.
case "$input" in
  /* | [A-Za-z]:*) ;;
  *) input="$(pwd)/$input" ;;
esac

work=$(mktemp -d)
# Git Bash names the folder /tmp/...; the binary is a Windows program and reads the variables set
# below as Windows paths, so convert once (C:/Users/..., which both sides accept).
if command -v cygpath >/dev/null 2>&1; then
  work=$(cygpath -m "$work")
fi
# Leave the folder before deleting it: Windows refuses to delete a process's current directory.
trap 'cd / && rm -rf "${work:?}" 2>/dev/null || true' EXIT

case "$input" in
  *.tar.gz | *.zip)
    mkdir "$work/bin"
    case "$input" in
      *.tar.gz) tar -xzf "$input" -C "$work/bin" ;;
      *) unzip -q "$input" -d "$work/bin" ;;
    esac
    # Point 1. The positional parameters are reused as the list of extracted files.
    set -- "$work/bin"/*
    [ $# -eq 1 ] || fail "the archive must hold exactly one file, found $#: $*"
    binary=$1
    case "$(basename "$binary")" in
      optunasharp | optunasharp.exe) ;;
      *) fail "the archive holds '$(basename "$binary")', not optunasharp[.exe]." ;;
    esac
    [ -x "$binary" ] || fail "$(basename "$binary") is not executable after extraction."
    ;;
  *)
    binary=$input
    ;;
esac

# Point 2: no installed .NET may help. These are the variables an application host reads to find a
# runtime; a self-contained single-file binary never reads them. The folder must EXIST: the host
# ignores a DOTNET_ROOT that names a missing folder and goes on to the machine's installation (a
# framework-dependent build passed this test until the folder was created).
mkdir "$work/no-dotnet-here"
DOTNET_ROOT="$work/no-dotnet-here"
DOTNET_ROOT_X64=$DOTNET_ROOT
DOTNET_ROOT_ARM64=$DOTNET_ROOT
export DOTNET_ROOT DOTNET_ROOT_X64 DOTNET_ROOT_ARM64

# Point 4: an empty home and temp folder, so the native libraries are written out NOW (the first-run
# path, the one that can fail) into the default place a user's first run uses: ~/.net on Linux and
# macOS, %TEMP%\.net on Windows. It also keeps the test from leaving them on the machine. On macOS the
# same home holds NumSharp's relocated copy of OpenBLAS (~/.cache/NumSharp/openblas).
mkdir "$work/home" "$work/tmp"
HOME="$work/home"
export HOME
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*)
    TMP="$work/tmp"
    TEMP="$work/tmp"
    export TMP TEMP
    ;;
esac

mkdir "$work/study"
cd "$work/study"
echo "smoke_cli.sh: $(basename "$binary") from $(basename "$input") on $(uname -s) $(uname -m)"

# Runs one optunasharp command. Its standard output lands in $out without carriage returns (a Windows
# program ends lines with CR LF, and `$(...)` only drops the LF); a non-zero exit stops the test and
# shows both streams.
#   $@  the command line after the program name.
invoke() {
  if "$binary" "$@" >"$work/stdout" 2>"$work/stderr"; then
    out=$(tr -d '\r' <"$work/stdout")
  else
    status=$?
    echo "--- stdout" >&2
    cat "$work/stdout" >&2
    echo "--- stderr" >&2
    cat "$work/stderr" >&2
    fail "'optunasharp $*' exited with $status."
  fi
}

# Point 3.
invoke --version
[ "$out" = "$expected" ] || fail "--version printed '$out', expected '$expected'."

storage=sqlite:///smoke.db
space='{"x": {"name": "FloatDistribution", "attributes": {"step": null, "low": -10.0, "high": 10.0, "log": false}}, "y": {"name": "FloatDistribution", "attributes": {"step": null, "low": -10.0, "high": 10.0, "log": false}}}'

invoke create-study --storage "$storage" --study-name smoke
[ "$out" = "smoke" ] || fail "create-study printed '$out', expected the study name."

# Points 4 and 5. Trial 0 is CMA-ES's random start-up trial; from trial 1 on every `ask` decomposes the
# covariance matrix, and trial 7 starts the second generation, which needs the update `tell` computed
# from the first six. The value told is the trial number plus one, so trial 0 is the best trial.
i=0
while [ "$i" -lt 9 ]; do
  invoke ask --storage "$storage" --study-name smoke --sampler CmaEsSampler \
    --sampler-kwargs '{"seed": 1}' --search-space "$space"
  case "$out" in
    "{\"number\": $i, \"params\": {\"x\": "*", \"y\": "*"}}") ;;
    *) fail "ask for trial $i printed: $out" ;;
  esac
  invoke tell --storage "$storage" --study-name smoke --trial-number "$i" --values "$((i + 1))"
  i=$((i + 1))
done

invoke trials --storage "$storage" --study-name smoke -f json
complete=$(printf '%s\n' "$out" | grep -o '"state": "COMPLETE"' | wc -l | tr -d ' ')
[ "$complete" -eq 9 ] || fail "expected 9 COMPLETE trials in the study, found $complete: $out"

invoke best-trial --storage "$storage" --study-name smoke -f json
case "$out" in
  *'"number": 0, "value": 1.0,'*) ;;
  *) fail "best-trial printed: $out" ;;
esac

# Point 4, the other half: the libraries that ran were the bundled ones, written under the empty home
# or temp folder. Finding none would mean the binary loaded copies that happen to exist on this
# machine, which a user's machine does not have.
for library in openblas sqlite3; do
  found=$(find "$work/home" "$work/tmp" -type f -name "*$library*" | head -n 1)
  [ -n "$found" ] || fail "no bundled $library library was written out under the test's home or temp folder."
done

echo "smoke_cli.sh: OK - version $expected, self-contained, 9 CMA-ES trials over SQLite, one process per command."
