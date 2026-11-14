#!/usr/bin/env bash
# Sourced by the test steps of build-and-release.yml: decides OpenBLAS's kernel family for this runner and exports
# OPENBLAS_CORETYPE accordingly.
#
# Why a pin at all: OpenBLAS (a DYNAMIC_ARCH build, both in NumSharp.Interop.OpenBLAS and in NumPy's wheels) picks its
# kernels per CPU. The committed linalg oracles (AGE-MOEA-2's operator parity, the CMA-ES state oracle, the xNES digest)
# were captured on its Haswell kernels, and the other families (SkylakeX / Zen on AVX-512 hosts) land 1-2 ULP away, so
# CI pins Haswell on x64; every x64 runner has the AVX2/FMA those kernels need.
#
# The one exception, Windows on an AMD CPU with AVX-512: NumSharp 0.81.0 bundles OpenBLAS 0.3.34.106.0, whose WINDOWS
# build running FORCED Haswell kernels on Zen 4 / Zen 5 (the EPYC 9V74 and 9V45 runners) faults with an access violation
# (0xC0000005) inside JANUS-DE's quadratic chart, on net8.0 and net10.0, with one OpenBLAS thread or four, with .NET's
# AVX-512 on or off. On the same machines OpenBLAS's own kernel choice passes, and so does NumSharp 0.80.0's OpenBLAS
# 0.3.31.22.0 with the pin; AMD EPYC 7763 (Zen 3) and the Intel Xeons pass every way. Found 2026-11-14 by the diagnostic
# runs 43548926143, 43549856491 and 43550350376 (throwaway branch ci/win-crash-diag). There the pin is left off:
# OpenBLAS chooses, and the oracle guards (OpenBlasEngine.CoreName must be "Haswell") report the Haswell-captured cases
# Inconclusive instead of red.
#
# Linux keeps the pin on every CPU: the same console repro with Haswell forced passed on three Linux EPYC 9V74 (Zen 4)
# runners, both TFMs (run 43553335934, branch ci/linux-haswell-check), and leaving the pin off there broke a live
# byte-for-byte gate (CmaEsSamplerParityTests.LongRun_40D_FullState_IsBitIdentical on an AMD AVX-512 runner,
# rehearsal 43551908135), so the exception would cost a parity gate for nothing.
#
# Usage: source "$GITHUB_WORKSPACE/.github/scripts/openblas_coretype.sh"   (bash; reads RUNNER_ARCH and RUNNER_OS)
# Effect: exports OPENBLAS_CORETYPE=Haswell or unsets it, and prints one line saying which and why. Off x64 (the macOS
# arm64 runner) it does nothing: Haswell kernels do not exist there.

# True on an AMD CPU with AVX-512 (Zen 4 and later). Windows asks .NET through PowerShell 7, which every hosted Windows
# runner has; it is tested first because Git Bash also has a /proc/cpuinfo (MSYS2's emulation), whose flag list is not
# known to name AVX-512. Elsewhere it reads the kernel's /proc/cpuinfo. Anything it cannot tell counts as "not AMD with
# AVX-512", i.e. pinned.
openblas_amd_avx512() {
  if [[ "${RUNNER_OS:-}" == "Windows" || "$(uname -s)" == MINGW* || "$(uname -s)" == MSYS* ]]; then
    command -v pwsh >/dev/null 2>&1 || return 1
    [[ "$(pwsh -NoProfile -NonInteractive -Command '[Console]::Write(((Get-CimInstance Win32_Processor | Select-Object -First 1).Manufacturer -eq "AuthenticAMD") -and [System.Runtime.Intrinsics.X86.Avx512F]::IsSupported)')" == "True" ]]
  elif [[ -r /proc/cpuinfo ]]; then
    grep -m1 '^vendor_id' /proc/cpuinfo | grep -q AuthenticAMD && grep -qw avx512f /proc/cpuinfo
  else
    return 1
  fi
}

if [[ "${RUNNER_ARCH:-}" == "X64" ]]; then
  # The exception applies to Windows only (see above); Linux and any other x64 host keep the pin.
  if [[ "${RUNNER_OS:-}" == "Windows" ]] && openblas_amd_avx512; then
    unset OPENBLAS_CORETYPE
    echo "OpenBLAS: Windows on an AMD CPU with AVX-512, Haswell pin left off (OpenBLAS 0.3.34's forced Haswell kernels fault there); the Haswell-captured oracles report Inconclusive"
  else
    export OPENBLAS_CORETYPE=Haswell
    echo "OpenBLAS: OPENBLAS_CORETYPE=Haswell (the kernels the committed linalg oracles were captured on)"
  fi
fi
