# Getting Started

## Install

```bash
dotnet add package OptunaSharp
```

OptunaSharp multi-targets **.NET 8** and **.NET 10** and depends on **NumSharp**.

## Build from source

```bash
git clone https://github.com/SciSharp/OptunaSharp
cd OptunaSharp
dotnet build SciSharp.OptunaSharp.sln -c Release
```

### The NumSharp reference

`OptunaSharp.Core` depends on **NumSharp** and its native LAPACK/CBLAS backend
**NumSharp.Interop.OpenBLAS**, both consumed as **NuGet packages** from nuget.org (version pinned by
`NumSharpVersion` in `Directory.Build.props`). The OpenBLAS package ships the native LAPACK binaries,
so `np.linalg` is bit-identical to NumPy with no extra setup — build and CI just restore from nuget.org.

The matching NumSharp source is vendored reference-only at `refs/NumSharp` (a pinned submodule on the
same release tag) for browsing; it is not part of the build.

## Test

```bash
# Unit tests (both target frameworks)
dotnet test test/OptunaSharp.Tests/OptunaSharp.Tests.csproj -c Release \
  --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory&TestCategory!=RequiresPython"

# Live-Optuna interop parity (needs a discoverable Python with `pip install optuna`)
dotnet test test/OptunaSharp.Tests.Interop/OptunaSharp.Tests.Interop.csproj -c Release
```

The interop suite self-skips (Inconclusive) when no Python/Optuna is found, so it never turns a
machine without Python red.

## Pack

```bash
dotnet pack src/OptunaSharp.Core/OptunaSharp.Core.csproj -c Release -o packages
```

## Build the documentation site locally

```bash
dotnet tool install -g docfx
docfx docs/website-src/docfx.json --serve
```

DocFX generates the API reference from `OptunaSharp.Core` and renders the conceptual docs into
`docs/website/`.
