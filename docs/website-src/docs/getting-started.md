# Getting Started

## Install

```bash
dotnet add package OptunaSharp
```

OptunaSharp multi-targets **.NET 8** and **.NET 10** and depends on **NumSharp**.

**OptunaSharp needs dynamic code.** NumSharp compiles its kernels at run time, so OptunaSharp does not run under
NativeAOT, or anywhere `RuntimeFeature.IsDynamicCodeSupported` is false. A .NET 10 file-based app (`dotnet run app.cs`)
turns NativeAOT on by default, so add `PublishAot=false` next to the package directive:

```csharp
#:package OptunaSharp@*
#:property PublishAot=false
```

Without it, the first trial fails with `PlatformNotSupportedException: Dynamic code generation is not supported on this
platform.`

## Your first study

```csharp
using OptunaSharp;
using OptunaSharp.Samplers;

var study = Optuna.CreateStudy(sampler: new TPESampler(seed: 42));

study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);   // define-by-run: the search space is declared here
    return (x - 2) * (x - 2);                      // the value to minimize
}, nTrials: 20);

Console.WriteLine(study.BestValue);        // 0.0004973004749973951
Console.WriteLine(study.BestParams["x"]);  // 2.0223002348641756
```

Python Optuna 5.0.0 prints the same two numbers for
`optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))` optimized for 20 trials: seeded
sampling reproduces Optuna's draws exactly. Names follow .NET casing (`suggest_float` → `SuggestFloat`,
`n_trials` → `nTrials`); parameter order and defaults are Optuna's. Next:
[Study and Trials](study-and-trials.md) covers ask-and-tell, multi-objective studies and attributes,
and [Samplers](samplers.md) and [Pruners](pruners.md) cover the algorithms.

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
# Unit tests (both target frameworks), as per-push CI runs them; drop `&TestCategory!=Slow` for the full run
dotnet test test/OptunaSharp.Tests/OptunaSharp.Tests.csproj -c Release \
  --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory&TestCategory!=RequiresPython&TestCategory!=Slow"

# Live-Optuna interop parity: needs a discoverable Python with the vendored Optuna, the version parity is
# measured against (`git submodule update --init refs/optuna`, then `pip install -e refs/optuna`)
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
