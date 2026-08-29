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

### The NumSharp reference — "csproj for now, nuget in future"

`OptunaSharp.Core` depends on NumSharp, resolved conditionally:

- **Local dev** — when a sibling `NumSharp` checkout exists next to this repo, NumSharp is
  referenced by **project**, so changes flow live.
- **Standalone / CI** — otherwise the published **NuGet** package is used.

Both are mutually exclusive on `Exists(...)`, so exactly one applies and packaging always records a
`NumSharp` package dependency.

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
