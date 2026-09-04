# Temporary NumSharp feed for CI (remove at NumSharp 0.70.0)

OptunaSharp.Core targets a **NumSharp newer than the published 0.60.0**. In a clean container or CI
runner (no sibling `NumSharp` checkout) Core therefore cannot restore, compile, or run:

| Need | Why nuget.org 0.60.0 fails |
|------|----------------------------|
| `NumSharp.Interop.OpenBLAS` (whole package) | not published to nuget.org at all → `NU1101` |
| `np.linalg`, `NumSharp.RuntimeError`, `NumSharp.KeyError` | not present in the published `NumSharp` 0.60.0 → `CS0117` / `CS0234` |

Both come from the sibling NumSharp (the 0.70.0-to-be line). Until 0.70.0 ships, CI bridges the gap by
**packing both from a sibling checkout into a local feed listed before nuget.org**.

## Use in CI

```bash
# 1. get the sibling NumSharp (pin the commit/branch that matches Core)
git clone --depth 1 https://github.com/SciSharp/NumSharp ../NumSharp

# 2. pack the temp feed
.github/scripts/pack-temp-numsharp-feed.sh ../NumSharp ci-feed
#    -> ci-feed/NumSharp.0.60.0.nupkg
#    -> ci-feed/NumSharp.Interop.OpenBLAS.0.60.0.nupkg   (native win/linux/osx x64+arm64)

# 3. restore with the feed FIRST so the local 0.60.0 shadows nuget's stale 0.60.0
#    (a NuGet.config with <clear/>, then ci-feed, then nuget.org)
```

For the **containerized** suites, the workflow packs `ci-feed` once on the runner and each suite's
Dockerfile `COPY`s it in (with a feed-first `NuGet.config`) before `dotnet test`. No sibling source has
to enter the images.

Verified end-to-end against `OptunaSharp.Tests` (scaffolding + `np` numeric smoke), net8.0 + net10.0,
in a clean force-fallback build (`-p:NumSharpCoreProject=__none__ -p:NumSharpOpenBlasProject=__none__`).

## Removing it (at NumSharp 0.70.0)

1. Delete `pack-temp-numsharp-feed.sh` + this file.
2. Remove the `ci-feed` steps and the feed entry from the CI `NuGet.config`.
3. Bump Core's `NumSharp` / `NumSharp.Interop.OpenBLAS` PackageReference fallback to `0.70.0`.
