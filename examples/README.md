# OptunaSharp examples

Runnable OptunaSharp programs — [.NET 10 file-based apps](../.claude/CLAUDE.md) you launch with
`dotnet run examples/<Name>.cs`. Two kinds: **ports of battle-tested public gists** (the most-starred
public Optuna usage, harvested with the [`interop-test-porting`](../.claude/skills/interop-test-porting/SKILL.md)
skill), and **feature showcases** of canonical Optuna capabilities.

## Ported from real-world gists

| Example | Source gist | ★ | What it does |
|---|---|---|---|
| [`FFmpegX264Tuning.cs`](FFmpegX264Tuning.cs) | [`sile/8aa1ff78…`](https://gist.github.com/sile/8aa1ff7808dd55298f51dd70c8b83092) (Optuna maintainer) | 25 | **Real ffmpeg encode** of the rickroll — maximizes libx264's SSIM over a conditional x264 space + `SuccessiveHalvingPruner` |
| [`XGBoostHpo.cs`](XGBoostHpo.cs) | [`bexgboost/823df08d…`](https://gist.github.com/bexgboost/823df08d2e3760538e9b931d38439a68) | 16 | the classic 8-parameter XGBoost grid, minimizing RMSE |
| [`ParamRepeatPruner.cs`](ParamRepeatPruner.cs) | [`aisu-wata0/24045ab5…`](https://gist.github.com/aisu-wata0/24045ab5e8ad007b4a09f708adfe359f) | 6 | a userland pruner that skips trials whose params repeat an earlier trial's |

## Feature showcases

Each exercises a whole Optuna capability, with a deterministic surrogate objective so it runs anywhere.

| Example | First shows |
|---|---|
| [`MedianPrunedTraining.cs`](MedianPrunedTraining.cs) | `MedianPruner` + `report`/`should_prune` over epochs — the canonical prune-the-laggards loop (the sesame-finetune 113★ pattern) |
| [`MultiObjectivePareto.cs`](MultiObjectivePareto.cs) | multi-objective with `NSGAIISampler`, `directions=[…]`, and `study.BestTrials` — the Pareto front |
| [`HyperparameterImportance.cs`](HyperparameterImportance.cs) | `ImportanceFunctions.GetParamImportances` — fANOVA vs PedAnova on an objective rigged so `a ≫ b ≫ c` |
| [`AutomaticTermination.cs`](AutomaticTermination.cs) | the `Terminator` / `TerminatorCallback` — stop automatically when the search stops paying off |

## Run them

```bash
dotnet run examples/XGBoostHpo.cs            # pure managed — nothing extra
dotnet run examples/ParamRepeatPruner.cs     # pure managed
dotnet run examples/MedianPrunedTraining.cs  # pure managed
dotnet run examples/MultiObjectivePareto.cs  # pure managed
dotnet run examples/HyperparameterImportance.cs
dotnet run examples/AutomaticTermination.cs  # uses the OpenBLAS-backed GP terminator path
dotnet run examples/FFmpegX264Tuning.cs      # needs ffmpeg (+ yt-dlp to fetch the rickroll)
```

`FFmpegX264Tuning` shells out to **ffmpeg** and fetches the rickroll (`dQw4w9WgXcQ`) with **yt-dlp**; if
YouTube blocks the download (a `403` without cookies/PO-token) it falls back to an ffmpeg `testsrc2`
clip, so it always runs a real encode.

## Real vs. surrogate objectives

`FFmpegX264Tuning` (an actual encode) and `ParamRepeatPruner` (a real repeat check) run their **real**
objectives. The rest use a **deterministic surrogate** for the objective body — OptunaSharp can't run
XGBoost/train a net in-process, and a sampler's trajectory is fixed by `(seed, search space, value
sequence)` alone, so the surrogate keeps the example self-contained while exercising the real feature.
The **real ML** runs (XGBoost, LightGBM, sklearn) happen in `OptunaSharp.Tests.PackageParity`, where
Python trains the real model and the packaged OptunaSharp replays the trajectory.

## Byte-for-byte parity

The gist search spaces are asserted bit-identical to live Optuna 5.0.0.dev —
[`GistExampleParityTests`](../test/OptunaSharp.Tests.Interop/GistExampleParityTests.cs) (FFmpeg +
XGBoost) and [`ParamRepeatPrunerParityTests`](../test/OptunaSharp.Tests.Interop/ParamRepeatPrunerParityTests.cs).
The features the showcases exercise have their own parity coverage in the interop suite (pruner
decisions, NSGA-II, importance, terminator). Parity is asserted on the RNG-exact `RandomSampler` path;
TPE over integer spaces isn't byte-comparable (NumPy simd-sort tie order — see the `testing` skill).
