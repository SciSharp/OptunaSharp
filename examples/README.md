# OptunaSharp examples — ported from real-world Optuna gists

Runnable OptunaSharp ports of **battle-tested, in-the-wild Optuna usage**: the most-starred public
gists that use Optuna (ranked by `Gist.stargazerCount`), harvested with the
[`interop-test-porting`](../.claude/skills/interop-test-porting/SKILL.md) skill. Each example mirrors
its source gist's **search space, study configuration, and pruning** as closely as OptunaSharp's
public API allows, so you can see how a real Optuna script looks in C#.

| Example | Source gist | Author | ★ | What it tunes |
|---|---|---|---|---|
| [`FFmpegX264Tuning.cs`](FFmpegX264Tuning.cs) | [`sile/8aa1ff78…`](https://gist.github.com/sile/8aa1ff7808dd55298f51dd70c8b83092) | `sile` (Optuna maintainer) | 25 | x264 encoder params — a **conditional** space (psy-rd, deblock branches; dynamic `partitions-*`/flag names) with a `SuccessiveHalvingPruner`, maximizing SSIM |
| [`XGBoostHpo.cs`](XGBoostHpo.cs) | [`bexgboost/823df08d…`](https://gist.github.com/bexgboost/823df08d2e3760538e9b931d38439a68) | `bexgboost` | 16 | the classic 8-parameter XGBoost grid (`eta`, `max_depth`, `subsample`, …), minimizing RMSE |

## Run them

```bash
dotnet run examples/XGBoostHpo.cs
dotnet run examples/FFmpegX264Tuning.cs
```

They are [.NET 10 file-based apps](../.claude/CLAUDE.md) that reference `OptunaSharp.Core` directly;
the `#:property` directives at the top keep the build side-effect-free (no NumSharp weaver/pack). No
Python or Optuna is needed to run them.

## Why a "surrogate" objective

A gist's real objective is a heavy external process — an ffmpeg encode measuring SSIM, an XGBoost
training run measuring RMSE. Those aren't portable and aren't deterministic. But a **sampler's
trajectory is fixed entirely by `(seed, search space, sequence of returned values)`** — not by what
computed those values. So each example keeps the gist's real *search space* verbatim and replaces the
*objective body* with a small deterministic surrogate. That makes the example run anywhere, stay
reproducible under a fixed seed, and — most importantly — makes it **byte-for-byte comparable to live
Optuna**.

## Byte-for-byte parity

The exact search spaces in these examples are asserted **bit-identical to live Optuna 5.0.0.dev** in
[`test/OptunaSharp.Tests.Interop/GistExampleParityTests.cs`](../test/OptunaSharp.Tests.Interop/GistExampleParityTests.cs)
— full-state (`StudySnapshot.AssertParity`) over the `RandomSampler` path, including the FFmpeg
example's pruned trials and pruner system-attributes. Parity is asserted on the RNG-exact
`RandomSampler` path; the examples default to it (seeded) for that reason. TPE parity over these
particular spaces is intentionally not asserted — both carry an integer parameter with few distinct
values, whose repeated observations hit the documented NumPy x86-simd-sort tie-order divergence in
TPE's `compute_sigmas` (see the `ParzenEstimator` / `TpeSamplerParityTests` remarks). Swap the sampler
to `new TPESampler()` in an example to use the gist's default sampler for exploration.
