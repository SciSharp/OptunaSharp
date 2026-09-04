# OptunaSharp examples — ported from real-world Optuna gists

Runnable OptunaSharp ports of **battle-tested, in-the-wild Optuna usage**: the most-starred public
gists that use Optuna (ranked by `Gist.stargazerCount`), harvested with the
[`interop-test-porting`](../.claude/skills/interop-test-porting/SKILL.md) skill. Each example mirrors
its source gist's **search space, study configuration, and pruning** faithfully, so you can see how a
real Optuna script looks in C#.

| Example | Source gist | Author | ★ | What it does |
|---|---|---|---|---|
| [`FFmpegX264Tuning.cs`](FFmpegX264Tuning.cs) | [`sile/8aa1ff78…`](https://gist.github.com/sile/8aa1ff7808dd55298f51dd70c8b83092) | `sile` (Optuna maintainer) | 25 | **Real ffmpeg encode** of the rickroll — maximizes libx264's SSIM over a **conditional** x264 space (psy-rd/deblock branches, dynamic `partitions-*`/flag names) with a `SuccessiveHalvingPruner` |
| [`XGBoostHpo.cs`](XGBoostHpo.cs) | [`bexgboost/823df08d…`](https://gist.github.com/bexgboost/823df08d2e3760538e9b931d38439a68) | `bexgboost` | 16 | the classic 8-parameter XGBoost grid (`eta`, `max_depth`, `subsample`, …), minimizing RMSE |
| [`ParamRepeatPruner.cs`](ParamRepeatPruner.cs) | [`aisu-wata0/24045ab5…`](https://gist.github.com/aisu-wata0/24045ab5e8ad007b4a09f708adfe359f) | `aisu-wata0` | 6 | a userland **pruner that skips trials whose params repeat** an earlier trial's — ported and demonstrated end-to-end |

## Run them

```bash
dotnet run examples/XGBoostHpo.cs          # pure managed — nothing extra needed
dotnet run examples/ParamRepeatPruner.cs   # pure managed — nothing extra needed
dotnet run examples/FFmpegX264Tuning.cs    # needs ffmpeg (+ yt-dlp to fetch the rickroll)
```

They are [.NET 10 file-based apps](../.claude/CLAUDE.md) referencing `OptunaSharp.Core` directly; the
`#:property` directives keep the build side-effect-free. `FFmpegX264Tuning` shells out to **ffmpeg**
and fetches the rickroll (`dQw4w9WgXcQ`) with **yt-dlp**; if YouTube blocks the download (a `403` is
common without cookies/PO-token) it falls back to a synthetic `ffmpeg testsrc2` clip, so it always runs
a real encode. Override the clip with `$FFMPEG_TUNE_VIDEO`, the trial count with `$FFMPEG_TUNE_TRIALS`.

## Real vs. surrogate objectives

- **`FFmpegX264Tuning`** and **`ParamRepeatPruner`** run their **real** objectives (an actual encode; a
  real repeat check) — they need no external ML runtime.
- **`XGBoostHpo`** uses a **deterministic surrogate** for the objective body: OptunaSharp can't run
  XGBoost in-process, and a sampler's trajectory is fixed by `(seed, search space, value sequence)`
  alone — so the example keeps bexgboost's exact search space and stands in a closed-form value. The
  **real XGBoost** run happens over in `OptunaSharp.Tests.PackageParity` (`gen_oracle.py`), where Python
  trains real XGBoost on the same grid and the packaged OptunaSharp replays the trajectory.

## Byte-for-byte parity

The gist search spaces are asserted **bit-identical to live Optuna 5.0.0.dev**:

- [`GistExampleParityTests`](../test/OptunaSharp.Tests.Interop/GistExampleParityTests.cs) — the FFmpeg
  (conditional space + `SuccessiveHalvingPruner`) and XGBoost grids, full-state
  `StudySnapshot.AssertParity` over the RNG-exact `RandomSampler` path. It uses a deterministic
  surrogate objective so the comparison is reproducible; the runnable `FFmpegX264Tuning` example swaps
  in the real encode over the identical space.
- [`ParamRepeatPrunerParityTests`](../test/OptunaSharp.Tests.Interop/ParamRepeatPrunerParityTests.cs) —
  runs the gist's pruner in Python and the C# port side by side under `RandomSampler`, asserting the
  same trials end up COMPLETE vs PRUNED.

TPE parity over these particular spaces is intentionally not asserted — each carries an integer
parameter whose repeated observations hit the documented NumPy x86-simd-sort tie-order divergence in
TPE's `compute_sigmas`, so parity is asserted on the RNG-exact path (see the `testing` skill).
