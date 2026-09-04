#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false
#:property SkipNDScopeWeave=true
#:property GeneratePackageOnBuild=false

// FFmpeg / x264 encoder-parameter tuning — an OptunaSharp port of the most-starred public Optuna
// gist (25 stars), by Optuna maintainer `sile`:
//   https://gist.github.com/sile/8aa1ff7808dd55298f51dd70c8b83092
//   "Optunaを使ってFFmpegのエンコードパラメータを最適化してみる"
//
// The gist maximizes SSIM under fixed bitrate/resolution by searching a rich, CONDITIONAL x264
// parameter space and prunes hopeless trials early with a SuccessiveHalvingPruner (reporting SSIM at
// 100/300/900-second cut points). Here the real ffmpeg encode is replaced by a deterministic
// surrogate so the example runs anywhere — the search space, the branches (psy-rd under trellis>0;
// deblock under not-no-deblock), the dynamic parameter names (partitions-*, feature flags) and the
// pruner are faithful to the gist. (`suggest_uniform`, removed in Optuna 3.0, is written as its
// modern equivalent `SuggestFloat`.)
//
// The exact search space below is byte-for-byte parity-checked against live Optuna in
// test/OptunaSharp.Tests.Interop/GistExampleParityTests.cs.
//
//   Run:  dotnet run examples/FFmpegX264Tuning.cs

using OptunaSharp;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

var study = Optuna.CreateStudy(
    storage: new InMemoryStorage(),
    sampler: new RandomSampler(seed: 0),   // seeded for reproducibility; swap for `new TPESampler()` for the gist's default
    pruner: new SuccessiveHalvingPruner(minResource: 100, reductionFactor: 3),
    direction: StudyDirection.Maximize,
    studyName: "ffmpeg-x264");

study.Optimize(Objective, nTrials: 40);

Console.WriteLine($"Finished trials: {study.Trials.Count}");
Console.WriteLine($"Best surrogate SSIM: {study.BestValue:0.000000}");
Console.WriteLine("Best x264 params:");
foreach (var kv in study.BestParams)
    Console.WriteLine($"    {kv.Key}: {kv.Value}");

// The x264 search space of the gist (encode replaced by a deterministic surrogate SSIM).
static double Objective(Trial t)
{
    t.SuggestInt("refs", 1, 16);
    double qcomp = t.SuggestFloat("qcomp", 0.0, 1.0);
    t.SuggestInt("b-adapt", 0, 2);
    t.SuggestInt("subme", 1, 9);
    long trellis = t.SuggestInt("trellis", 0, 2);
    if (trellis > 0)
    {
        t.SuggestFloat("psy-rd.0", 0.0, 1.0);
        t.SuggestFloat("psy-rd.1", 0.0, 1.0);
    }

    foreach (var p in new[] { "p8x8", "p4x4", "b8x8", "i8x8", "i4x4" })
        t.SuggestCategorical($"partitions-{p}", new[] { 0, 1 });

    t.SuggestCategorical("me", new[] { "dia", "hex", "umh", "esa" });
    t.SuggestInt("me_range", 4, 16);
    t.SuggestCategorical("direct", new[] { "none", "spatial", "temporal", "auto" });
    t.SuggestInt("rc-lookahead", 10, 100);
    t.SuggestInt("min-keyint", 1, 12);
    t.SuggestInt("scenecut", 10, 100);
    t.SuggestInt("qdiff", 1, 51);

    long noDeblock = t.SuggestCategorical("no-deblock", new[] { 0, 1 });
    if (noDeblock == 0)  // gist: `if not no_deblock:`
    {
        t.SuggestInt("deblockalpha", -6, 6);
        t.SuggestInt("deblockbeta", -6, 6);
    }

    foreach (var f in new[]
             {
                 "b-pyramid", "no-cabac", "mixed-refs", "no-chroma-me", "8x8dct",
                 "no-fast-pskip", "no-dct-decimate", "deadzone-inter", "deadzone-intra", "nr",
             })
        t.SuggestCategorical(f, new[] { 0, 1 });

    // Surrogate for the ffmpeg SSIM measurement, reported at the pruner's 100/300/900 rungs.
    foreach (int d in new[] { 100, 300, 900 })
    {
        t.Report(qcomp + 0.001 * d, d);
        if (t.ShouldPrune())
            throw new TrialPruned();
    }
    return qcomp;
}
