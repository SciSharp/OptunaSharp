// Measures how Optimize(nJobs) scales a CPU-bound objective across cores (README "Parallel trials"; py/njobs_scaling.py
// mirrors it for Optuna and Rustuna).
// The objective burns ~50 ms of single-threaded floating-point work per trial; the sampler is RandomSampler, so sampling cost
// is negligible next to it. Each nJobs value runs under bench/common/Bench.cs's protocol (warm-up runs never counted, then
// measured runs, each a fresh 64-trial study); until 2026-11-23 it was three runs per nJobs with no warm-up, and the median.
using System.Diagnostics;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);
/// <summary>Whether an assembly was compiled with the JIT optimizer disabled (a Debug build).</summary>
/// <param name="a">The assembly to inspect.</param>
/// <returns><see langword="true"/> when its <see cref="DebuggableAttribute"/> disables the JIT optimizer; <see langword="false"/> when it has none or keeps the optimizer on.</returns>
/// <remarks>
/// <c>dotnet run file.cs</c> once ran these probes with the JIT optimizer off (it ignores a file's <c>Configuration</c>
/// property), so every run states the build it actually measured, for the probe and for OptunaSharp.
/// </remarks>
static bool JitOff(System.Reflection.Assembly a) =>
    (Attribute.GetCustomAttribute(a, typeof(DebuggableAttribute)) as DebuggableAttribute)?.IsJITOptimizerDisabled ?? false;
Console.WriteLine($"JIT optimizer disabled: script={JitOff(typeof(Program).Assembly)}, OptunaSharp={JitOff(typeof(Optuna).Assembly)}");
// The Windows power-throttling opt-out this probe used to apply inline now comes from ../../common/ProbeEnv.cs (a module
// initializer that also prints the CPUs the process may use), so the same build runs on Linux, where the worker-machine run
// pins it with taskset instead.

// Calibrate the loop length to ~50 ms on one core, after a warm-up so the JIT has tiered the loop up. Each process calibrates
// itself (as each Python process does), so the 1-worker time is ~64 x 50 ms = 3.2 s everywhere; the parser's validity rule
// rejects a process whose calibration missed by more than 10 %.
int iters = 1_000_000;
for (int w = 0; w < 3; w++) Burn(0.5, iters);
var calib = Stopwatch.StartNew();
Burn(0.5, iters);
iters = (int)(iters * 50.0 / calib.Elapsed.TotalMilliseconds);
const int nTrials = 64;
Console.WriteLine($"OptunaSharp | {Environment.ProcessorCount} logical CPUs | {nTrials} trials x ~50 ms CPU-bound objective | iters={iters}");

double baseline = 0;
foreach (int nJobs in new[] { 1, 2, 4, 8, 16 })
{
    IReadOnlyList<BenchRun> runs = Bench.Run($"njobs{nJobs}", () =>
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 1));
        var sw = Stopwatch.StartNew();
        study.Optimize(trial =>
        {
            double x = trial.SuggestFloat("x", -10, 10);
            // the burn's result feeds the value so the JIT cannot drop it; 1e-300 keeps the value itself unchanged
            return (x - 2) * (x - 2) + Burn(x, iters) * 1e-300;
        }, nTrials: nTrials, nJobs: nJobs);
        sw.Stop();
        if (study.Trials.Count != nTrials) throw new InvalidOperationException($"expected {nTrials} trials, got {study.Trials.Count}");
        // The fingerprint is the trial count only: with nJobs > 1 the sampler's generator is reseeded per worker (Optuna's
        // reseed_rng, which OptunaSharp follows), so the best value varies between runs, while the work does not (every trial
        // burns the same ~50 ms whatever x it drew).
        return new BenchRun(sw.Elapsed.TotalSeconds, study.Trials.Count.ToString(), $"njobs={nJobs}");
    });
    double median = Bench.Median(runs);
    if (nJobs == 1) baseline = median;
    Console.WriteLine($"nJobs={nJobs,2}  median {median,7:F3} s over {runs.Count} measured runs  speedup {baseline / median,5:F2}x");
}

/// <summary><paramref name="iters"/> sines: about 50 ms of single-threaded floating-point work once calibrated.</summary>
/// <param name="x">The trial's parameter: the loop depends on it, so the JIT cannot hoist or fold the work away.</param>
/// <param name="iters">Iterations; calibrated once per process to ~50 ms.</param>
/// <returns>The sum of the sines; the objective adds it times 1e-300, which keeps the loop alive without changing the value.</returns>
static double Burn(double x, int iters)
{
    double s = 0;
    for (int i = 0; i < iters; i++) s += Math.Sin(x + i * 1e-3);
    return s;
}
