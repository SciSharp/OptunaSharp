// Plan 11's mode probe (README "Per-worker samplers"). What do SamplerConcurrency.PerWorker and the OpenBLAS policy buy?
// Each configuration runs under bench/common/Bench.cs's protocol (warm-up runs never counted, then measured runs, each a fresh
// study); until 2026-11-23 it ran three times and kept the best, a minimum, which reads low against every other table's
// statistic. Modes (first argument):
//   tpe-heavy: TPE multivariate over 10 parameters with a 5 ms objective, and a 40-dimensional CMA-ES on a CPU-only objective,
//              each Shared vs PerWorker at nJobs 8;
//   samplers : TPE Shared vs PerWorker, GP with OpenBLAS automatic vs all threads, on a 20 ms sleeping objective;
//   cma      : a 40-dimensional CMA-ES at nJobs 8 on a CPU-only objective, OpenBLAS automatic (1 thread) vs all threads vs the
//              old default (no policy: BlasThreads = the count OpenBLAS starts with);
//   mem      : the memory OpenBLAS commits for its threads (not timed).
using System.Diagnostics;
using NumSharp;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Error);
bool jitOff = typeof(Study).Assembly.GetCustomAttributes(typeof(DebuggableAttribute), false)
    .OfType<DebuggableAttribute>().FirstOrDefault()?.IsJITOptimizerDisabled ?? false;
int startThreads = (TensorEngine.Threading.TryGet(TensorEngine.Threading.OpenBlas, out var knob) ? knob.Threads : null) ?? 1;
Console.WriteLine($"JIT optimizer disabled: {jitOff}; cores {Environment.ProcessorCount}; OpenBLAS starts with {startThreads} threads");
string mode = args.Length > 0 ? args[0] : "samplers";

if (mode == "mem")
{
    // The memory OpenBLAS commits for its threads: run once with OPENBLAS_NUM_THREADS=1 set before start, once without.
    // Every intermediate array is named and disposed (NumSharp's NDW012): the probe measures the process's memory, so arrays
    // left to the finalizer would sit in the numbers.
    using var ones = np.ones(new Shape(200, 200));
    using var eye = np.eye(200);
    using var diagonal = eye * 200;
    using var a = ones + diagonal;
    using var c = np.linalg.cholesky(a);
    var self = Process.GetCurrentProcess();
    self.Refresh();
    Console.WriteLine($"OPENBLAS_NUM_THREADS={Environment.GetEnvironmentVariable("OPENBLAS_NUM_THREADS") ?? "(unset)"}: " +
                      $"private bytes {self.PrivateMemorySize64 / (1024 * 1024)} MB, working set {self.WorkingSet64 / (1024 * 1024)} MB");
    return;
}

if (mode == "tpe-heavy")
{
    // TPE where sampling dominates: multivariate over 10 parameters, 600 trials, a 5 ms objective.
    var perWorker = new TrialExecutorOptions { SamplerConcurrency = SamplerConcurrency.PerWorker };
    Run("TPE-mv 10-d nJobs 1", () => new TPESampler(seed: 1, multivariate: true), 600, 1, null, Ten);
    Run("TPE-mv 10-d nJobs 8 Shared", () => new TPESampler(seed: 1, multivariate: true), 600, 8, null, Ten);
    Run("TPE-mv 10-d nJobs 8 PerWorker", () => new TPESampler(seed: 1, multivariate: true), 600, 8, perWorker, Ten);
    Run("CMA-ES 40-d nJobs 8 Shared", () => new CmaEsSampler(seed: 1), 2000, 8, null, Sphere40);
    Run("CMA-ES 40-d nJobs 8 PerWorker", () => new CmaEsSampler(seed: 1), 2000, 8, perWorker, Sphere40);
    return;
}

if (mode == "samplers")
{
    var perWorker = new TrialExecutorOptions { SamplerConcurrency = SamplerConcurrency.PerWorker };
    Run("TPE nJobs 1", () => new TPESampler(seed: 1), 400, 1, null, SleepObjective);
    Run("TPE nJobs 8 Shared", () => new TPESampler(seed: 1), 400, 8, null, SleepObjective);
    Run("TPE nJobs 8 PerWorker", () => new TPESampler(seed: 1), 400, 8, perWorker, SleepObjective);
    Run("GP nJobs 1", () => new GPSampler(seed: 1), 120, 1, null, SleepObjective);
    Run("GP nJobs 8, OpenBLAS 1 thread (automatic)", () => new GPSampler(seed: 1), 120, 8, null, SleepObjective);
    Run("GP nJobs 8, OpenBLAS all threads", () => new GPSampler(seed: 1), 120, 8,
        new TrialExecutorOptions { BlasThreads = startThreads }, SleepObjective);
}
else
{
    Run("CMA-ES 40-d nJobs 1", () => new CmaEsSampler(seed: 1), 2000, 1, null, Sphere40);
    Run("CMA-ES 40-d nJobs 8, OpenBLAS 1 thread (automatic)", () => new CmaEsSampler(seed: 1), 2000, 8, null, Sphere40);
    Run($"CMA-ES 40-d nJobs 8, OpenBLAS {startThreads} threads (no policy)", () => new CmaEsSampler(seed: 1), 2000, 8,
        new TrialExecutorOptions { BlasThreads = startThreads }, Sphere40);
}

/// <summary>
/// Measures one configuration under the protocol (bench/common/Bench.cs): a fresh study per run with only
/// <c>Optimize</c> timed, then a summary line with the median of the measured runs.
/// </summary>
/// <param name="name">The configuration's name; the label is the name with spaces as underscores (the line format splits on spaces), which the parser prints back as spaces.</param>
/// <param name="make">Makes the run's sampler; called once per run, so every run starts from the same seed.</param>
/// <param name="trials">Trials per run.</param>
/// <param name="nJobs">Workers per run.</param>
/// <param name="options">Executor options (sampler concurrency, OpenBLAS threads), or <see langword="null"/> for the defaults.</param>
/// <param name="objective">The objective every trial evaluates.</param>
/// <remarks>
/// With <paramref name="nJobs"/> above 1 the order in which workers take trials varies, so the fingerprint is the trial
/// count, not the best value. Any exception from the study propagates and ends the process with a non-zero exit code, which
/// the runner logs and the parser drops.
/// </remarks>
static void Run(string name, Func<BaseSampler> make, int trials, int nJobs, TrialExecutorOptions? options,
    Func<Trial, double> objective)
{
    var runs = Bench.Run(name.Replace(' ', '_'), () =>
    {
        var study = Optuna.CreateStudy(sampler: make());
        var sw = Stopwatch.StartNew();
        study.Optimize(objective, nTrials: trials, nJobs: nJobs, executorOptions: options);
        sw.Stop();
        return new BenchRun(sw.Elapsed.TotalSeconds, study.Trials.Count.ToString(), $"trials={trials} njobs={nJobs}");
    });
    double median = Bench.Median(runs);
    Console.WriteLine($"{name}: {median:F3} s for {trials} trials (median of {runs.Count} measured runs); {trials / median:F1} trials/s");
}

/// <summary>
/// Three floats in [-5, 5] and a 20 ms sleep: a trial that waits, so the sampler's share of the trial is what
/// <c>SamplerConcurrency.PerWorker</c> can parallelize (little, which is the point of this row).
/// </summary>
/// <param name="trial">The trial to suggest from.</param>
/// <returns>The sphere value x² + y² + z².</returns>
static double SleepObjective(Trial trial)
{
    double x = trial.SuggestFloat("x", -5, 5);
    double y = trial.SuggestFloat("y", -5, 5);
    double z = trial.SuggestFloat("z", -5, 5);
    Thread.Sleep(20);
    return x * x + y * y + z * z;
}

/// <summary>Ten floats in [-5, 5] and a 5 ms sleep: multivariate TPE's sampling dominates the trial.</summary>
/// <param name="trial">The trial to suggest from.</param>
/// <returns>The sum of the ten squares.</returns>
static double Ten(Trial trial)
{
    double sum = 0;
    for (int i = 0; i < 10; i++)
    {
        double v = trial.SuggestFloat("x" + i, -5, 5);
        sum += v * v;
    }
    Thread.Sleep(5);
    return sum;
}

/// <summary>Forty floats in [-5, 5] and no sleep: CMA-ES's sampling and the study's bookkeeping are the whole trial.</summary>
/// <param name="trial">The trial to suggest from.</param>
/// <returns>The sum of the forty squares.</returns>
static double Sphere40(Trial trial)
{
    double sum = 0;
    for (int i = 0; i < 40; i++)
    {
        double v = trial.SuggestFloat("x" + i, -5, 5);
        sum += v * v;
    }
    return sum;
}
