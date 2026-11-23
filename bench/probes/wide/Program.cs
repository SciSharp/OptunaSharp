// High-dimensional rows of the Samplers table (C# side; py/wide.py mirrors it): many parameters per trial.
// Configs: TPE independent 100-D x 150 trials, TPE multivariate 100-D x 150, CMA-ES 200-D x 100, GP 30-D x 40,
// Random 1000-D x 200. Each runs under bench/common/Bench.cs's protocol (warm-up runs never counted, then measured runs, each a
// fresh sampler and study with the same seed, timing only Optimize); until 2026-11-23 each ran once in a fresh process.
// Usage: wide <tpe_ind|tpe_mv|cmaes|gp|random>
using System.Diagnostics;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Error);
// A factory per configuration: every run needs a new sampler with the same seed (a sampler's state carries between trials).
var (make, dims, trials) = args[0] switch
{
    "tpe_ind" => ((Func<BaseSampler>)(() => new TPESampler(seed: 0, multivariate: false)), 100, 150),
    "tpe_mv" => (() => new TPESampler(seed: 0, multivariate: true), 100, 150),
    "cmaes" => (() => new CmaEsSampler(seed: 0), 200, 100),
    "gp" => (() => new GPSampler(seed: 0), 30, 40),
    _ => (() => new RandomSampler(seed: 0), 1000, 200),
};

Bench.Run(args[0], () =>
{
    var study = Optuna.CreateStudy(sampler: make());
    var sw = Stopwatch.StartNew();
    // sum((x_i - 1)^2) over dims floats in [-5, 5]; the names are formatted on every call, as Python's f"x{i}" is.
    study.Optimize(t =>
    {
        double s = 0;
        for (int i = 0; i < dims; i++) { double v = t.SuggestFloat($"x{i}", -5, 5); s += (v - 1) * (v - 1); }
        return s;
    }, nTrials: trials);
    sw.Stop();
    return new BenchRun(sw.Elapsed.TotalSeconds, Bench.Fp(study.BestValue), $"dims={dims} trials={trials}");
});
Console.WriteLine($"C# {args[0]} {dims}-D x {trials}: peakWS={Process.GetCurrentProcess().PeakWorkingSet64 / 1048576}MB");
