// Scalability probe (C# side; py/scale.py mirrors it), under bench/common/Bench.cs's protocol since 2026-11-23.
// "inmem": one run is a whole 50,000-trial RandomSampler study in memory, timed per block of 10,000. The run's seconds are the
//          whole study; the blocks (printed as blocks=) show whether the per-trial cost stays flat as the history grows. In a
//          fresh process the blocks used to fall (0.54 -> 0.12 s on the worker), which was the JIT warming up, not the study:
//          after the warm-up runs they are the study's own cost. The cost of one study.Trials copy at 50k is printed too.
// "tpe":   2,000 TPE trials, timed per 250-trial block, showing how the per-trial cost grows with history (not in the README).
// Usage: scale <inmem|tpe>
using System.Diagnostics;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Error);
string mode = args[0];

// sum(x_i^2) over four floats in [-5, 5].
double Obj(Trial t)
{
    double s = 0;
    for (int i = 0; i < 4; i++) { double v = t.SuggestFloat($"x{i}", -5, 5); s += v * v; }
    return s;
}

if (mode == "inmem")
{
    Bench.Run("inmem", () =>
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 0));
        var blocks = new List<string>();
        double total = 0;
        for (int b = 0; b < 5; b++)
        {
            var sw = Stopwatch.StartNew();
            study.Optimize(Obj, nTrials: 10_000);
            double s = sw.Elapsed.TotalSeconds;
            total += s;
            blocks.Add(s.ToString("F6", System.Globalization.CultureInfo.InvariantCulture));
        }
        // Not part of the run's seconds: one deep copy of the 50,000 trials, the cost a caller pays for study.Trials.
        var cp = Stopwatch.StartNew();
        int n = study.Trials.Count;
        double copyMs = cp.Elapsed.TotalMilliseconds;
        return new BenchRun(total, $"{n}:{Bench.Fp(study.BestValue)}",
            $"blocks={string.Join(",", blocks)} trials-copy={copyMs:F0}ms");
    });
    GC.Collect();
    Console.WriteLine($"C# inmem 50k: heap={GC.GetTotalMemory(true) / 1048576}MB peakWS={Process.GetCurrentProcess().PeakWorkingSet64 / 1048576}MB");
}
else
{
    Bench.Run("tpe", () =>
    {
        var study = Optuna.CreateStudy(sampler: new TPESampler(seed: 0));
        var blocks = new List<string>();
        double total = 0;
        for (int b = 0; b < 8; b++)
        {
            var sw = Stopwatch.StartNew();
            study.Optimize(Obj, nTrials: 250);
            double s = sw.Elapsed.TotalSeconds;
            total += s;
            blocks.Add(s.ToString("F3", System.Globalization.CultureInfo.InvariantCulture));
        }
        return new BenchRun(total, Bench.Fp(study.BestValue), $"blocks={string.Join(",", blocks)}");
    });
    Console.WriteLine($"C# tpe 2000: peakWS={Process.GetCurrentProcess().PeakWorkingSet64 / 1048576}MB");
}
