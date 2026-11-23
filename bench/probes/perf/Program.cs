// Samplers table, the 6-parameter rows (C# side; py/perf.py mirrors it). For each sampler a seeded study on a 6-D continuous
// objective runs a fixed trial count. Every configuration runs under bench/common/Bench.cs's protocol: warm-up runs that are
// never counted, then measured runs, each one a fresh sampler and study with the same seed, timing only Optimize. (Until
// 2026-11-23 each configuration ran once in a fresh process, so the JIT's warm-up was most of the number.)
// "storage_*" configurations time in-memory against SQLite and journal-file storage with RandomSampler; the README does not
// print them, and they follow the same protocol.
// Usage: perf <random|tpe|tpe_mv|gp|cmaes|nsga2|qmc|storage_inmem|storage_sqlite|storage_journal>
using System.Diagnostics;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

OptunaLogging.SetVerbosity(LogLevel.Error);
string cfg = args[0];

// sum((x_i - 1)^2) over six floats in [-5, 5]; the names are formatted on every call, as Python's f"x{i}" is.
double Obj(Trial t)
{
    double s = 0;
    for (int i = 0; i < 6; i++) { double v = t.SuggestFloat($"x{i}", -5, 5); s += (v - 1) * (v - 1); }
    return s;
}

// A factory, not a sampler: a sampler keeps state between trials (its random stream, its caches), so a reused one would make
// the second run continue the first run's stream and do different work. Every run gets a new one with the same seed.
(Func<BaseSampler> Make, int Trials) Pick() => cfg switch
{
    "random" => (() => new RandomSampler(seed: 0), 5000),
    "tpe" => (() => new TPESampler(seed: 0), 500),
    "tpe_mv" => (() => new TPESampler(seed: 0, multivariate: true), 500),
    "gp" => (() => new GPSampler(seed: 0), 100),
    "cmaes" => (() => new CmaEsSampler(seed: 0), 3000),
    "nsga2" => (() => new NSGAIISampler(seed: 0), 3000),
    "qmc" => (() => new QMCSampler(seed: 0), 3000),
    _ => (() => new RandomSampler(seed: 0), 2000),
};

if (cfg.StartsWith("storage_"))
{
    Bench.Run(cfg, () =>
    {
        // One folder per run, created and removed outside the timer, so no run reads another's database or log.
        string dir = Path.Combine(Path.GetTempPath(), $"optunasharp-perf-{Environment.ProcessId}-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        BaseStorage storage = cfg switch
        {
            "storage_sqlite" => new RDBStorage($"sqlite:///{Path.Combine(dir, "p.db")}"),
            "storage_journal" => new JournalStorage(new JournalFileBackend(Path.Combine(dir, "p.log"))),
            _ => new InMemoryStorage(),
        };
        try
        {
            var st = Optuna.CreateStudy(storage: storage, sampler: new RandomSampler(seed: 0));
            var sw0 = Stopwatch.StartNew();
            st.Optimize(Obj, nTrials: 1000);
            sw0.Stop();
            return new BenchRun(sw0.Elapsed.TotalSeconds, Bench.Fp(st.BestValue), "trials=1000");
        }
        finally
        {
            (storage as IDisposable)?.Dispose();
            Microsoft.Data.Sqlite.SqliteConnection.ClearAllPools();
            try { Directory.Delete(dir, true); } catch (IOException) { /* a leftover temp folder does not affect the timing */ }
        }
    });
    return;
}

var (make, trials) = Pick();
Bench.Run(cfg, () =>
{
    var study = Optuna.CreateStudy(sampler: make());
    var sw = Stopwatch.StartNew();
    study.Optimize(Obj, nTrials: trials);
    sw.Stop();
    return new BenchRun(sw.Elapsed.TotalSeconds, Bench.Fp(study.BestValue), $"trials={trials}");
});
Console.WriteLine($"C# {cfg}: peakWS={Process.GetCurrentProcess().PeakWorkingSet64 / 1048576}MB");
