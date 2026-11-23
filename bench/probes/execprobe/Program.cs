// Table B of benchmark 3 ("executor alternatives", README "Executors"), regenerated 2026-11-08 from the 2026-10-29 probe of
// session 6c824873 (execprobe/Program.cs, recovered from its transcript): the real Optimize(nJobs) under each way of running
// trials. The workloads and their measurement code are the original's; only the executor switch changed.
// On 2026-10-29 one experimental Core build switched executors through an environment variable. Now:
//   perTrial : the code from before 9cfcb12 (a new OS thread per trial) -> built WITHOUT the HEAD symbol, against a snapshot of
//              9cfcb12^ (f42fde7);
//   reused   : HEAD's default, TrialExecutorKind.DedicatedThreads (OptunaSharp's own threads, reused across trials and Optimize
//              calls). The 2026-10-29 "reused" column was a prototype that joined its nJobs threads at the end of every call;
//              the shipped executor keeps them for 30 s, so manycalls is not comparable;
//   pool     : HEAD's TrialExecutorKind.ThreadPool.
// Since 2026-11-23 soak, tpe and manycalls run under bench/common/Bench.cs's protocol (warm-up runs never counted, then
// measured runs; the parser turns seconds into trials/s or calls/s). starve does not: its subject is a thread pool that has
// not grown yet, which a second run in the same process would not see, so it warms the JIT with a short non-blocking study and
// measures once per process; the runner's fresh processes give the samples.
// Usage: execprobe soak | tpe | manycalls | starve [multiplier of the pool's minimum worker threads, default 2]
using System.Diagnostics;
using OptunaSharp;
using OptunaSharp.Samplers;

OptunaSharp.Logging.OptunaLogging.SetVerbosity(OptunaSharp.Logging.LogLevel.Error);
string executor = Environment.GetEnvironmentVariable("OPTUNASHARP_NJOBS_EXECUTOR") ?? "default";
#if HEAD
// The process-wide default is the lowest of the executor's four levels; nothing else in this probe sets one.
TrialExecutorOptions.Default = executor switch
{
    "reused" or "default" => new TrialExecutorOptions { Executor = TrialExecutorKind.DedicatedThreads },
    "pool" => new TrialExecutorOptions { Executor = TrialExecutorKind.ThreadPool },
    _ => throw new ArgumentException($"executor '{executor}' needs the pre-9cfcb12 build (old); HEAD has reused | pool"),
};
#else
if (executor is not ("perTrial" or "default"))
    throw new ArgumentException($"the pre-9cfcb12 build only has perTrial, not '{executor}'");
#endif
bool jitOff = (Attribute.GetCustomAttribute(typeof(Study).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute)?.IsJITOptimizerDisabled ?? false;
string mode = args[0];
ThreadPool.GetMinThreads(out int minWorkers, out _);
Console.WriteLine($"[{executor}] JIT optimizer disabled: {jitOff}; pool minimum {minWorkers}");

switch (mode)
{
    case "soak":
    {
        // the P3-4 workload: Random sampler, trivial objective, in-memory storage
        var runs = Bench.Run("soak", () =>
        {
            var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 1));
            var sw = Stopwatch.StartNew();
            study.Optimize(t => { double x = t.SuggestFloat("x", -1, 1); return x * x; }, nTrials: 5000, nJobs: 8);
            sw.Stop();
            return new BenchRun(sw.Elapsed.TotalSeconds, study.Trials.Count.ToString(), "count=5000");
        });
        Console.WriteLine($"{executor,-8} soak random nJobs=8: {5000 / Bench.Median(runs),7:F0} trials/s (median of {runs.Count})");
        break;
    }
    case "tpe":
    {
        var runs = Bench.Run("tpe", () =>
        {
            var study = Optuna.CreateStudy(sampler: new TPESampler(seed: 1));
            var sw = Stopwatch.StartNew();
            study.Optimize(t => { double x = t.SuggestFloat("x", -1, 1); return x * x; }, nTrials: 2000, nJobs: 8);
            sw.Stop();
            return new BenchRun(sw.Elapsed.TotalSeconds, study.Trials.Count.ToString(), "count=2000");
        });
        Console.WriteLine($"{executor,-8} soak tpe nJobs=8: {2000 / Bench.Median(runs),7:F0} trials/s (median of {runs.Count})");
        break;
    }
    case "manycalls":
    {
        // many short Optimize calls, e.g. one small batch per request in a service
        var runs = Bench.Run("manycalls", () =>
        {
            var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 1));
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < 200; i++)
                study.Optimize(t => { double x = t.SuggestFloat("x", -1, 1); return x * x; }, nTrials: 8, nJobs: 8);
            sw.Stop();
            return new BenchRun(sw.Elapsed.TotalSeconds, study.Trials.Count.ToString(), "count=200");
        });
        Console.WriteLine($"{executor,-8} manycalls 200 x Optimize(nTrials: 8, nJobs: 8): {200 / Bench.Median(runs),6:F0} calls/s (median of {runs.Count})");
        break;
    }
    case "starve":
    {
        // A long blocking study in a process that also serves other work: how late does that other work start? The JIT is
        // warmed first with a short study that never blocks a pool thread, so the pool stays at its minimum for the measured run.
        Warm();
        // optional multiplier of the pool's minimum worker threads (default 2): 1 = "nJobs equals the pool's size"
        int nJobs = minWorkers * (args.Length > 1 ? int.Parse(args[1]) : 2);
        Bench.Once("starve", "measure", 0, () =>
        {
            var latencies = new List<double>();
            var stop = new ManualResetEventSlim();
            var prober = new Thread(() =>
            {
                while (!stop.IsSet)
                {
                    long t0 = Stopwatch.GetTimestamp();
                    Task.Run(() => { lock (latencies) latencies.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds); }).Wait();
                    Thread.Sleep(20);
                }
            }) { IsBackground = true };
            prober.Start();
            Thread.Sleep(200);
            var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 1));
            var sw = Stopwatch.StartNew();
            study.Optimize(t => { t.SuggestFloat("x", -1, 1); Thread.Sleep(1500); return 0.0; }, nTrials: nJobs * 2, nJobs: nJobs);
            sw.Stop();
            stop.Set();
            prober.Join();
            latencies.Sort();
            return new BenchRun(sw.Elapsed.TotalSeconds, study.Trials.Count.ToString(), FormattableString.Invariant(
                $"njobs={nJobs} ideal=3.0s p50={latencies[latencies.Count / 2]:F1}ms p95={latencies[(int)(latencies.Count * 0.95)]:F1}ms max={latencies[^1]:F1}ms"));
        });
        break;
    }
    default:
        throw new ArgumentException("mode must be soak | tpe | manycalls | starve");
}

/// <summary>
/// Runs a short parallel study (50 random trials, <c>nJobs: 4</c>), so starve's single timed run pays for neither JIT tiering
/// of the executor's code nor thread start-up.
/// </summary>
/// <remarks>
/// A non-blocking objective at <c>nJobs: 4</c> keeps the ThreadPool at its minimum size, the state starve measures; a warm-up
/// that blocked pool threads would grow the pool and erase the effect being measured. Any exception from the study propagates
/// and ends the process with a non-zero exit code, which the parser drops.
/// </remarks>
static void Warm() =>
    Optuna.CreateStudy(sampler: new RandomSampler(seed: 0)).Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 50, nJobs: 4);
