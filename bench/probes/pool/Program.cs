// Waiting-trials probe (README "Parallel trials", the sentence on trials that sleep; py/pool.py mirrors it): Optimize(nJobs)
// with an I/O-bound (sleeping) objective and nJobs above the core count. Ideal time = ceil(trials / jobs) * sleepMs.
// Under bench/common/Bench.cs's protocol since 2026-11-23 (warm-up runs never counted, then measured runs, each a fresh study);
// until then one run per process. OptunaSharp's default executor keeps its dedicated worker threads for 30 s between Optimize
// calls, so a measured run reuses the threads the warm-up started, the steady state of a long-lived process; Python's n_jobs
// starts a thread pool per optimize call either way.
// Usage: pool <jobs> <trials> <sleepMs>
using System.Diagnostics;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);
int jobs = args.Length > 0 ? int.Parse(args[0]) : 64;
int trials = args.Length > 1 ? int.Parse(args[1]) : 640;
int sleepMs = args.Length > 2 ? int.Parse(args[2]) : 100;
ThreadPool.GetMinThreads(out int minWorkers, out _);
double ideal = Math.Ceiling(trials / (double)jobs) * sleepMs / 1000.0;

IReadOnlyList<BenchRun> runs = Bench.Run($"jobs{jobs}", () =>
{
    // How many trials ran at once in this run: the proof that the workers really overlapped.
    int peak = 0, live = 0;
    var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 0));
    var sw = Stopwatch.StartNew();
    study.Optimize(t =>
    {
        int now = Interlocked.Increment(ref live);
        int seen;
        while (now > (seen = Volatile.Read(ref peak)) && Interlocked.CompareExchange(ref peak, now, seen) != seen) { }
        double x = t.SuggestFloat("x", -1, 1);
        Thread.Sleep(sleepMs);
        Interlocked.Decrement(ref live);
        return x * x;
    }, nTrials: trials, nJobs: jobs);
    sw.Stop();
    // The fingerprint is the trial count only: with nJobs > 1 the sampler's generator is reseeded per worker (Optuna's
    // reseed_rng, which OptunaSharp follows), so the best value varies between runs, while the work (a 100 ms sleep) does not.
    return new BenchRun(sw.Elapsed.TotalSeconds, study.Trials.Count.ToString(),
        FormattableString.Invariant($"jobs={jobs} trials={trials} ideal={ideal:F2}s peakConcurrent={peak}"));
});
Console.WriteLine(FormattableString.Invariant(
    $"C#: cores={Environment.ProcessorCount} poolMin={minWorkers} jobs={jobs} trials={trials} median={Bench.Median(runs):F3}s ideal={ideal:F2}s"));
