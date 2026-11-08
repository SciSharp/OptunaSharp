#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:project ../src/OptunaSharp.Storage.Sqlite/OptunaSharp.Storage.Sqlite.csproj
#:property PublishAot=false

// Crash-tolerant workers on one SQLite study — three worker processes share `sqlite:///…` (Optuna's own
// schema), one of them is killed mid-trial, and heartbeats + RetryHeartbeatStaleTrialCallback turn its orphaned
// RUNNING trial into FAIL and re-run the same parameters — the setup a team uses to tune on spot/preemptible
// machines. If a Python with optuna is on PATH, Python reads the very same database at the end.
//
// source: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html
//         (`heartbeat_interval`, `grace_period`, `heartbeat_stale_trial_callback` — Optuna 4.9 renamed it
//         from `failed_trial_callback` / RetryFailedTrialCallback, which still work but warn)
//         https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RetryHeartbeatStaleTrialCallback.html
//         https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
//         (several processes optimizing one study through a shared storage)
//
// Real objective, Monte Carlo: a client's retry policy against a flaky downstream (2 % timeouts, a
// heavy-tailed latency) — max attempts, per-attempt timeout, base backoff and jitter — scored as the
// simulated p95 end-to-end latency of 4,000 requests, plus a heavy penalty when more than 0.1 % of requests
// still fail. Each evaluation also sleeps 150 ms to stand in for the real load test, so trials overlap
// across the workers. Worker 2 calls Environment.Exit mid-trial, as an OOM-kill or a preempted VM would.
//
//   Run:  dotnet run examples/0_CrashTolerantSqliteWorkers.cs

using System.Diagnostics;
using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

const string StudyName = "retry-policy";
string dbPath = args.Length > 1 ? args[1] : Path.Combine(Path.GetTempPath(), "optunasharp-retry-policy.db");
string url = "sqlite:///" + dbPath.Replace('\\', '/');

if (args.Length > 0 && args[0] == "--worker")
{
    RunWorker(url, StudyName, workerId: int.Parse(args[2], CultureInfo.InvariantCulture));   // child mode
    return;
}

OptunaLogging.SetVerbosity(LogLevel.Warning);
File.Delete(dbPath);                                        // a fresh study on every run of the example
using (var storage = OpenStorage(url))
    Optuna.CreateStudy(storage: storage, studyName: StudyName, direction: StudyDirection.Minimize);

// three worker processes, started together, all pointing at the same database file.
string dll = typeof(Policy).Assembly.Location;
var workers = Enumerable.Range(1, 3).Select(id =>
{
    var psi = new ProcessStartInfo("dotnet") { RedirectStandardError = true, RedirectStandardOutput = true };
    foreach (var a in new[] { "exec", dll, "--worker", dbPath, id.ToString(CultureInfo.InvariantCulture) })
        psi.ArgumentList.Add(a);
    return Process.Start(psi)!;
}).ToList();
foreach (var w in workers)
{
    w.StandardOutput.ReadToEnd();
    w.StandardError.ReadToEnd();
    w.WaitForExit();
}
Console.WriteLine("worker exit codes: " + string.Join(", ", workers.Select(w => w.ExitCode)));

// let the crashed worker's heartbeat go stale (grace period 3 s), then run one more worker in-process: its
// first trial notices the stale RUNNING trial, marks it FAIL, and the callback enqueues a retry of it.
Thread.Sleep(4000);
RunWorker(url, StudyName, workerId: 4);

using (var storage = OpenStorage(url))
{
    var study = Optuna.LoadStudy(StudyName, storage);
    Console.WriteLine();
    Console.WriteLine(" #   state      worker  p95 ms   retry of");
    foreach (var t in study.Trials)
        Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0,2}   {1,-9} {2,6} {3,8} {4,10}",
            t.Number, t.State, t.UserAttrs.GetValueOrDefault("worker"),
            t.Value is double v ? v.ToString("F1", CultureInfo.InvariantCulture) : "-",
            RetryHeartbeatStaleTrialCallback.RetriedTrialNumber(t)?.ToString(CultureInfo.InvariantCulture) ?? ""));
    Console.WriteLine();
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "best p95: {0:F1} ms with {1}", study.BestValue,
        string.Join(", ", study.BestParams.Select(kv => $"{kv.Key}={kv.Value}"))));
}
PythonReadsTheSameFile(dbPath, StudyName);

/// <summary>Opens the shared storage with heartbeats and the retry callback, as every worker must.</summary>
/// <param name="url">The SQLAlchemy-style SQLite URL.</param>
/// <returns>The storage (dispose it to close the connection).</returns>
static RDBStorage OpenStorage(string url) => new(
    url,
    heartbeatInterval: 1,                                        // a RUNNING trial proves it is alive every 1 s
    gracePeriod: 3,                                              // silent for 3 s → considered dead
    heartbeatStaleTrialCallback: new RetryHeartbeatStaleTrialCallback(maxRetry: 2));   // re-enqueue a dead trial (≤ 2 times)

/// <summary>One worker: loads the shared study and runs its share of trials.</summary>
/// <param name="url">The database URL.</param>
/// <param name="studyName">The study to join.</param>
/// <param name="workerId">This worker's id (worker 2 dies in the middle of its second trial).</param>
static void RunWorker(string url, string studyName, int workerId)
{
    OptunaLogging.SetVerbosity(LogLevel.Warning);
    using var storage = OpenStorage(url);
    var study = Optuna.LoadStudy(studyName, storage, sampler: new TPESampler(seed: workerId, nStartupTrials: 6));
    int started = 0;
    study.Optimize(t =>
    {
        t.SetUserAttr("worker", workerId);
        if (workerId == 2 && ++started == 2)
        {
            Thread.Sleep(300);
            Environment.Exit(137);                               // killed: no tell, no cleanup, trial stays RUNNING
        }
        return Policy.Objective(t);
    }, nTrials: 6);
}

/// <summary>Opens the same database from Python Optuna, when one is installed, and prints what it sees.</summary>
/// <param name="dbPath">The SQLite file.</param>
/// <param name="studyName">The study name.</param>
static void PythonReadsTheSameFile(string dbPath, string studyName)
{
    string code = "import optuna,sys; optuna.logging.set_verbosity(optuna.logging.ERROR); "
                  + $"s=optuna.load_study(study_name='{studyName}', storage='sqlite:///'+sys.argv[1]); "
                  + "print('python optuna', optuna.__version__, 'sees', len(s.trials), 'trials, best', round(s.best_value, 1), s.best_params)";
    try
    {
        var psi = new ProcessStartInfo("python") { RedirectStandardOutput = true, RedirectStandardError = true };
        foreach (var a in new[] { "-c", code, dbPath.Replace('\\', '/') })
            psi.ArgumentList.Add(a);
        using var p = Process.Start(psi)!;
        string output = p.StandardOutput.ReadToEnd();
        p.StandardError.ReadToEnd();
        p.WaitForExit();
        Console.WriteLine(p.ExitCode == 0 ? output.Trim() : "(python found, but `import optuna` failed — skipped)");
    }
    catch (System.ComponentModel.Win32Exception)
    {
        Console.WriteLine("(no python on PATH — skipped the cross-language read)");
    }
}

/// <summary>The retry policy under test and its Monte Carlo load test.</summary>
static class Policy
{
    /// <summary>Simulates 4,000 requests through the trial's retry policy.</summary>
    /// <param name="t">The trial proposing the policy.</param>
    /// <returns>p95 latency in ms, plus 10,000 ms per percent of requests that still failed above 0.1 %.</returns>
    public static double Objective(Trial t)
    {
        long attempts = t.SuggestInt("max_attempts", 1, 5);
        double timeoutMs = t.SuggestFloat("timeout_ms", 50, 2000, log: true);   // per-attempt deadline
        double backoffMs = t.SuggestFloat("backoff_ms", 1, 500, log: true);     // base of the exponential backoff
        string jitter = t.SuggestCategorical("jitter", new[] { "none", "full", "equal" });
        Thread.Sleep(150);                                                      // the "load test" takes a while

        var rng = new Random(99);                                               // common random numbers
        var latencies = new double[4000];
        int failed = 0;
        for (int r = 0; r < latencies.Length; r++)
        {
            double total = 0;
            bool ok = false;
            for (int a = 0; a < attempts && !ok; a++)
            {
                if (a > 0)
                {
                    double cap = backoffMs * Math.Pow(2, a - 1);
                    total += jitter switch { "full" => rng.NextDouble() * cap, "equal" => cap / 2 + rng.NextDouble() * cap / 2, _ => cap };
                }
                // the downstream: 2 % of calls hang; the rest are log-normal around 40 ms with a heavy tail.
                double service = rng.NextDouble() < 0.02 ? double.PositiveInfinity
                    : Math.Exp(Math.Log(40) + 0.6 * Gaussian(rng));
                if (service <= timeoutMs)
                {
                    total += service;
                    ok = true;
                }
                else
                    total += timeoutMs;
            }
            failed += ok ? 0 : 1;
            latencies[r] = total;
        }
        Array.Sort(latencies);
        double p95 = latencies[(int)(0.95 * latencies.Length)];
        double failPct = 100.0 * failed / latencies.Length;
        t.SetUserAttr("fail_pct", failPct);
        return p95 + 10_000 * Math.Max(0, failPct - 0.1);
    }

    /// <summary>A standard normal draw (Box–Muller).</summary>
    /// <param name="rng">The random stream.</param>
    /// <returns>A N(0, 1) variate.</returns>
    private static double Gaussian(Random rng) =>
        Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
}
