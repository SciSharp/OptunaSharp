// Large studies (README "Large studies" table) and compact storage (README's memory paragraph, performance.md's table): plan
// 13's proof workloads on InMemoryStorage with and without compactFinishedTrials, against this checkout's src/.
// Usage: proof <tpe|motpe|cmaes|nsga2|random|qmc> <trials> <warmupTrials> <plain|compact>
// Workloads as in plan 13 section 2.1: 40 float parameters x0..x39 in [-10, 10], seed 0, default sampler settings.
// Single-objective: sum((x - 1)^2). Two objectives (motpe, nsga2): sum(x^2) and sum((x - 2)^2).
// Under bench/common/Bench.cs's protocol since 2026-11-23: warm-up runs of the same study at <warmupTrials> trials (a tenth of
// the measured size: it runs the same code, and Python Optuna's 1,000-trial studies take 38-197 s each, so full-size warm-up
// runs would triple their cost), never counted; then measured runs of the full study. Each run builds a new study; only
// Optimize is timed. Until 2026-11-23 the README took rep 0, the first run in the process, JIT included.
// "retained" is the heap the finished study keeps alive after a full collection, measured from just before the study was
// created; "opt_alloc" and "opt_gc" are the allocation and collections during Optimize alone (Bench's own alloc= and gc= also
// cover building the study and the hash); the hash covers every trial (number, state, each parameter's name and value bits,
// each value's bits), so equal hashes for plain and compact mean the switch is not observable in the results, and equal
// hashes against py/pbench.py's mean OptunaSharp ran Optuna's exact trials.
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

OptunaLogging.SetVerbosity(LogLevel.Error);
string cfg = args[0];
int trials = int.Parse(args[1]);
int warmupTrials = int.Parse(args[2]);
bool compact = args[3] == "compact";
bool jitOff = (Attribute.GetCustomAttribute(typeof(Study).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute)?.IsJITOptimizerDisabled ?? false;
if (jitOff)
    Console.WriteLine("WARNING: OptunaSharp was built with the JIT optimizer off");

string[] names = Enumerable.Range(0, 40).Select(i => "x" + i).ToArray();
bool multiObjective = cfg is "motpe" or "nsga2";
string mode = compact ? "compact" : "plain";

// Plain and compact runs of the same workload in one process would share warmed code; the runner gives each its own process.
Bench.Run($"{cfg}-{mode}-{trials}", () => One(warmupTrials), () => One(trials));

// One study of n trials: builds it untimed, times Optimize, then measures what the finished study retains.
BenchRun One(int n)
{
    long before = GC.GetTotalMemory(forceFullCollection: true);
    BaseSampler sampler = cfg switch
    {
        "tpe" or "motpe" => new TPESampler(seed: 0),
        "cmaes" => new CmaEsSampler(seed: 0),
        "nsga2" => new NSGAIISampler(seed: 0),
        "random" => new RandomSampler(seed: 0),
        "qmc" => new QMCSampler(seed: 0, warnIndependentSampling: false),
        _ => throw new ArgumentException("config must be tpe | motpe | cmaes | nsga2 | random | qmc"),
    };
    var storage = new InMemoryStorage(compactFinishedTrials: compact);
    var study = multiObjective
        ? Optuna.CreateStudy(storage: storage, sampler: sampler, directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize })
        : Optuna.CreateStudy(storage: storage, sampler: sampler);
    long allocated0 = GC.GetTotalAllocatedBytes(precise: false);
    int[] collections0 = { GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2) };
    TimeSpan paused0 = GC.GetTotalPauseDuration();
    var sw = Stopwatch.StartNew();
    if (multiObjective)
    {
        study.Optimize(t =>
        {
            double a = 0, b = 0;
            foreach (string name in names)
            {
                double v = t.SuggestFloat(name, -10, 10);
                a += v * v;
                b += (v - 2) * (v - 2);
            }
            return new[] { a, b };
        }, nTrials: n);
    }
    else
    {
        study.Optimize(t =>
        {
            double s = 0;
            foreach (string name in names)
            {
                double v = t.SuggestFloat(name, -10, 10);
                s += (v - 1) * (v - 1);
            }
            return s;
        }, nTrials: n);
    }
    double seconds = sw.Elapsed.TotalSeconds;
    long allocatedMb = (GC.GetTotalAllocatedBytes(precise: false) - allocated0) / 1048576;
    string gc = $"{GC.CollectionCount(0) - collections0[0]}/{GC.CollectionCount(1) - collections0[1]}/{GC.CollectionCount(2) - collections0[2]}";
    double paused = (GC.GetTotalPauseDuration() - paused0).TotalSeconds;
    long retained = GC.GetTotalMemory(forceFullCollection: true) - before;
    string hash = Hash(study);
    GC.KeepAlive(storage);
    return new BenchRun(seconds, hash, FormattableString.Invariant(
        $"trials={n} retained={retained / 1048576.0:F1}MB perTrial={retained / (double)n:F0}B opt_alloc={allocatedMb}MB opt_gc={gc} paused={paused:F3}s peakWS={Process.GetCurrentProcess().PeakWorkingSet64 / 1048576}MB"));
}

/// <summary>
/// The study's trial hash: the first 16 hex digits of a SHA-256 over every trial's number and state (little-endian int32),
/// each parameter's UTF-8 name, a zero byte and its value (a float's IEEE-754 bits as int64), then each objective value's bits.
/// </summary>
/// <param name="study">The finished study.</param>
/// <returns>16 upper-case hex digits; equal for two studies only if every trial's parameters and values are equal bit for bit.</returns>
/// <remarks>
/// py/pbench.py builds the same bytes in the same order, so OptunaSharp's and Optuna's hashes are comparable: equal hashes
/// prove the seeded trials identical. The bytes are hashed in chunks so a 100,000-trial study does not build one ~50 MB
/// buffer.
/// </remarks>
static string Hash(Study study)
{
    using var sha = SHA256.Create();
    var buffer = new List<byte>(1 << 16);
    foreach (FrozenTrial trial in study.GetTrials(deepcopy: false))
    {
        buffer.AddRange(BitConverter.GetBytes(trial.Number));
        buffer.AddRange(BitConverter.GetBytes((int)trial.State));
        foreach (var (name, value) in trial.Params)
        {
            buffer.AddRange(Encoding.UTF8.GetBytes(name));
            buffer.Add(0);
            buffer.AddRange(value switch
            {
                double d => BitConverter.GetBytes(BitConverter.DoubleToInt64Bits(d)),
                long l => BitConverter.GetBytes(l),
                _ => Encoding.UTF8.GetBytes(value?.ToString() ?? "None"),
            });
        }
        foreach (double v in trial.Values ?? (IReadOnlyList<double>)Array.Empty<double>())
            buffer.AddRange(BitConverter.GetBytes(BitConverter.DoubleToInt64Bits(v)));
        // Hash in chunks so a 100,000-trial study does not build one ~50 MB buffer.
        if (buffer.Count > 1 << 15)
        {
            sha.TransformBlock(buffer.ToArray(), 0, buffer.Count, null, 0);
            buffer.Clear();
        }
    }
    sha.TransformFinalBlock(buffer.ToArray(), 0, buffer.Count);
    return Convert.ToHexString(sha.Hash!)[..16];
}
