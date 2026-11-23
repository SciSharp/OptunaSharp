// The measurement protocol every C# probe shares (bench/README.md, "Protocol"), compiled into each probe with
// <Compile Include>. py/benchlib.py is its Python twin and prints the same line format, so bench/parse.py reads OptunaSharp,
// Optuna and Rustuna alike.
//
// Why a protocol at all: until 2026-11-23 each workload was timed once, in a fresh process, so OptunaSharp's numbers were
// mostly .NET's tiered JIT compiling OptunaSharp and NumSharp (about four fifths of the short Samplers rows on the worker),
// and Optuna's included its lazy imports (torch, cmaes, SciPy's QMC). A comparison of first runs measures start-up, not the
// libraries. Here every process first repeats the workload until it is warm, discards those runs, and then measures.
using System.Diagnostics;
using System.Globalization;
using System.Runtime;

/// <summary>
/// One timed run of a workload, as the workload's own code measured it.
/// </summary>
/// <param name="Seconds">
/// Wall seconds of the measured part only (for an optimization, <c>Optimize</c>; creating the study and the sampler is not
/// timed, as in the probes before 2026-11-23).
/// </param>
/// <param name="Fingerprint">
/// A token that is the same for every run that did the same work: the best value, a trial hash, a trial count. The parser
/// checks that every measured run of a seeded workload printed one fingerprint, so a run that silently did different work
/// cannot be averaged in. It must not contain spaces.
/// </param>
/// <param name="Extra">Further <c>key=value</c> fields to print after the fingerprint (memory, per-block times), or empty.</param>
internal readonly record struct BenchRun(double Seconds, string Fingerprint, string Extra = "");

/// <summary>
/// The warm-up and measuring budgets of one process, read from <c>BENCH_*</c> environment variables so the runner sets
/// them once for every probe of every library.
/// </summary>
/// <param name="WarmupSeconds">Warm up for at least this many seconds (<c>BENCH_WARMUP_SECONDS</c>, default 5).</param>
/// <param name="WarmupRuns">...and at least this many runs (<c>BENCH_WARMUP_RUNS</c>, default 2).</param>
/// <param name="TieringRuns">
/// ...and, unless <paramref name="MaxWarmupSeconds"/> passes first, at least this many runs, the .NET runtime's own need
/// (<c>BENCH_DOTNET_TIERING_RUNS</c>, default 70; the remarks say why; py/benchlib.py has no such rule).
/// </param>
/// <param name="QuietShare">
/// ...then until the runtime compiled nothing during the last <c>ceil(QuietShare x warm-up runs)</c> runs, at least the last
/// one (<c>BENCH_WARMUP_QUIET_SHARE</c>, default 0.25; 0 turns the rule off).
/// </param>
/// <param name="MaxWarmupRuns">Stop warming up after this many runs even if the runtime is not quiet yet (<c>BENCH_WARMUP_MAX_RUNS</c>, default 500).</param>
/// <param name="MaxWarmupSeconds">...or once this many seconds have passed with the first two minimums met (<c>BENCH_WARMUP_MAX_SECONDS</c>, default 60).</param>
/// <param name="MeasureRuns">Measure at least this many runs (<c>BENCH_MEASURE_RUNS</c>, default 3).</param>
/// <param name="MeasureSeconds">...then keep measuring until this many measured seconds (<c>BENCH_MEASURE_SECONDS</c>, default 10)...</param>
/// <param name="MaxMeasureRuns">...or this many runs, whichever comes first (<c>BENCH_MAX_RUNS</c>, default 30).</param>
/// <remarks>
/// <para>
/// Why the runtime's own count and not a fixed time: .NET's tiered compilation with Dynamic PGO moves a method from tier 0
/// to instrumented code after 30 calls and to tier 1 after 30 more, so a method the workload calls once per run (creating
/// the study, finishing it) settles only around its 61st to 63rd run, with the JIT silent for the 30 runs between the two
/// steps. On the worker on 2026-11-23, OptunaSharp's 5,000-trial Random row was still compiling 16-67 methods per measured
/// run (up to 72 ms of JIT beside a 70 ms run, and runs up to 9 % slower) in runs 59-68, after a fixed 5-s warm-up of 58
/// runs; the TPE rows compiled 60 methods in runs 16-17, 2-4 % slower. No quiet streak can tell "settled" from "between
/// the steps" without being longer than the gap, so the warm-up runs past both steps (70 runs) whenever that fits in the
/// time cap, and then waits for its last quarter to have compiled nothing, which also catches methods called more rarely.
/// </para>
/// <para>
/// The time cap keeps long workloads (a second or more per run) from warming up for minutes. For those, a method called
/// once per run is a vanishing share of the run, and its first step comes after about 30 runs, which a 60-s warm-up plus
/// 10 measured seconds reaches only for runs of about 1.9-2.3 s (its second step only for about 0.95-1.15 s): the parser's
/// <c>jit</c> check (JIT time over 1 % of the measured time) reports any process that compiled while measuring. Python has
/// no JIT, but its first run imports lazily (torch for GP, cmaes for CMA-ES, SciPy's QMC), and py/benchlib.py applies the
/// quiet rule to imports. The 30-run cap lets a short workload fill its 10 measured seconds with more runs (10 before
/// 2026-11-23 10:45), so its per-process median is firmer at no cost to the long ones.
/// </para>
/// </remarks>
internal sealed record BenchSettings(double WarmupSeconds, int WarmupRuns, int TieringRuns, double QuietShare, int MaxWarmupRuns, double MaxWarmupSeconds, int MeasureRuns, double MeasureSeconds, int MaxMeasureRuns)
{
    /// <summary>Reads the settings from the environment, with the defaults above for any variable that is not set.</summary>
    /// <returns>The settings.</returns>
    /// <exception cref="ArgumentException">
    /// A variable is set but is not a number, or a count or time is negative, a minimum exceeds its maximum, or the quiet
    /// share is outside 0..1.
    /// </exception>
    internal static BenchSettings FromEnvironment()
    {
        var s = new BenchSettings(
            Number("BENCH_WARMUP_SECONDS", 5),
            (int)Number("BENCH_WARMUP_RUNS", 2),
            (int)Number("BENCH_DOTNET_TIERING_RUNS", 70),
            Number("BENCH_WARMUP_QUIET_SHARE", 0.25),
            (int)Number("BENCH_WARMUP_MAX_RUNS", 500),
            Number("BENCH_WARMUP_MAX_SECONDS", 60),
            (int)Number("BENCH_MEASURE_RUNS", 3),
            Number("BENCH_MEASURE_SECONDS", 10),
            (int)Number("BENCH_MAX_RUNS", 30));
        // A minimum above its maximum would make the loops below stop at the maximum without saying so, which would print
        // fewer runs than the README's method promises; refuse instead.
        if (s.WarmupSeconds < 0 || s.WarmupRuns < 0 || s.TieringRuns < 0 || s.MeasureRuns < 1 || s.MeasureSeconds < 0 ||
            s.WarmupRuns > s.MaxWarmupRuns || s.TieringRuns > s.MaxWarmupRuns || s.WarmupSeconds > s.MaxWarmupSeconds ||
            s.MeasureRuns > s.MaxMeasureRuns || s.QuietShare < 0 || s.QuietShare > 1)
            throw new ArgumentException($"inconsistent BENCH_* settings: {s}");
        return s;
    }

    /// <summary>Whether the warm-up's trailing runs were quiet enough to stop warming up.</summary>
    /// <param name="quiet">One flag per warm-up run so far, in order: <see langword="true"/> when nothing was compiled during it.</param>
    /// <returns>
    /// <see langword="true"/> when the last <c>max(1, ceil(QuietShare x runs))</c> runs were all quiet, or when the rule is
    /// off (<see cref="QuietShare"/> 0); <see langword="false"/> with no runs yet, since nothing has shown the runtime quiet.
    /// </returns>
    internal bool QuietTail(IReadOnlyList<bool> quiet)
    {
        if (QuietShare <= 0)
            return true;
        int need = Math.Max(1, (int)Math.Ceiling(QuietShare * quiet.Count));
        if (quiet.Count < need)
            return false;
        for (int i = quiet.Count - need; i < quiet.Count; i++)
            if (!quiet[i])
                return false;
        return true;
    }

    /// <summary>One numeric environment variable.</summary>
    /// <param name="name">The variable's name.</param>
    /// <param name="fallback">The value when the variable is not set or empty.</param>
    /// <returns>The value, parsed with the invariant culture.</returns>
    /// <exception cref="ArgumentException">The variable is set but is not a number.</exception>
    private static double Number(string name, double fallback)
    {
        string? text = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(text))
            return fallback;
        return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double v)
            ? v
            : throw new ArgumentException($"{name}={text} is not a number");
    }
}

/// <summary>
/// Runs a workload under the protocol: warm-up runs that are printed but never counted, then measured runs, one line each.
/// </summary>
/// <remarks>
/// <para>
/// Every run starts from a collected heap: a full blocking collection and the pending finalizers run before it, outside its
/// timer, so no run pays for the previous run's garbage. Each line also states the JIT activity during the run (methods
/// compiled and compilation time, from <see cref="JitInfo"/>), the collections and the bytes allocated: a measured run that
/// still compiles methods was not warm, and the parser reports it.
/// </para>
/// <para>
/// Line format (bench/parse.py reads it): <c>bench &lt;label&gt; &lt;warmup|measure&gt; &lt;index&gt; &lt;seconds&gt;s
/// jit=+&lt;methods&gt;/&lt;ms&gt;ms gc=&lt;g0&gt;/&lt;g1&gt;/&lt;g2&gt; alloc=&lt;MB&gt;MB fp=&lt;fingerprint&gt; [extra]</c>, then
/// <c>bench &lt;label&gt; done warmup=&lt;n&gt; measure=&lt;m&gt; stop=&lt;quiet|maxruns|maxseconds&gt; quiet_tail=&lt;k&gt;</c>:
/// why the warm-up ended (the runtime was quiet, or a cap) and how many of its last runs compiled nothing.
/// </para>
/// </remarks>
internal static class Bench
{
    /// <summary>Whether this process already printed its <c>bench settings</c> line.</summary>
    private static bool _settingsPrinted;

    /// <summary>Warms up and measures a workload whose warm-up run is the measured run itself.</summary>
    /// <param name="label">The workload's name within this process; no spaces (the parser splits on them).</param>
    /// <param name="workload">One run; it times its own measured part and returns it.</param>
    /// <param name="settings">The budgets, or <see langword="null"/> for <see cref="BenchSettings.FromEnvironment"/>.</param>
    /// <returns>The measured runs, in order (never the warm-up runs), for a probe that prints its own summary line.</returns>
    /// <exception cref="ArgumentException">The label contains white space, or the environment's settings are invalid.</exception>
    /// <remarks>Any exception the workload throws propagates: a failed run ends the process with a non-zero exit code, which the runner records and the parser drops.</remarks>
    internal static IReadOnlyList<BenchRun> Run(string label, Func<BenchRun> workload, BenchSettings? settings = null) =>
        Run(label, workload, workload, settings);

    /// <summary>Warms up with one workload and measures another (a smaller warm-up study for a long measured one).</summary>
    /// <param name="label">The workload's name within this process; no spaces (the parser splits on them).</param>
    /// <param name="warmup">One warm-up run. It must exercise the same code as <paramref name="measure"/>, only less of it.</param>
    /// <param name="measure">One measured run; it times its own measured part and returns it.</param>
    /// <param name="settings">The budgets, or <see langword="null"/> for <see cref="BenchSettings.FromEnvironment"/>.</param>
    /// <returns>The measured runs, in order (never the warm-up runs), for a probe that prints its own summary line.</returns>
    /// <exception cref="ArgumentException">The label contains white space, or the environment's settings are invalid.</exception>
    /// <remarks>
    /// A shorter warm-up is only for studies whose measured run takes tens of seconds or minutes (Python Optuna's
    /// 1,000-trial TPE studies take 38-197 s): the same sampler, objective and seed at a tenth of the trials compiles and
    /// imports the same code. The measured runs then show whether that was enough (the parser's steady-state check).
    /// </remarks>
    internal static IReadOnlyList<BenchRun> Run(string label, Func<BenchRun> warmup, Func<BenchRun> measure, BenchSettings? settings = null)
    {
        if (label.Any(char.IsWhiteSpace))
            throw new ArgumentException($"label '{label}' contains white space", nameof(label));
        var s = settings ?? BenchSettings.FromEnvironment();
        if (!_settingsPrinted)
        {
            _settingsPrinted = true;
            Console.WriteLine(FormattableString.Invariant(
                $"bench settings warmup>={s.WarmupSeconds}s,{s.WarmupRuns}runs,tiering{s.TieringRuns}runs,quiet{s.QuietShare}(max{s.MaxWarmupRuns}runs,{s.MaxWarmupSeconds}s) measure>={s.MeasureRuns}runs,until{s.MeasureSeconds}s(max{s.MaxMeasureRuns})"));
        }

        // Warm-up: at least WarmupRuns runs AND WarmupSeconds of wall time; then at least TieringRuns runs and a quiet tail
        // (the last QuietShare of the runs compiled nothing; BenchSettings' remarks), unless MaxWarmupRuns or MaxWarmupSeconds
        // is reached first. The first two minimums win over the time cap, so a workload longer than the cap still gets its
        // WarmupRuns runs.
        var phase = Stopwatch.StartNew();
        var quiet = new List<bool>();
        string stop;
        while (true)
        {
            bool minimums = quiet.Count >= s.WarmupRuns && phase.Elapsed.TotalSeconds >= s.WarmupSeconds;
            if (minimums && quiet.Count >= s.TieringRuns && s.QuietTail(quiet))
            {
                stop = "quiet";
                break;
            }
            if (quiet.Count >= s.MaxWarmupRuns)
            {
                stop = "maxruns";
                break;
            }
            if (minimums && phase.Elapsed.TotalSeconds >= s.MaxWarmupSeconds)
            {
                stop = "maxseconds";
                break;
            }
            // Counted around the whole iteration, not only the timed body Once reports: the background tier-1 compiler
            // can finish a method while Once collects the heap, and that is runtime activity all the same.
            long compiled0 = JitInfo.GetCompiledMethodCount(currentThread: false);
            Once(label, "warmup", quiet.Count, warmup);
            quiet.Add(JitInfo.GetCompiledMethodCount(currentThread: false) == compiled0);
        }
        int quietTail = 0;
        for (int i = quiet.Count - 1; i >= 0 && quiet[i]; i--)
            quietTail++;
        int warm = quiet.Count;

        // Measured: at least MeasureRuns runs, then on until MeasureSeconds of measured time or MaxMeasureRuns runs.
        var runs = new List<BenchRun>();
        double total = 0;
        while (runs.Count < s.MeasureRuns || (total < s.MeasureSeconds && runs.Count < s.MaxMeasureRuns))
        {
            BenchRun run = Once(label, "measure", runs.Count, measure);
            runs.Add(run);
            total += run.Seconds;
        }

        Console.WriteLine($"bench {label} done warmup={warm} measure={runs.Count} stop={stop} quiet_tail={quietTail}");
        return runs;
    }

    /// <summary>The median of the measured runs' seconds, for a probe's own summary line.</summary>
    /// <param name="runs">Measured runs; at least one.</param>
    /// <returns>The median (the mean of the two middle values for an even count).</returns>
    /// <exception cref="ArgumentException"><paramref name="runs"/> is empty.</exception>
    /// <remarks>Only for human-readable summaries; bench/parse.py recomputes every statistic from the per-run lines.</remarks>
    internal static double Median(IReadOnlyList<BenchRun> runs)
    {
        if (runs.Count == 0)
            throw new ArgumentException("no runs", nameof(runs));
        double[] s = runs.Select(r => r.Seconds).Order().ToArray();
        return s.Length % 2 == 1 ? s[s.Length / 2] : (s[s.Length / 2 - 1] + s[s.Length / 2]) / 2;
    }

    /// <summary>Runs a workload once from a collected heap and prints its line.</summary>
    /// <param name="label">The workload's name.</param>
    /// <param name="phase"><c>warmup</c> or <c>measure</c>.</param>
    /// <param name="index">The run's index within its phase.</param>
    /// <param name="body">The run.</param>
    /// <returns>What the run returned.</returns>
    /// <remarks>
    /// Public to the probes for the one workload that must not repeat in a process (execprobe's <c>starve</c>, whose subject is
    /// a thread pool that has not grown yet): it prints a single <c>measure</c> line in the same format.
    /// </remarks>
    internal static BenchRun Once(string label, string phase, int index, Func<BenchRun> body)
    {
        // Outside the timer: the previous run's garbage and finalizers, so this run starts from the same heap state.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        long methods0 = JitInfo.GetCompiledMethodCount(currentThread: false);
        TimeSpan jit0 = JitInfo.GetCompilationTime(currentThread: false);
        int g0 = GC.CollectionCount(0), g1 = GC.CollectionCount(1), g2 = GC.CollectionCount(2);
        long allocated0 = GC.GetTotalAllocatedBytes(precise: false);

        BenchRun run = body();

        long methods = JitInfo.GetCompiledMethodCount(currentThread: false) - methods0;
        double jitMs = (JitInfo.GetCompilationTime(currentThread: false) - jit0).TotalMilliseconds;
        double allocMb = (GC.GetTotalAllocatedBytes(precise: false) - allocated0) / 1048576.0;
        string extra = string.IsNullOrEmpty(run.Extra) ? "" : " " + run.Extra;
        Console.WriteLine(FormattableString.Invariant(
            $"bench {label} {phase} {index} {run.Seconds:F6}s jit=+{methods}/{jitMs:F1}ms gc={GC.CollectionCount(0) - g0}/{GC.CollectionCount(1) - g1}/{GC.CollectionCount(2) - g2} alloc={allocMb:F1}MB fp={run.Fingerprint}{extra}"));
        return run;
    }

    /// <summary>The round-trip text of a double, the fingerprint form of a best value (equal text = equal bits).</summary>
    /// <param name="value">The value.</param>
    /// <returns>Its shortest round-trip representation, invariant culture.</returns>
    internal static string Fp(double value) => value.ToString("R", CultureInfo.InvariantCulture);
}
