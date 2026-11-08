#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Serverless power tuning — multi-objective TPE (MOTPE) picks the memory size, CPU architecture, SQS batch
// size and Native-AOT setting of a queue-driven AWS-Lambda-style function, trading the cost per million
// messages against p99 end-to-end latency, with "the batch must fit in memory" as a constraint.
//
// source: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
//         (TPESampler handles several objectives — MOTPE — and constraints)
//         https://github.com/alexcasalboni/aws-lambda-power-tuning
//         (the well-known tool for exactly this trade-off, which sweeps memory sizes by brute force)
//
// Surrogate of the platform's documented pricing/performance rules: CPU share grows linearly with memory (1
// vCPU at 1,769 MB, up to 6 vCPUs); arm64 is 20 % cheaper per GB-second and ~8 % slower on this workload;
// each message costs 120 ms of CPU and a batch is processed in parallel up to the vCPU count; a cold start
// (2 % of invocations) costs 900 ms with the JIT or 180 ms with Native AOT; a batch waits for the queue to
// fill at 50 messages/s. Cost = GB-seconds × price + $0.20 per million requests. The constraint keeps
// batch × 12 MB + an 80 MB runtime under the configured memory. The Pareto front lists the trade-offs; the
// team picks the knee.
//
//   Run:  dotnet run examples/0_ServerlessPowerTuningMotpe.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var study = Optuna.CreateStudy(
    sampler: new TPESampler(seed: 12, nStartupTrials: 20),      // with two directions TPE runs as MOTPE
    directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },   // ($ per 1M messages, p99 ms)
    studyName: "lambda-power-tuning");
study.SetMetricNames(new[] { "usd_per_million", "p99_ms" });
study.Optimize(Objective, nTrials: 150);

var front = study.BestTrials.OrderBy(t => t.Values![0]).ToList();
Console.WriteLine($"{study.Trials.Count} configurations, {study.Trials.Count(t => t.Constraints.Values.All(c => c <= 0))} fit in memory; Pareto front: {front.Count}");
Console.WriteLine();
Console.WriteLine("  $/1M msgs   p99 ms   memory   arch     batch  AOT");
foreach (var t in front)
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0,10:F3} {1,8:F0} {2,7} MB  {3,-8} {4,5}  {5}",
        t.Values![0], t.Values[1], t.Params["memory_mb"], t.Params["arch"], t.Params["batch_size"], t.Params["native_aot"]));

/// <summary>Prices one function configuration and estimates its tail latency.</summary>
/// <param name="t">The trial proposing memory, architecture, batch size and AOT.</param>
/// <returns>USD per million messages and p99 end-to-end latency in milliseconds.</returns>
static IReadOnlyList<double> Objective(Trial t)
{
    long memory = t.SuggestInt("memory_mb", 128, 4096, log: true);
    string arch = t.SuggestCategorical("arch", new[] { "x86_64", "arm64" });
    long batch = t.SuggestInt("batch_size", 1, 50);
    bool aot = t.SuggestCategorical("native_aot", new[] { false, true });

    t.SetConstraint("memory_fit_mb", batch * 12 + 80 - memory);          // ≤ 0: the batch fits

    double vcpus = Math.Min(6.0, memory / 1769.0);
    double speed = arch == "arm64" ? 0.92 : 1.0;
    double cpuSeconds = batch * 0.120 / speed;
    double parallel = Math.Max(Math.Min(vcpus, batch), Math.Min(vcpus, 1.0));
    double durationS = 0.015 + cpuSeconds / parallel;                     // invocation overhead + work
    double coldS = aot ? 0.180 : 0.900;
    double pricePerGbS = arch == "arm64" ? 0.0000133334 : 0.0000166667;
    double invocations = 1e6 / batch;
    double cost = invocations * (memory / 1024.0) * (durationS + 0.02 * coldS) * pricePerGbS + invocations * 0.20 / 1e6;
    // p99: the slowest 1 % includes the cold starts (2 % of invocations) and the wait for the batch to fill.
    double fillWaitS = (batch - 1) / 50.0;
    double p99Ms = 1000 * (fillWaitS + durationS + coldS);
    return new[] { cost, p99Ms };
}
