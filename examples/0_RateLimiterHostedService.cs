#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:package Microsoft.Extensions.Hosting@10.0.12
#:property PublishAot=false

// Rate-limiter tuning inside a .NET Generic Host — a BackgroundService keeps re-tuning an API gateway's
// token-bucket limiter against the latest traffic, with OptunaSharp wired through dependency injection
// (storage, a per-study sampler factory, the pruner and the study factory are all registered services) and
// the optimization cancelled cleanly when the host shuts down.
//
// source: https://learn.microsoft.com/en-us/dotnet/core/extensions/workers
//         (worker services: BackgroundService, cancellation on host shutdown)
//         https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
//         (a long-lived study in a shared storage that workers keep adding trials to)
//         https://en.wikipedia.org/wiki/Token_bucket
//
// Real objective, discrete-event simulation: a minute of bursty API traffic (a Poisson base load plus
// flash-crowd bursts, regenerated every round to stand in for fresh telemetry) hits a gateway that admits a
// request when a token is available, queues it (up to a limit) otherwise, and sheds it when the queue is
// full; the backend serves 400 requests/s. The knobs: bucket capacity, refill rate, queue limit. The score
// is the p99 latency of admitted requests plus a 5 ms penalty per 0.1 % of shed requests. Each round is its
// own study on its own traffic window, warm-started with the deployed configuration (`enqueue_trial`). The
// host runs for about eight seconds; then StopAsync cancels the token, the trial in flight finishes, and
// Optimize throws OperationCanceledException, which the service treats as a normal shutdown.
//
//   Run:  dotnet run examples/0_RateLimiterHostedService.cs

using System.Globalization;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using OptunaSharp;
using OptunaSharp.Extensions;
using OptunaSharp.Logging;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

OptunaLogging.SetVerbosity(OptunaSharp.Logging.LogLevel.Warning);   // OptunaSharp's own per-trial lines; the host logs ours

var builder = Host.CreateApplicationBuilder(args);
builder.Logging.SetMinimumLevel(Microsoft.Extensions.Logging.LogLevel.Information);
builder.Services
    .AddOptunaStorage(new InMemoryStorage())                                        // one store for the app
    .AddOptunaSampler(_ => new TPESampler(seed: 6, nStartupTrials: 10))             // a fresh sampler per study
    .AddOptunaPruner(new NopPruner())
    .AddOptunaStudyFactory(o => { o.DefaultDirection = StudyDirection.Minimize; o.LoadIfExists = true; })
    .AddHostedService<LimiterTuner>();
using var host = builder.Build();

await host.StartAsync();
await Task.Delay(TimeSpan.FromSeconds(8));                                        // the app runs a while…
await host.StopAsync();                                                           // …then shuts down

/// <summary>The background service that keeps tuning the limiter until the host stops.</summary>
/// <param name="factory">The DI-provided study factory (storage, sampler and pruner come from the container).</param>
/// <param name="log">The host's logger.</param>
sealed class LimiterTuner(IOptunaStudyFactory factory, ILogger<LimiterTuner> log) : BackgroundService
{
    /// <summary>Optimizes one study per round against fresh traffic until cancelled.</summary>
    /// <param name="stoppingToken">Signalled when the host begins shutting down.</param>
    /// <returns>A task that completes when tuning has stopped.</returns>
    protected override Task ExecuteAsync(CancellationToken stoppingToken) => Task.Run(() =>
    {
        // one study per round: each round scores against ITS traffic window, so values are comparable
        // within a study (mixing windows in one study would compare different traffic). Each round starts
        // from the configuration currently deployed, enqueued as its first trial (a warm start).
        OptunaSharp.Collections.OrderedDict<string, object?>? deployed = null;
        int round = 0;
        try
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                var traffic = Gateway.Traffic(seconds: 60, seed: round);           // "the latest minute"
                var study = factory.CreateStudy(studyName: $"gateway-limiter-round-{round}");
                if (deployed is not null)
                    study.EnqueueTrial(deployed);
                study.Optimize(t => Objective(t, traffic), nTrials: 40, ct: stoppingToken);
                var best = study.BestTrial;
                double baseline = deployed is null ? double.NaN : study.Trials[0].Value ?? double.NaN;
                deployed = new OptunaSharp.Collections.OrderedDict<string, object?>(best.Params);
                log.LogInformation("round {Round}: deployed config scored {Baseline:F1} ms on the new traffic; tuned to {Best:F1} ms (capacity={Cap}, refill={Rate:F0}/s, queue={Queue})",
                    round++, baseline, best.Value, best.Params["capacity"], best.Params["refill_per_s"], best.Params["queue_limit"]);
                stoppingToken.WaitHandle.WaitOne(TimeSpan.FromMilliseconds(500));   // wait for the next telemetry window
            }
        }
        catch (OperationCanceledException)
        {
            // expected on shutdown: the trial in flight was allowed to finish, nothing is lost.
            log.LogInformation("host stopping: tuning cancelled during round {Round}; the last deployed limiter stays live", round);
        }
    }, CancellationToken.None);

    /// <summary>Simulates the gateway with one limiter configuration.</summary>
    /// <param name="t">The trial proposing the limiter.</param>
    /// <param name="traffic">Arrival times of the requests, in seconds.</param>
    /// <returns>p99 latency of admitted requests plus the shedding penalty, in ms.</returns>
    private static double Objective(Trial t, double[] traffic)
    {
        long capacity = t.SuggestInt("capacity", 1, 2000, log: true);         // burst allowance
        double refill = t.SuggestFloat("refill_per_s", 100, 1000, log: true);   // sustained admit rate
        long queue = t.SuggestInt("queue_limit", 0, 5000);                     // waiting room before shedding
        var (p99Ms, shedFrac) = Gateway.Simulate(traffic, capacity, refill, queue, serviceRate: 400);
        return p99Ms + 5.0 * (shedFrac * 1000);
    }
}

/// <summary>Traffic generation and the gateway simulation.</summary>
static class Gateway
{
    /// <summary>Generates request arrival times: Poisson base load plus flash-crowd bursts.</summary>
    /// <param name="seconds">Length of the window.</param>
    /// <param name="seed">Traffic seed.</param>
    /// <returns>Sorted arrival times in seconds.</returns>
    public static double[] Traffic(int seconds, int seed)
    {
        var rng = new Random(seed);   // traffic generation only — samplers never use System.Random
        var arrivals = new List<double>();
        for (double tm = 0; tm < seconds;)
        {
            bool burst = (int)tm % 15 is >= 5 and < 7;                    // a 2 s flash crowd every 15 s
            double rate = burst ? 900 : 250;
            tm += -Math.Log(1 - rng.NextDouble()) / rate;
            arrivals.Add(tm);
        }
        return arrivals.ToArray();
    }

    /// <summary>Token bucket in front of a FIFO server with a bounded queue.</summary>
    /// <param name="arrivals">Arrival times.</param>
    /// <param name="capacity">Bucket size (tokens).</param>
    /// <param name="refill">Tokens added per second.</param>
    /// <param name="queueLimit">Requests allowed to wait for a token before shedding.</param>
    /// <param name="serviceRate">Backend requests per second.</param>
    /// <returns>p99 latency (ms) of the requests served and the fraction shed.</returns>
    public static (double P99Ms, double ShedFraction) Simulate(double[] arrivals, long capacity, double refill, long queueLimit, double serviceRate)
    {
        double tokens = capacity, lastRefill = 0, serverFree = 0;
        var waiting = new Queue<double>();
        var latencies = new List<double>(arrivals.Length);
        int shed = 0;
        foreach (double a in arrivals)
        {
            tokens = Math.Min(capacity, tokens + (a - lastRefill) * refill);
            lastRefill = a;
            // requests waiting for a token are admitted as tokens accrue (at the token's time).
            while (waiting.Count > 0 && tokens >= 1)
            {
                tokens -= 1;
                Serve(waiting.Dequeue(), a);
            }
            if (tokens >= 1 && waiting.Count == 0)
            {
                tokens -= 1;
                Serve(a, a);
            }
            else if (waiting.Count < queueLimit)
                waiting.Enqueue(a);
            else
                shed++;
        }
        latencies.Sort();
        double p99 = latencies.Count == 0 ? 10_000 : 1000 * latencies[(int)(0.99 * (latencies.Count - 1))];
        return (p99, (double)shed / arrivals.Length);

        // the backend: single FIFO server; a request admitted at `admit` starts when the server is free.
        void Serve(double arrived, double admit)
        {
            double start = Math.Max(admit, serverFree);
            serverFree = start + 1.0 / serviceRate;
            latencies.Add(serverFree - arrived);
        }
    }
}
