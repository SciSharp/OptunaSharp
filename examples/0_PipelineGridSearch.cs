#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Ingest pipeline grid search — GridSampler measures every combination of channel capacity, consumer count
// and batch size of a real System.Threading.Channels producer/consumer pipeline, then stops on its own when
// the grid is exhausted (no trial budget to guess).
//
// source: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.GridSampler.html
//         (exhaustive grid search; the study stops once every grid point has been evaluated)
//         https://learn.microsoft.com/en-us/dotnet/core/extensions/channels
//
// Real objective: one producer pushes 300,000 events (a small JSON-ish payload each) into a bounded channel;
// N consumers read them in batches and do the per-event work a real ingester does — parse two fields and
// fold a hash. The score is the measured throughput in thousands of events per second (best of 2 runs).
// GridSampler visits the 3 × 4 × 3 = 36 points in a seeded random order; asking for 1,000 trials is safe
// because the sampler stops the study after the 36th. Timings are real, so the ranking is
// machine-dependent — which is exactly why you measure instead of guessing.
//
//   Run:  dotnet run -c Release examples/0_PipelineGridSearch.cs

using System.Diagnostics;
using System.Globalization;
using System.Threading.Channels;
using OptunaSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var grid = new Dictionary<string, IReadOnlyList<CategoricalChoice>>
{
    ["capacity"] = new CategoricalChoice[] { 64, 1024, 16384 },   // bounded channel size (back-pressure)
    ["consumers"] = new CategoricalChoice[] { 1, 2, 4, 8 },        // reader tasks
    ["batch"] = new CategoricalChoice[] { 1, 64, 512 },            // events drained per wake-up
};
var study = Optuna.CreateStudy(sampler: new GridSampler(grid, seed: 0), direction: StudyDirection.Maximize, studyName: "ingest-grid");
study.Optimize(Objective, nTrials: 1000);                         // the grid ends the study at 36

Console.WriteLine($"trials run: {study.Trials.Count} (grid size 36; asked for up to 1000)");
Console.WriteLine();
Console.WriteLine("k events/s by consumers (rows) × batch (cols), best capacity per cell:");
Console.WriteLine("            batch=1     batch=64    batch=512");
foreach (long consumers in new long[] { 1, 2, 4, 8 })
{
    var cells = new[] { 1L, 64L, 512L }.Select(b => study.Trials
        .Where(t => (long)t.Params["consumers"]! == consumers && (long)t.Params["batch"]! == b)
        .Max(t => t.Value!.Value));
    Console.WriteLine($"consumers={consumers,-2}" + string.Concat(cells.Select(v => v.ToString("F0", CultureInfo.InvariantCulture).PadLeft(12))));
}
var best = study.BestTrial;
Console.WriteLine();
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "best: {0:F0} k events/s with capacity={1}, consumers={2}, batch={3}",
    best.Value, best.Params["capacity"], best.Params["consumers"], best.Params["batch"]));

/// <summary>Runs the pipeline twice with the trial's settings and returns the better throughput.</summary>
/// <param name="t">The trial; GridSampler fixes all three parameters to one grid point.</param>
/// <returns>Thousands of events processed per second.</returns>
static double Objective(Trial t)
{
    int capacity = (int)t.SuggestCategorical("capacity", new[] { 64, 1024, 16384 });
    int consumers = (int)t.SuggestCategorical("consumers", new[] { 1, 2, 4, 8 });
    int batch = (int)t.SuggestCategorical("batch", new[] { 1, 64, 512 });
    double best = 0;
    for (int rep = 0; rep < 2; rep++)
        best = Math.Max(best, Pipeline.Run(capacity, consumers, batch, events: 300_000));
    return best;
}

/// <summary>A bounded-channel ingest pipeline.</summary>
static class Pipeline
{
    /// <summary>Pushes events through the pipeline and measures throughput.</summary>
    /// <param name="capacity">Bounded channel capacity.</param>
    /// <param name="consumers">Number of consumer tasks.</param>
    /// <param name="batch">Maximum events a consumer drains per wake-up.</param>
    /// <param name="events">Events to push.</param>
    /// <returns>Thousands of events per second, end to end.</returns>
    /// <exception cref="InvalidOperationException">The consumers processed a different number of events than were sent.</exception>
    public static double Run(int capacity, int consumers, int batch, int events)
    {
        var channel = Channel.CreateBounded<string>(new BoundedChannelOptions(capacity)
        {
            SingleWriter = true,
            SingleReader = consumers == 1,
            FullMode = BoundedChannelFullMode.Wait,               // back-pressure: the producer waits
        });
        long processed = 0, checksum = 0;
        var sw = Stopwatch.StartNew();
        var readers = Enumerable.Range(0, consumers).Select(_ => Task.Run(async () =>
        {
            long local = 0, hash = 0;
            var reader = channel.Reader;
            while (await reader.WaitToReadAsync())
            {
                for (int k = 0; k < batch && reader.TryRead(out var evt); k++)
                {
                    hash = hash * 31 + Parse(evt);
                    local++;
                }
            }
            Interlocked.Add(ref processed, local);
            Interlocked.Add(ref checksum, hash);
        })).ToArray();
        var writer = Task.Run(async () =>
        {
            for (int i = 0; i < events; i++)
                await channel.Writer.WriteAsync($"{{\"id\":{i},\"user\":{i % 977},\"amount\":{i % 1000}.{i % 97:D2}}}");
            channel.Writer.Complete();
        });
        Task.WaitAll(readers.Append(writer).ToArray());
        sw.Stop();
        if (processed != events)
            throw new InvalidOperationException($"lost events: {processed} of {events}");
        return events / sw.Elapsed.TotalSeconds / 1000.0;
    }

    /// <summary>The per-event work: pull the user id and the integer part of the amount out of the payload.</summary>
    /// <param name="evt">One event.</param>
    /// <returns>A value folded into the consumer's checksum (keeps the work from being optimized away).</returns>
    private static long Parse(string evt)
    {
        int u = evt.IndexOf("\"user\":", StringComparison.Ordinal) + 7;
        int a = evt.IndexOf("\"amount\":", StringComparison.Ordinal) + 9;
        long user = long.Parse(evt.AsSpan(u, evt.IndexOf(',', u) - u), CultureInfo.InvariantCulture);
        long amount = long.Parse(evt.AsSpan(a, evt.IndexOf('.', a) - a), CultureInfo.InvariantCulture);
        return user ^ amount;
    }
}
