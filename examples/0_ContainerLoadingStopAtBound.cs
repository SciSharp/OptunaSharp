#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Container loading — tune a bin-packing heuristic that assigns a week's shipments to containers, starting
// from the load planners' rules of thumb (enqueued with `skip_if_exists`, so re-running never duplicates
// them) and stopping the study the moment a plan reaches the provable lower bound (`study.stop()` from a
// callback): there is nothing better to find, so the remaining budget is not spent.
//
// source: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.enqueue_trial
//         (`enqueue_trial(params, skip_if_exists=True)`)
//         https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.stop
//         https://en.wikipedia.org/wiki/Bin_packing_problem (first-fit / best-fit decreasing, the L1 bound)
//
// Real objective, a real heuristic: 120 shipments (pallet-metres) generated as Falkenauer-style triplets —
// every trailer of the known optimum holds exactly three of them — are packed into 13.6 m trailers. The heuristic orders the shipments by size^p (p tunable;
// with noise of a tunable seed to break ties differently), places each by first-, best- or worst-fit, then
// runs a tunable number of repair passes that try to empty the least-full trailer into the others. The
// score is the number of trailers plus a small term for how empty the emptiest one is (an emptier last
// trailer is easier to cancel). The lower bound ⌈Σ size / capacity⌉ is known, so reaching it ends the study.
//
//   Run:  dotnet run examples/0_ContainerLoadingStopAtBound.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var shipments = Shipments.Week(count: 120, seed: 37);
const double Capacity = 13.6;
int lowerBound = (int)Math.Ceiling(shipments.Sum() / Capacity);
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0} shipments, {1:F1} pallet-metres; lower bound: {2} trailers", shipments.Length, shipments.Sum(), lowerBound));

var study = Optuna.CreateStudy(sampler: new TPESampler(seed: 38, nStartupTrials: 15), studyName: "trailer-loading");
// the planners' rules of thumb: first-fit-decreasing and best-fit-decreasing, no repair.
foreach (string fit in new[] { "first", "best" })
    for (int twice = 0; twice < 2; twice++)                   // the second enqueue is skipped: already queued
        study.EnqueueTrial(new OrderedDict<string, object?> { ["order_power"] = 1.0, ["noise_seed"] = 0L, ["fit"] = fit, ["repair_passes"] = 0L },
            skipIfExists: true);

void StopAtBound(Study s, FrozenTrial t)
{
    if (t.Value is double v && Math.Floor(v) <= lowerBound)
        s.Stop();                                              // provably optimal trailer count: done
}
study.Optimize(t => Objective(t, shipments, Capacity), nTrials: 300, callbacks: new Action<Study, FrozenTrial>[] { StopAtBound });

// note: if the study stops early, a still-queued enqueued trial stays WAITING in the storage (it would run
// first on the next optimize) — so summarize only the COMPLETE trials.
var completed = study.Trials.Where(t => t.State == TrialState.Complete).ToList();
var rulesOfThumb = completed.Where(t => (long)t.Params["repair_passes"]! == 0 && (double)t.Params["order_power"]! == 1.0 && (long)t.Params["noise_seed"]! == 0).Take(2).Select(t => $"{t.Params["fit"]}-fit decreasing → {Math.Floor(t.Value!.Value)} trailers").ToList();
Console.WriteLine("planners' heuristics: " + string.Join("; ", rulesOfThumb));
var best = study.BestTrial;
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "tuned: {0} trailers after {1} trials ({2}); emptiest trailer {3:P0} full",
    Math.Floor(best.Value!.Value), completed.Count, Math.Floor(best.Value.Value) <= lowerBound ? "hit the lower bound — study stopped" : "budget used up",
    best.UserAttrs["emptiest_fill"]));
foreach (var kv in best.Params)
    Console.WriteLine($"    {kv.Key,-14} {kv.Value}");

/// <summary>Packs the week's shipments with the trial's heuristic settings.</summary>
/// <param name="t">The trial proposing the ordering power, tie-break seed, fit rule and repair effort.</param>
/// <param name="items">Shipment sizes in pallet-metres.</param>
/// <param name="capacity">Trailer capacity in pallet-metres.</param>
/// <returns>Trailers used plus (0.99 × the emptiest trailer's fill), so fewer and emptier-last plans win.</returns>
static double Objective(Trial t, double[] items, double capacity)
{
    double power = t.SuggestFloat("order_power", 0.25, 4.0);       // sort key: size^p (p = 1: plain decreasing)
    long noiseSeed = t.SuggestInt("noise_seed", 0, 50);             // tie-breaking perturbation of the order
    string fit = t.SuggestCategorical("fit", new[] { "first", "best", "worst" });
    long passes = t.SuggestInt("repair_passes", 0, 20);            // try to empty the least-full trailer
    var bins = Packer.Pack(items, capacity, power, (int)noiseSeed, fit, (int)passes);
    double emptiest = bins.Min(b => b.Sum()) / capacity;
    t.SetUserAttr("emptiest_fill", emptiest);
    return bins.Count + 0.99 * emptiest;
}

/// <summary>The week's shipments.</summary>
static class Shipments
{
    /// <summary>
    ///     Generates a "triplet" week (Falkenauer's hard class): the shipments come in groups of three that
    ///     exactly fill one trailer, so the lower bound is achievable — but only by packing the right triples
    ///     together, which decreasing-order heuristics rarely do.
    /// </summary>
    /// <param name="count">Number of shipments (a multiple of three).</param>
    /// <param name="seed">Data seed.</param>
    /// <returns>Sizes in pallet-metres, shuffled.</returns>
    public static double[] Week(int count, int seed)
    {
        const double Capacity = 13.6;
        var rng = new Random(seed);   // data generation only — samplers never use System.Random
        var items = new List<double>();
        for (int k = 0; k < count / 3; k++)
        {
            // three loads of roughly a quarter to a half trailer that sum to exactly one trailer (0.1 m grid).
            int total = (int)Math.Round(Capacity * 10), a = 34 + rng.Next(34), b = 34 + rng.Next(Math.Max(1, total - a - 34 - 34));
            items.AddRange(new[] { a / 10.0, b / 10.0, (total - a - b) / 10.0 });
        }
        var arr = items.ToArray();
        rng.Shuffle(arr);
        return arr;
    }
}

/// <summary>A parameterized decreasing-order bin-packing heuristic with a repair step.</summary>
static class Packer
{
    /// <summary>Packs items into bins.</summary>
    /// <param name="items">Item sizes.</param>
    /// <param name="capacity">Bin capacity.</param>
    /// <param name="power">Exponent of the ordering key.</param>
    /// <param name="noiseSeed">Seed of the small multiplicative jitter applied to the key.</param>
    /// <param name="fit">"first", "best" or "worst" fit.</param>
    /// <param name="passes">Repair passes.</param>
    /// <returns>The bins, each a list of item sizes.</returns>
    public static List<List<double>> Pack(double[] items, double capacity, double power, int noiseSeed, string fit, int passes)
    {
        var rng = new Random(noiseSeed);
        var order = items.Select(s => (Size: s, Key: Math.Pow(s, power) * (1 + 0.05 * rng.NextDouble()))).OrderByDescending(p => p.Key).Select(p => p.Size);
        var bins = new List<List<double>>();
        var loads = new List<double>();
        foreach (double s in order)
            Place(s);
        for (int p = 0; p < passes && bins.Count > 1; p++)
        {
            int victim = loads.IndexOf(loads.Min());
            var moved = bins[victim];
            bins.RemoveAt(victim);
            loads.RemoveAt(victim);
            int before = bins.Count;
            foreach (double s in moved.OrderByDescending(x => x))
                Place(s);
            if (bins.Count > before)
                break;                                             // could not empty it: stop repairing
        }
        return bins;

        // put one item into an existing bin by the fit rule, or open a new bin.
        void Place(double s)
        {
            int chosen = -1;
            for (int b = 0; b < bins.Count; b++)
            {
                if (loads[b] + s > capacity + 1e-9)
                    continue;
                // best fit = the fullest bin that still fits; worst fit = the emptiest one.
                if (chosen < 0 || (fit == "best" && loads[b] > loads[chosen]) || (fit == "worst" && loads[b] < loads[chosen]))
                    chosen = b;
                if (fit == "first")
                    break;                                         // first fit: the first bin that fits wins
            }
            if (chosen < 0)
            {
                bins.Add(new List<double>());
                loads.Add(0);
                chosen = bins.Count - 1;
            }
            bins[chosen].Add(s);
            loads[chosen] += s;
        }
    }
}
