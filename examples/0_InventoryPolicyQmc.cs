#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Inventory policy — find the (s, S) reorder policy with the lowest expected daily cost for a warehouse
// item with random demand and a random supplier lead time. Three samplers get the same 64-evaluation budget
// on the same noisy simulator: plain random search, a scrambled Sobol sequence (QMCSampler) and TPE; then
// each winner is re-simulated with fresh demand to show how much of its score was luck.
//
// source: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.QMCSampler.html
//         (quasi-Monte Carlo sampling: low-discrepancy coverage of the space for the same budget)
//         https://en.wikipedia.org/wiki/Inventory_control (the (s, S) "min-max" policy)
//
// Real objective, Monte Carlo simulation: a year of days per replication — Poisson(mean 18) demand, an
// order placed whenever the inventory position (on hand + on order) drops to s brings it back up to S,
// orders arrive after 2–6 days, unmet demand is lost. Daily cost = holding $0.15/unit + lost sale $4/unit +
// $60 per order placed. Each trial averages 12 replications with common random numbers (the same demand
// streams for every policy), which is how a simulation analyst compares policies fairly. The winner is then
// re-scored on 200 fresh replications it was never selected on — the gap is the "winner's curse".
//
//   Run:  dotnet run examples/0_InventoryPolicyQmc.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var contenders = new (string Name, Func<BaseSampler> Make)[]
{
    ("random", () => new RandomSampler(seed: 3)),
    ("qmc (sobol, scrambled)", () => new QMCSampler(qmcType: "sobol", scramble: true, seed: 3)),
    ("tpe", () => new TPESampler(seed: 3)),
};

Console.WriteLine("sampler                    best s   best S   cost/day (selection)   cost/day (fresh 200 reps)");
foreach (var (name, make) in contenders)
{
    var study = Optuna.CreateStudy(sampler: make(), direction: StudyDirection.Minimize, studyName: name);
    study.Optimize(t => Objective(t), nTrials: 64);
    long s = (long)study.BestParams["s"]!, gap = (long)study.BestParams["gap"]!;
    double fresh = Inventory.MeanDailyCost(s, s + gap, replications: 200, seedBase: 1_000_000);
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0,-24} {1,8} {2,8} {3,15:F3} {4,22:F3}",
        name, s, s + gap, study.BestValue, fresh));
}

/// <summary>Scores one (s, S) policy on 12 common-random-number replications.</summary>
/// <param name="t">The trial proposing the reorder point and the order-up-to gap.</param>
/// <returns>The mean daily cost in USD.</returns>
static double Objective(Trial t)
{
    long s = t.SuggestInt("s", 0, 250);          // reorder point: order when position ≤ s
    long gap = t.SuggestInt("gap", 1, 400);      // S − s: the order-up-to level sits this far above s
    return Inventory.MeanDailyCost(s, s + gap, replications: 12, seedBase: 0);
}

/// <summary>A periodic-review lost-sales inventory simulator for one item.</summary>
static class Inventory
{
    /// <summary>Average daily cost of an (s, S) policy over several one-year replications.</summary>
    /// <param name="s">Reorder point.</param>
    /// <param name="bigS">Order-up-to level (S ≥ s + 1).</param>
    /// <param name="replications">Number of independent years to simulate.</param>
    /// <param name="seedBase">First demand-stream seed; equal bases give every policy the same demand (CRN).</param>
    /// <returns>The mean cost per day across all replications.</returns>
    public static double MeanDailyCost(long s, long bigS, int replications, int seedBase)
    {
        double total = 0;
        for (int r = 0; r < replications; r++)
            total += SimulateYear(s, bigS, new Random(seedBase + r));
        return total / replications;
    }

    /// <summary>Simulates 365 days of demand, ordering and arrivals.</summary>
    /// <param name="s">Reorder point.</param>
    /// <param name="bigS">Order-up-to level.</param>
    /// <param name="rng">The replication's random stream (demand and lead times).</param>
    /// <returns>The year's mean daily cost.</returns>
    private static double SimulateYear(long s, long bigS, Random rng)
    {
        const double Hold = 0.15, Lost = 4.0, OrderCost = 60.0;
        long onHand = bigS;
        var pipeline = new List<(int Day, long Qty)>();
        double cost = 0;
        for (int day = 0; day < 365; day++)
        {
            for (int k = pipeline.Count - 1; k >= 0; k--)          // receive today's deliveries
                if (pipeline[k].Day == day)
                {
                    onHand += pipeline[k].Qty;
                    pipeline.RemoveAt(k);
                }
            long demand = Poisson(rng, 18);
            long sold = Math.Min(demand, onHand);
            onHand -= sold;
            cost += Lost * (demand - sold);                        // lost sales: the customer walks
            long position = onHand + pipeline.Sum(p => p.Qty);
            if (position <= s)
            {
                pipeline.Add((day + rng.Next(2, 7), bigS - position));
                cost += OrderCost;
            }
            cost += Hold * onHand;
        }
        return cost / 365;
    }

    /// <summary>Draws a Poisson variate by Knuth's multiplication method.</summary>
    /// <param name="rng">The random stream.</param>
    /// <param name="lambda">The mean.</param>
    /// <returns>A non-negative count.</returns>
    private static long Poisson(Random rng, double lambda)
    {
        double l = Math.Exp(-lambda), p = 1;
        long k = 0;
        do { k++; p *= rng.NextDouble(); } while (p > l);
        return k - 1;
    }
}
