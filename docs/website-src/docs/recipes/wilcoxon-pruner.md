# Wilcoxon pruner

`WilcoxonPruner` is built for objectives that **average many independent evaluations** — a mean over a
dataset of problem instances, cross-validation folds, or Monte-Carlo samples. It watches the
per-instance results a trial reports and, using a
[Wilcoxon signed-rank test](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test), prunes the trial
once it is *statistically* worse than the best trial so far — without waiting for every instance.

This recipe tunes a **Simulated Annealing** solver for the **Traveling Salesman Problem (TSP)**, whose
objective is the mean tour cost over 50 random instances. The SA/TSP code below is ordinary user
objective code, so it uses `System.Random`; the sampler's own randomness is NumSharp-backed (that is
what the [seeded parity guarantee](../study-and-trials.md#relationship-to-optuna) rests on).

## The TSP solver

```csharp
using System;
using System.Collections.Generic;
using System.Linq;

public readonly record struct SAOptions(int MaxIter, double T0, double Alpha, int Patience);

static class Tsp
{
    static double Dist(double[] a, double[] b) => Math.Sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]));

    // Total length of the closed tour visiting vertices in `idxs` order.
    public static double Cost(double[][] v, int[] idxs)
    {
        int n = idxs.Length;
        double sum = 0;
        for (int i = 0; i < n; i++)
            sum += Dist(v[idxs[i]], v[idxs[(i - 1 + n) % n]]);
        return sum;
    }

    // Nearest-neighbor construction, used as the SA starting point.
    public static int[] Greedy(double[][] v)
    {
        int n = v.Length;
        var idxs = new List<int> { 0 };
        var visited = new bool[n]; visited[0] = true;
        for (int k = 0; k < n - 1; k++)
        {
            int last = idxs[^1], best = -1; double bestD = double.PositiveInfinity;
            for (int j = 0; j < n; j++)
                if (!visited[j] && Dist(v[last], v[j]) < bestD) { bestD = Dist(v[last], v[j]); best = j; }
            visited[best] = true; idxs.Add(best);
        }
        return idxs.ToArray();
    }

    // 2-opt Simulated Annealing.
    public static int[] SimulatedAnnealing(double[][] v, SAOptions o, Random rng)
    {
        double Temperature(double t) => o.T0 * Math.Pow(1 - t, o.Alpha);
        int n = v.Length;

        int[] idxs = Greedy(v);
        double cost = Cost(v, idxs);
        int[] best = (int[])idxs.Clone(); double bestCost = cost;
        int patienceLeft = o.Patience;

        for (int iter = 0; iter < o.MaxIter; iter++)
        {
            int i = rng.Next(0, n);
            int j = (i + 2 + rng.Next(0, n - 3)) % n;
            (i, j) = (Math.Min(i, j), Math.Max(i, j));   // reverse the segment (i, j]

            // Cost change of the 2-opt reversal — computable from just the four affected edges.
            double delta =
                -Dist(v[idxs[(i + 1) % n]], v[idxs[i]]) - Dist(v[idxs[j]], v[idxs[(j + 1) % n]])
                +Dist(v[idxs[i]], v[idxs[j]])          + Dist(v[idxs[(i + 1) % n]], v[idxs[(j + 1) % n]]);
            double temp = Temperature((double)iter / o.MaxIter);

            if (delta <= 0.0 || rng.NextDouble() < Math.Exp(-delta / temp))
            {
                cost += delta;
                Array.Reverse(idxs, i + 1, j - i);       // apply the reversal
                if (cost < bestCost) { Array.Copy(idxs, best, n); bestCost = cost; patienceLeft = o.Patience; }
            }
            if (cost > bestCost && --patienceLeft == 0)   // roll back to best after too long without improvement
            {
                Array.Copy(best, idxs, n); cost = bestCost; patienceLeft = o.Patience;
            }
        }
        return best;
    }
}
```

A random dataset of 50 TSP instances, each 100 vertices in the unit square:

```csharp
static double[][][] MakeDataset(int numVertex, int numProblem, int seed = 0)
{
    var rng = new Random(seed);
    return Enumerable.Range(0, numProblem)
        .Select(_ => Enumerable.Range(0, numVertex)
            .Select(_ => new[] { rng.NextDouble(), rng.NextDouble() }).ToArray())
        .ToArray();
}

double[][][] dataset = MakeDataset(numVertex: 100, numProblem: 50);
const int nSaIter = 10_000;   // small, for demonstration; use ~1_000_000 in practice
var objRng = new Random(0);
```

## The objective: report each instance, prune early

The objective solves every instance, **reports** the cost per instance, and averages them. The key
move: when `ShouldPrune()` fires, *return the running mean* instead of throwing `TrialPruned` — this
hands the sampler an estimate of the final value for the pruned trial, which most samplers use more
effectively than a bare prune.

```csharp
using OptunaSharp;

double Objective(Trial trial)
{
    var options = new SAOptions(
        MaxIter: nSaIter,
        T0:       trial.SuggestFloat("T0", 0.01, 10.0, log: true),
        Alpha:    trial.SuggestFloat("alpha", 1.0, 10.0, log: true),
        Patience: (int)trial.SuggestInt("patience", 10, 1000, log: true));

    var results = new List<double>();
    // Shuffle the evaluation order each trial for a fairer paired comparison.
    int[] order = Enumerable.Range(0, dataset.Length).OrderBy(_ => objRng.Next()).ToArray();
    foreach (int instanceId in order)
    {
        int[] tour = Tsp.SimulatedAnnealing(dataset[instanceId], options, objRng);
        double cost = Tsp.Cost(dataset[instanceId], tour);
        results.Add(cost);

        trial.Report(cost, instanceId);
        if (trial.ShouldPrune())
            return results.Average();   // workaround: report the running estimate, don't raise
    }
    return results.Average();
}
```

## Run it

Pair `TPESampler` with `WilcoxonPruner`, and enqueue a sensible default configuration first:

```csharp
using OptunaSharp.Samplers;
using OptunaSharp.Pruners;
using OptunaSharp.Collections;

var sampler = new TPESampler(seed: 1);
var pruner  = new WilcoxonPruner(pThreshold: 0.1);   // prune when p < 0.1 that this trial is better
var study   = Optuna.CreateStudy(direction: StudyDirection.Minimize, sampler: sampler, pruner: pruner);

study.EnqueueTrial(new OrderedDict<string, object?> { ["T0"] = 1.0, ["alpha"] = 2.0, ["patience"] = 50 });
study.Optimize(Objective, nTrials: 50);

Console.WriteLine($"Best value: {study.BestValue} (params: T0={study.BestParams["T0"]}, " +
                  $"alpha={study.BestParams["alpha"]}, patience={study.BestParams["patience"]})");
```

Because pruned trials stop after only a handful of instances, the total number of SA runs is far below
`50 trials × 50 instances` — that saving is the point of the pruner. You can chart the run with
[`PlotlyVisualization.PlotOptimizationHistory(study)`](../visualization.md); it shows completed and
pruned trials alike.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `optuna.pruners.WilcoxonPruner(p_threshold=0.1)` | `new WilcoxonPruner(pThreshold: 0.1)` |
| `optuna.samplers.TPESampler(seed=1)` | `new TPESampler(seed: 1)` |
| `trial.report(cost, instance_id)` | `trial.Report(cost, instanceId)` |
| `if trial.should_prune(): return sum(results)/len(results)` | `if (trial.ShouldPrune()) return results.Average();` |
| `study.enqueue_trial({"T0": 1.0, …})` | `study.EnqueueTrial(new OrderedDict<string, object?> { ["T0"] = 1.0, … })` |

## See also

[Pruners](../pruners.md) · [Wilcoxon pruner method page](../methods/wilcoxon.md) ·
[User-defined pruner](user-defined-pruner.md) · [Visualization](../visualization.md) ·
the [API reference](../../api/index.md).
