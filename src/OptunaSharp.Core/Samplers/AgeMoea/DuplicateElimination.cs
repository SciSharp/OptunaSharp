using System;

namespace OptunaSharp.Samplers.AgeMoea;

/// <summary>
///     Default duplicate elimination — port of
///     <c>pymoo.core.duplicate.DefaultDuplicateElimination</c> (<c>@0.6.1.3</c>). Two decision
///     vectors are duplicates when their Euclidean distance is <c>≤ epsilon</c> (<c>1e-16</c>). It
///     consumes <b>no</b> randomness itself; its effect on the RNG stream is indirect — removing
///     offspring makes the mating retry loop (<see cref="Mating" />) re-mate, consuming further draws.
/// </summary>
internal sealed class DuplicateElimination
{
    private const double Epsilon = 1e-16;

    /// <summary>
    ///     <c>do(pop, *others)</c>: drop within-<paramref name="pop" /> duplicates (keeping the first
    ///     occurrence), then drop any individual duplicating a row of one of the <paramref name="others" />.
    /// </summary>
    public Population Do(Population pop, params Population[] others)
    {
        if (pop.Count == 0) return pop;

        // to_itself: cdist(X, X), upper triangle (i <= j) masked to inf -> mark i duplicate if any j < i near.
        var keep = new bool[pop.Count];
        for (int i = 0; i < pop.Count; i++)
        {
            bool dup = false;
            for (int j = 0; j < i; j++)
                if (Dist(pop.X[i], pop.X[j]) <= Epsilon) { dup = true; break; }
            keep[i] = !dup;
        }
        pop = pop.Subset(keep);

        foreach (Population other in others)
        {
            if (pop.Count == 0) break;
            if (other.Count == 0) continue;
            var keep2 = new bool[pop.Count];
            for (int i = 0; i < pop.Count; i++)
            {
                bool dup = false;
                for (int j = 0; j < other.Count; j++)
                    if (Dist(pop.X[i], other.X[j]) <= Epsilon) { dup = true; break; }
                keep2[i] = !dup;
            }
            pop = pop.Subset(keep2);
        }

        return pop;
    }

    /// <summary>Euclidean distance (scipy <c>cdist</c> default). NaN coordinates propagate to inf (never ≤ eps).</summary>
    private static double Dist(double[] a, double[] b)
    {
        double s = 0.0;
        for (int k = 0; k < a.Length; k++)
        {
            double d = a[k] - b[k];
            s += d * d;
        }
        double dist = Math.Sqrt(s);
        return double.IsNaN(dist) ? double.PositiveInfinity : dist;
    }
}
