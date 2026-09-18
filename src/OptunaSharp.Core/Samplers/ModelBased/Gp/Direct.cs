// DIRECT — DIviding RECTangles, the deterministic Lipschitzian global optimizer of Jones, Perttunen &
// Stuckman, "Lipschitzian Optimization Without the Lipschitz Constant", J. Optimization Theory and
// Applications 79(1) (1993) 157-181. REMBO (Wang et al. 2016, §3.3) maximizes its acquisition — and fits
// its length scale — by "running both DIRECT and CMA-ES and using the best of the two"; this is the DIRECT
// half. It is a NEW, beyond-core class (the original paper is a paywalled 1993 Springer article that the
// repo's paper gateway could not retrieve; this implements the canonical algorithm, the same one NLopt's
// `direct` and SciPy's `scipy.optimize.direct` follow).
//
// The method needs no gradient and no Lipschitz constant: it normalizes the box to the unit hypercube,
// keeps a partition into hyperrectangles (each with a sampled center value and a per-dimension trisection
// depth), and each iteration (1) identifies the "potentially optimal" rectangles — those that could hold
// the optimum for SOME rate-of-change constant K > 0 (a balance of a low center value and a large size,
// filtered so the promised improvement over the incumbent is non-trivial, Jones eq. for the ε rule), then
// (2) trisects each along its longest sides, sampling two new centers per long side. Determinism makes it a
// natural complement to the stochastic CMA-ES search.

using System;
using System.Collections.Generic;

namespace OptunaSharp.Samplers.ModelBased.Gp;

/// <summary>
///     The DIRECT global optimizer (Jones et al. 1993) for box-constrained <b>minimization</b> of a cheap
///     deterministic function — REMBO uses it (paired with CMA-ES) to maximize the expected-improvement
///     acquisition and to fit the GP length scale (Wang et al. 2016, §3.3). Callers that want to maximize
///     pass the negated objective. The search is deterministic: the same function and bounds always yield
///     the same result, so it needs no RNG.
///
///     <para><b>Budget.</b> DIRECT trades breadth for depth by the potentially-optimal rule, and the paper
///     notes the acquisition is cheap enough to afford "tens of thousands of evaluations in seconds"; the
///     caller sets <c>maxEvals</c> accordingly. The optimizer stops early if no rectangle is potentially
///     optimal (it cannot happen in practice before the budget) or the iteration cap is hit.</para>
/// </summary>
internal static class Direct
{
    // The Jones ε balances exploitation against exploration in the potentially-optimal test: a rectangle is
    // only kept if the improvement it could promise over the incumbent exceeds ε·|f_min|. 1e-4 is the value
    // recommended by Jones et al. and used by the standard DIRECT implementations.
    private const double Eps = 1e-4;

    /// <summary>One hyperrectangle of the DIRECT partition, in unit-cube coordinates.</summary>
    private sealed class Rect
    {
        /// <summary>The rectangle center in the unit cube <c>[0, 1]^k</c>.</summary>
        internal double[] Center = Array.Empty<double>();
        /// <summary>The trisection depth of each dimension: side length along <c>d</c> is <c>3^(−Level[d])</c>.</summary>
        internal int[] Level = Array.Empty<int>();
        /// <summary>The objective value at the center (minimization convention).</summary>
        internal double F;
        /// <summary>The half-diagonal size measure <c>½·√Σ 3^(−2·Level[d])</c> (Jones' rectangle "size").</summary>
        internal double Size;
    }

    /// <summary>
    ///     Minimizes <paramref name="f" /> over the box <c>[lb, ub]</c> with DIRECT.
    /// </summary>
    /// <param name="f">The objective, called with a point in the original (un-normalized) box; must be finite-valued (a non-finite return is treated as very large so DIRECT avoids that region).</param>
    /// <param name="lb">Per-dimension lower bounds (length <c>k ≥ 1</c>).</param>
    /// <param name="ub">Per-dimension upper bounds; each must exceed the matching <paramref name="lb" />.</param>
    /// <param name="maxEvals">The maximum number of objective evaluations (the primary budget).</param>
    /// <param name="maxIters">A safety cap on divide iterations (each divides all potentially-optimal rectangles).</param>
    /// <returns>The best point found (in the original box) and its objective value.</returns>
    /// <exception cref="System.ArgumentException">The bound arrays differ in length, are empty, or an upper bound does not exceed its lower bound.</exception>
    internal static (double[] X, double F) Minimize(
        Func<double[], double> f, double[] lb, double[] ub, int maxEvals, int maxIters = 10_000)
    {
        int k = lb.Length;
        if (k < 1 || ub.Length != k)
            throw new ArgumentException("lb and ub must be non-empty and the same length.");
        for (int d = 0; d < k; d++)
            if (!(ub[d] > lb[d]))
                throw new ArgumentException($"ub[{d}] must exceed lb[{d}].");

        // Evaluate on the original box: map a unit-cube point through the affine box transform.
        double Eval(double[] unit)
        {
            var x = new double[k];
            for (int d = 0; d < k; d++)
                x[d] = lb[d] + unit[d] * (ub[d] - lb[d]);
            double v = f(x);
            return double.IsNaN(v) || double.IsPositiveInfinity(v) ? double.MaxValue : v;
        }

        var rects = new List<Rect>();
        var center0 = new double[k];
        for (int d = 0; d < k; d++) center0[d] = 0.5;
        var first = new Rect { Center = center0, Level = new int[k], F = Eval(center0) };
        first.Size = SizeOf(first.Level);
        rects.Add(first);
        int evals = 1;

        int bestIdx = 0;
        for (int iter = 0; iter < maxIters && evals < maxEvals; iter++)
        {
            List<int> po = PotentiallyOptimal(rects, rects[bestIdx].F);
            if (po.Count == 0)
                break;

            // Divide each potentially-optimal rectangle. New rectangles are appended; the divided rectangle
            // shrinks in place to become the center sub-rectangle. Snapshot the count so a rectangle created
            // this iteration is not itself divided until the next.
            foreach (int idx in po)
            {
                if (evals >= maxEvals) break;
                evals += Divide(rects, idx, Eval);
            }

            // Refresh the incumbent over all rectangles (centers are the only sampled points).
            for (int i = 0; i < rects.Count; i++)
                if (rects[i].F < rects[bestIdx].F) bestIdx = i;
        }

        Rect best = rects[bestIdx];
        var xBest = new double[k];
        for (int d = 0; d < k; d++)
            xBest[d] = lb[d] + best.Center[d] * (ub[d] - lb[d]);
        return (xBest, best.F);
    }

    /// <summary>
    ///     Identifies the potentially-optimal rectangles (Jones et al. 1993, Def. 3.1). For each distinct
    ///     size only the lowest-value rectangle can qualify, so the test runs over per-size representatives:
    ///     a size is potentially optimal iff there is a rate <c>K &gt; 0</c> with
    ///     <c>F − K·D ≤ F_j − K·D_j</c> for every other size <c>j</c> and <c>F − K·D ≤ f_min − ε·|f_min|</c>.
    ///     The feasible <c>K</c> range is <c>[K_left, K_right]</c> (slopes to smaller / larger sizes); the
    ///     ε rule is tested at the largest feasible <c>K</c> (which minimizes <c>F − K·D</c>). The largest
    ///     size always qualifies (unbounded <c>K</c> ⇒ pure exploration).
    /// </summary>
    /// <param name="rects">The current partition.</param>
    /// <param name="fMin">The incumbent (best center value) used by the ε rule.</param>
    /// <returns>The indices into <paramref name="rects" /> of the potentially-optimal rectangles.</returns>
    private static List<int> PotentiallyOptimal(List<Rect> rects, double fMin)
    {
        // Per distinct size, keep the index of the lowest-value rectangle. Sizes are keyed by a rounded
        // value so rectangles with the same level multiset group together despite float noise.
        var bestPerSize = new Dictionary<double, int>();
        for (int i = 0; i < rects.Count; i++)
        {
            double key = Math.Round(rects[i].Size, 10);
            if (!bestPerSize.TryGetValue(key, out int cur) || rects[i].F < rects[cur].F)
                bestPerSize[key] = i;
        }

        // Group representatives sorted by size ascending.
        var reps = new List<int>(bestPerSize.Values);
        reps.Sort((a, b) => rects[a].Size.CompareTo(rects[b].Size));
        int m = reps.Count;

        var result = new List<int>();
        for (int a = 0; a < m; a++)
        {
            double dA = rects[reps[a]].Size;
            double fA = rects[reps[a]].F;

            // Largest feasible K = min slope to any strictly-larger size; smallest feasible K = max slope
            // to any strictly-smaller size. (Equal sizes were collapsed into the per-size representative.)
            double kRight = double.PositiveInfinity;
            double kLeft = double.NegativeInfinity;
            for (int b = 0; b < m; b++)
            {
                if (b == a) continue;
                double dB = rects[reps[b]].Size;
                double fB = rects[reps[b]].F;
                if (dB > dA)
                    kRight = Math.Min(kRight, (fB - fA) / (dB - dA));
                else if (dB < dA)
                    kLeft = Math.Max(kLeft, (fA - fB) / (dA - dB));
            }

            if (double.IsPositiveInfinity(kRight))
            {
                // No larger size exists: K may be arbitrarily large, so F − K·D → −∞ satisfies the ε rule.
                // Still require a positive feasible K (kLeft can be −∞ when no smaller size exists either).
                result.Add(reps[a]);
                continue;
            }

            // A positive K in [max(kLeft, 0), kRight] must exist, and the ε rule must hold at K = kRight.
            if (kRight <= 0.0 || kRight < kLeft)
                continue;
            if (fA - kRight * dA <= fMin - Eps * Math.Abs(fMin))
                result.Add(reps[a]);
        }
        return result;
    }

    /// <summary>
    ///     Trisects rectangle <paramref name="idx" /> along its longest sides (Jones et al. 1993): for each
    ///     dimension at the minimum trisection depth it samples two centers at <c>±⅓·side</c>, records the
    ///     better of the two, then splits in ascending order of that value so the most promising directions
    ///     keep the largest sub-rectangles. The rectangle at <paramref name="idx" /> becomes the center
    ///     sub-rectangle (its center and value are unchanged; its levels increase along every split
    ///     dimension); the outer thirds are appended as new rectangles.
    /// </summary>
    /// <param name="rects">The partition; new rectangles are appended and <paramref name="idx" /> is updated in place.</param>
    /// <param name="idx">The index of the rectangle to divide.</param>
    /// <param name="eval">The unit-cube objective evaluator (returns the value at a unit-cube center).</param>
    /// <returns>The number of objective evaluations performed (two per long dimension).</returns>
    private static int Divide(List<Rect> rects, int idx, Func<double[], double> eval)
    {
        Rect rect = rects[idx];
        int k = rect.Center.Length;

        int minLevel = int.MaxValue;
        for (int d = 0; d < k; d++) minLevel = Math.Min(minLevel, rect.Level[d]);

        var longDims = new List<int>();
        for (int d = 0; d < k; d++)
            if (rect.Level[d] == minLevel) longDims.Add(d);

        double delta = Math.Pow(3.0, -(minLevel + 1));   // ⅓ of the current longest side

        // Sample the two outer-third centers per long dimension and score each dimension by min(f+, f−).
        var plusF = new double[longDims.Count];
        var minusF = new double[longDims.Count];
        var plusC = new double[longDims.Count][];
        var minusC = new double[longDims.Count][];
        var w = new double[longDims.Count];
        int evals = 0;
        for (int t = 0; t < longDims.Count; t++)
        {
            int d = longDims[t];
            double[] cp = (double[])rect.Center.Clone(); cp[d] += delta;
            double[] cm = (double[])rect.Center.Clone(); cm[d] -= delta;
            plusF[t] = eval(cp); minusF[t] = eval(cm);
            plusC[t] = cp; minusC[t] = cm;
            w[t] = Math.Min(plusF[t], minusF[t]);
            evals += 2;
        }

        // Split in ascending order of w: the best direction is divided first and, because the level of a
        // dimension is only incremented once it is processed, its two outer thirds inherit the fewest
        // subdivisions (largest sub-rectangles) — Jones' rule that gives promising directions more room.
        var order = new int[longDims.Count];
        for (int t = 0; t < order.Length; t++) order[t] = t;
        Array.Sort(order, (x, y) => w[x].CompareTo(w[y]));

        var level = (int[])rect.Level.Clone();
        foreach (int t in order)
        {
            int d = longDims[t];
            level[d] += 1;                       // cumulative: dims split earlier stay incremented
            var lvlCopy = (int[])level.Clone();
            rects.Add(new Rect { Center = plusC[t], Level = (int[])lvlCopy.Clone(), F = plusF[t], Size = SizeOf(lvlCopy) });
            rects.Add(new Rect { Center = minusC[t], Level = (int[])lvlCopy.Clone(), F = minusF[t], Size = SizeOf(lvlCopy) });
        }

        // The original center rectangle is now subdivided along every long dimension.
        rect.Level = level;
        rect.Size = SizeOf(level);
        return evals;
    }

    /// <summary>The Jones half-diagonal size of a rectangle with the given per-dimension trisection depths.</summary>
    /// <param name="level">The trisection depth of each dimension (side length <c>3^(−level[d])</c>).</param>
    /// <returns><c>½·√Σ 3^(−2·level[d])</c>.</returns>
    private static double SizeOf(int[] level)
    {
        double sumSq = 0.0;
        foreach (int lv in level)
        {
            double side = Math.Pow(3.0, -lv);
            sumSq += side * side;
        }
        return 0.5 * Math.Sqrt(sumSq);
    }
}
