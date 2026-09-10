using System;
using System.Collections.Generic;
using System.Linq;

namespace OptunaSharp.Samplers.AgeMoea;

/// <summary>
///     A column-oriented population — the analogue of pymoo's <c>pymoo.core.population.Population</c>
///     (an array of <c>Individual</c>s) as used on the AGE-MOEA path. Individuals always carry a
///     decision vector <see cref="X" />; objective values <see cref="F" />, constraint violation
///     <see cref="CV" />, and the survival-assigned <see cref="Rank" /> / <see cref="Crowding" /> are
///     filled in once evaluated and ranked.
///
///     Storing columns (rather than a list of objects) keeps the C-order boolean-mask flattening that
///     pymoo's vectorized operators rely on straightforward and bit-faithful.
/// </summary>
internal sealed class Population
{
    public double[][] X;
    public double[][]? F;
    public double[]? CV;
    public int[]? Rank;
    public double[]? Crowding;

    public Population(double[][] x) => X = x;

    public int Count => X.Length;

    /// <summary>New population from decision vectors only (<c>Population.new("X", X)</c>).</summary>
    public static Population New(double[][] x) => new(x);

    /// <summary>An empty population (<c>Population.empty()</c> / <c>Population.create()</c>).</summary>
    public static Population Empty() => new(Array.Empty<double[]>());

    /// <summary>
    ///     Concatenate two populations (<c>Population.merge(a, b)</c>). Column arrays are concatenated
    ///     when present on <em>both</em> operands, else dropped (mirroring how merged offspring carry
    ///     only <c>X</c> until re-evaluated).
    /// </summary>
    public static Population Merge(Population a, Population b)
    {
        var pop = new Population(a.X.Concat(b.X).ToArray());
        if (a.F is not null && b.F is not null) pop.F = a.F.Concat(b.F).ToArray();
        if (a.CV is not null && b.CV is not null) pop.CV = a.CV.Concat(b.CV).ToArray();
        if (a.Rank is not null && b.Rank is not null) pop.Rank = a.Rank.Concat(b.Rank).ToArray();
        if (a.Crowding is not null && b.Crowding is not null) pop.Crowding = a.Crowding.Concat(b.Crowding).ToArray();
        return pop;
    }

    /// <summary>Select individuals by a boolean mask (<c>pop[mask]</c>), preserving order.</summary>
    public Population Subset(bool[] mask)
    {
        var idx = new List<int>();
        for (int i = 0; i < mask.Length; i++)
            if (mask[i]) idx.Add(i);
        return Subset(idx.ToArray());
    }

    /// <summary>Select individuals by index array (<c>pop[idx]</c>), preserving the given order.</summary>
    public Population Subset(int[] idx)
    {
        var pop = new Population(idx.Select(i => X[i]).ToArray());
        if (F is not null) pop.F = idx.Select(i => F[i]).ToArray();
        if (CV is not null) pop.CV = idx.Select(i => CV[i]).ToArray();
        if (Rank is not null) pop.Rank = idx.Select(i => Rank[i]).ToArray();
        if (Crowding is not null) pop.Crowding = idx.Select(i => Crowding[i]).ToArray();
        return pop;
    }

    /// <summary>The first <paramref name="n" /> individuals (<c>pop[:n]</c>).</summary>
    public Population Take(int n) => Subset(Enumerable.Range(0, Math.Min(n, Count)).ToArray());
}
