using System;
using System.Collections.Generic;
using NumSharp;

namespace OptunaSharp.Samplers.AgeMoea;

/// <summary>
///     Binary tournament selection — port of <c>pymoo.operators.selection.tournament.TournamentSelection</c>
///     with AGE-MOEA's <c>comp_by_rank_and_crowding</c> comparator from
///     <c>pymoo.algorithms.moo.nsga2.binary_tournament</c> (<c>@0.6.1.3</c>).
///
///     RNG: the parent pool is <c>ceil(n_select · n_parents · pressure / N)</c> full
///     <c>permutation(N)</c> draws, sliced and reshaped C-order; each tournament additionally draws a
///     <c>choice([a, b])</c> (== <c>randint(0, 2)</c>-index) <b>only</b> on a rank tie that is also a
///     crowding tie (or an equal-CV infeasible pair).
/// </summary>
internal sealed class TournamentSelection
{
    private const int Pressure = 2;

    /// <summary>
    ///     <c>_do</c>: select <paramref name="nSelect" /> matings of <paramref name="nParents" />
    ///     parents each, returned as an <c>[n_select][n_parents]</c> index array into
    ///     <paramref name="pop" />.
    /// </summary>
    /// <param name="pop">Population whose decision and evaluation columns are consumed or updated.</param>
    /// <param name="nSelect">Number of mating groups to select.</param>
    /// <param name="nParents">Number of parents required for each mating.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <returns>Parent indices grouped by selected mating.</returns>
    public int[][] Do(Population pop, int nSelect, int nParents, NumPyRandom rng)
    {
        int nPop = pop.Count;
        int nRandom = nSelect * nParents * Pressure;
        int nPerms = (int)Math.Ceiling((double)nRandom / nPop);

        // P = random_permuations(n_perms, n_pop)[:n_random], reshaped (n_select*n_parents, pressure)
        var flat = new int[nPerms * nPop];
        for (int p = 0; p < nPerms; p++)
        {
            int[] perm = NpRandom.Permutation(rng, nPop);
            for (int i = 0; i < nPop; i++)
                flat[p * nPop + i] = perm[i];
        }

        int nTournaments = nSelect * nParents;
        var winners = new int[nTournaments];
        for (int t = 0; t < nTournaments; t++)
        {
            int a = flat[t * Pressure + 0];
            int b = flat[t * Pressure + 1];
            winners[t] = BinaryTournament(pop, a, b, rng);
        }

        // reshape S -> (n_select, n_parents)
        var parents = new int[nSelect][];
        for (int s = 0; s < nSelect; s++)
        {
            parents[s] = new int[nParents];
            for (int k = 0; k < nParents; k++)
                parents[s][k] = winners[s * nParents + k];
        }
        return parents;
    }

    /// <summary><c>binary_tournament</c> with tournament_type = <c>comp_by_rank_and_crowding</c>.</summary>
    /// <param name="pop">Population whose decision and evaluation columns are consumed or updated.</param>
    /// <param name="a">Scalar operand used in the comparison or numerical transformation.</param>
    /// <param name="b">Scalar operand used in the comparison or numerical transformation.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <returns>The selected individual index after rank, feasibility, and diversity comparison.</returns>
    private static int BinaryTournament(Population pop, int a, int b, NumPyRandom rng)
    {
        double aCv = pop.CV![a];
        double bCv = pop.CV![b];

        // at least one infeasible -> smaller CV wins (random on equal)
        if (aCv > 0.0 || bCv > 0.0)
            return Compare(a, aCv, b, bCv, smallerIsBetter: true, returnRandomIfEqual: true, rng) ?? a;

        // both feasible: compare by rank (smaller better); tie -> crowding (larger better, random on tie)
        int? s = Compare(a, pop.Rank![a], b, pop.Rank![b], smallerIsBetter: true, returnRandomIfEqual: false, rng);
        if (s is null)
            s = Compare(a, pop.Crowding![a], b, pop.Crowding![b], smallerIsBetter: false, returnRandomIfEqual: true, rng);
        return s!.Value;
    }

    /// <summary><c>compare</c>: pick a/b by the metric; on an exact tie either return null or draw.</summary>
    /// <param name="a">Scalar operand used in the comparison or numerical transformation.</param>
    /// <param name="aVal">Selection score of the first individual.</param>
    /// <param name="b">Scalar operand used in the comparison or numerical transformation.</param>
    /// <param name="bVal">Selection score of the second individual.</param>
    /// <param name="smallerIsBetter">Whether smaller selection scores win the comparison.</param>
    /// <param name="returnRandomIfEqual">Whether ties choose a random individual instead of returning no winner.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <returns>Winning individual index, or null for an unresolved tie when random tie-breaking is disabled.</returns>
    private static int? Compare(int a, double aVal, int b, double bVal, bool smallerIsBetter, bool returnRandomIfEqual, NumPyRandom rng)
    {
        bool aWins = smallerIsBetter ? aVal < bVal : aVal > bVal;
        bool bWins = smallerIsBetter ? aVal > bVal : aVal < bVal;
        if (aWins) return a;
        if (bWins) return b;
        if (returnRandomIfEqual)
        {
            // np.random.choice([a, b]) == [a, b][randint(0, 2)]
            int[] choices = { a, b };
            return choices[NpRandom.RandInt(rng, 0, 2)];
        }
        return null;
    }
}
