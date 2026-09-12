using NumSharp;

namespace OptunaSharp.Samplers.Nsgaii;

/// <summary>
///     Shared tail of <see cref="SBXCrossover" /> and <see cref="VSBXCrossover" />: both produce two
///     candidate children <c>c1</c>/<c>c2</c> from the two parents, then collapse them into one via
///     <c>use_child_gene_prob</c> and <c>uniform_crossover_prob</c>. The RNG draw order — for each
///     gene one <c>rand()</c> (use-child decision) then one <c>rand()</c> (swap decision), and a final
///     <c>rand()</c> to pick which candidate to return — is preserved exactly.
/// </summary>
internal static class SbxLike
{
    /// <summary>Chooses child or parent genes to reduce two simulated-binary offspring to one vector.</summary>
    /// <param name="c1">First child coordinate vector.</param>
    /// <param name="c2">Second child coordinate vector.</param>
    /// <param name="parent0">First numerical parent vector.</param>
    /// <param name="parent1">Second numerical parent vector.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <param name="useChildGeneProb">Probability of retaining a computed child gene instead of a parent gene.</param>
    /// <param name="uniformCrossoverProb">Probability used to choose which child/parent contributes a coordinate.</param>
    /// <returns>One numerical offspring assembled from child and parent genes.</returns>
    internal static double[] CollapseTwoChildren(
        double[] c1, double[] c2, double[] parent0, double[] parent1,
        NumPyRandom rng, double useChildGeneProb, double uniformCrossoverProb)
    {
        int n = c1.Length;
        var child1 = new double[n];
        var child2 = new double[n];
        for (int i = 0; i < n; i++)
        {
            if ((double)rng.rand() < useChildGeneProb)
            {
                if ((double)rng.rand() >= uniformCrossoverProb)
                {
                    child1[i] = c1[i];
                    child2[i] = c2[i];
                }
                else
                {
                    child1[i] = c2[i];
                    child2[i] = c1[i];
                }
            }
            else
            {
                if ((double)rng.rand() >= uniformCrossoverProb)
                {
                    child1[i] = parent0[i];
                    child2[i] = parent1[i];
                }
                else
                {
                    child1[i] = parent1[i];
                    child2[i] = parent0[i];
                }
            }
        }

        return (double)rng.rand() < 0.5 ? child1 : child2;
    }
}
