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
