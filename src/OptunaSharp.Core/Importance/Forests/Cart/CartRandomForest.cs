using NumSharp;
using OptunaSharp.Samplers;

namespace OptunaSharp.Importance.Forests.Internal.Cart;

/// <summary>
///     Port of <c>sklearn.ensemble.RandomForestRegressor</c>'s fit for the fANOVA/MDI configuration
///     (bootstrap, <c>max_features=1.0</c>, <c>min_samples_split=2</c>, <c>min_samples_leaf=1</c>,
///     squared-error). Reproduces the exact RNG flow: the forest draws one int32 seed per tree from
///     <c>np.random.RandomState(seed)</c> (NumSharp's byte-exact MT19937); each tree independently
///     re-seeds a fresh <c>RandomState(tree_seed)</c> for its bootstrap sample weights
///     (<c>bincount(randint(0, n, n))</c>) and, separately, for the splitter's <c>our_rand_r</c> seed
///     (<c>randint(0, RAND_R_MAX)</c>). Verified bit-exact node-for-node vs live scikit-learn.
///
///     <para>The seed bound is <c>2**31 - 1</c>, for which numpy's int32 and int64 <c>randint</c>
///     streams are identical, so the result is platform-independent (unlike a general default-int
///     draw); the bootstrap is explicitly int32.</para>
/// </summary>
internal static class CartRandomForest
{
    /// <summary>Returns the bootstrap-trained regression trees in ensemble order.</summary>
    /// <param name="x">Feature values with one sample per row and one feature per column.</param>
    /// <param name="y">Target values aligned with the feature rows.</param>
    /// <param name="nFeatures">The number of feature columns in the training matrix.</param>
    /// <param name="nTrees">The number of trees to fit in the ensemble.</param>
    /// <param name="maxDepth">The maximum tree depth; the root has depth zero.</param>
    /// <param name="seed">The seed controlling the reproducible random stream.</param>
    /// <returns>The bootstrap-trained regression trees in ensemble order.</returns>
    internal static SklearnLikeTree[] Fit(float[,] x, double[] y, int nFeatures, int nTrees, int maxDepth, int? seed)
    {
        int n = y.Length;

        // tree seeds: check_random_state(seed).randint(MAX_INT, size=n_estimators)
        var forestRng = new LazyRandomState(seed).Rng;
        int[] treeSeeds = forestRng.randint(0, SklearnRandom.RandRMax, new Shape(nTrees), typeof(int)).ToArray<int>();

        var trees = new SklearnLikeTree[nTrees];
        for (int t = 0; t < nTrees; t++)
        {
            int treeSeed = treeSeeds[t];

            // bootstrap sample weights: bincount(RandomState(tree_seed).randint(0, n, n, int32))
            int[] indices = new LazyRandomState(treeSeed).Rng.randint(0, n, new Shape(n), typeof(int)).ToArray<int>();
            var sampleWeight = new double[n];
            foreach (int i in indices)
                sampleWeight[i] += 1.0;

            // splitter our_rand_r seed: a *fresh* RandomState(tree_seed).randint(0, RAND_R_MAX)
            uint randRSeed = (uint)(long)new LazyRandomState(treeSeed).Rng.randint(0, SklearnRandom.RandRMax);

            trees[t] = DepthFirstTreeBuilder.Build(
                x, y, sampleWeight, nFeatures,
                maxDepth: maxDepth, minSamplesSplit: 2, minSamplesLeaf: 1,
                minWeightLeaf: 0.0, minImpurityDecrease: 0.0, randRSeed: randRSeed);
        }

        return trees;
    }
}
