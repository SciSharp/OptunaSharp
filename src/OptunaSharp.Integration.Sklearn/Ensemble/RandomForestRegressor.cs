using System;
using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Importance.Forests.Internal;
using OptunaSharp.Importance.Forests.Internal.Cart;
using OptunaSharp.Samplers;

namespace OptunaSharp.Integration.Sklearn;

/// <summary>
///     A byte-exact port of <c>sklearn.ensemble.RandomForestRegressor</c> — bootstrap-aggregated
///     squared-error CART regression trees — built on Core's already-bit-exact scikit-learn CART
///     (<c>DepthFirstTreeBuilder</c> / <c>BestSplitter</c> / <c>MseRegressionCriterion</c> /
///     <c>SklearnRandom</c>) and NumSharp's MT19937.
///
///     <para>
///     It reproduces scikit-learn's exact fit RNG: the forest draws one <c>int32</c> seed per tree from
///     <c>check_random_state(random_state)</c> (<c>_make_estimator</c>'s
///     <c>randint(np.iinfo(np.int32).max)</c>); each tree then re-seeds a fresh
///     <c>RandomState(tree_seed)</c> for its bootstrap sample counts
///     (<c>bincount(randint(0, n_samples, n_samples_bootstrap))</c>) and, separately, for the splitter's
///     <c>our_rand_r</c> seed (<c>randint(0, RAND_R_MAX)</c>) — the same flow the fANOVA/MDI forest uses,
///     here generalized to the full parameter surface (<c>n_estimators</c>, <c>max_depth</c>,
///     <c>min_samples_split</c>/<c>_leaf</c>, <c>min_weight_fraction_leaf</c>, <c>max_features</c>,
///     <c>min_impurity_decrease</c>, <c>bootstrap</c>, <c>max_samples</c>).
///     </para>
///
///     <para>
///     <b>Prediction</b> averages the per-tree leaf values in estimator order (<c>Σ tree.predict / n</c>),
///     with <c>X</c> cast to <c>float32</c> and thresholds compared in <c>float64</c> exactly as
///     scikit-learn's tree traversal does. <b>Scope of the first cut:</b> the default depth-first builder
///     (<c>max_leaf_nodes=None</c>) and <c>criterion="squared_error"</c>; a user <c>sample_weight</c> is not
///     threaded through fit (scikit-learn's <c>OptunaSearchCV</c> passes none by default).
///     </para>
///
///     <para>
///     Like the fANOVA forest, value parity holds on a <b>bit-exact host</b> (the float32 accumulation and
///     the platform <c>np.int_</c> width of the bootstrap <c>randint</c> pin it to x64); the interop suite
///     asserts it there.
///     </para>
/// </summary>
public sealed class RandomForestRegressor : IRegressor
{
    // Parameters (scikit-learn names + defaults). The polymorphic ones (int-or-float / str-or-number)
    // are stored as object, mirroring scikit-learn's isinstance(Integral)/isinstance(Real) dispatch.
    private int _nEstimators = 100;
    private string _criterion = "squared_error";
    private int? _maxDepth;
    private object _minSamplesSplit = 2;
    private object _minSamplesLeaf = 1;
    private double _minWeightFractionLeaf;
    private object? _maxFeatures = 1.0;
    private int? _maxLeafNodes;
    private double _minImpurityDecrease;
    private bool _bootstrap = true;
    private int? _randomState;
    private object? _maxSamples;

    // Fitted state.
    private SklearnLikeTree[]? _trees;
    private int _nFeatures;

    /// <param name="nEstimators">Number of trees (scikit-learn default 100).</param>
    /// <param name="maxDepth">Maximum tree depth; <c>null</c> means unlimited (scikit-learn's <c>int32.max</c>).</param>
    /// <param name="minSamplesSplit">Minimum samples to split an internal node (int count, default 2).</param>
    /// <param name="minSamplesLeaf">Minimum samples at a leaf (int count, default 1).</param>
    /// <param name="minWeightFractionLeaf">Minimum weighted fraction of the total at a leaf (default 0).</param>
    /// <param name="maxFeatures">
    ///     Features considered per split: <c>null</c> or <c>1.0</c> = all; a fraction in (0, 1];
    ///     <c>"sqrt"</c> / <c>"log2"</c>; or an integer count (scikit-learn default <c>1.0</c>).
    /// </param>
    /// <param name="minImpurityDecrease">Minimum impurity decrease required to split (default 0).</param>
    /// <param name="bootstrap">Whether to bootstrap-sample each tree's training set (default <c>true</c>).</param>
    /// <param name="randomState">Seed controlling the whole forest's randomness. <c>null</c> = unseeded.</param>
    /// <param name="maxSamples">
    ///     Bootstrap draw size when <paramref name="bootstrap" /> is <c>true</c>: <c>null</c> = <c>n_samples</c>;
    ///     an int count; or a fraction of <c>n_samples</c>.
    /// </param>
    /// <param name="criterion">Split criterion; only <c>"squared_error"</c> is supported.</param>
    public RandomForestRegressor(
        int nEstimators = 100,
        int? maxDepth = null,
        int minSamplesSplit = 2,
        int minSamplesLeaf = 1,
        double minWeightFractionLeaf = 0.0,
        object? maxFeatures = null,
        double minImpurityDecrease = 0.0,
        bool bootstrap = true,
        int? randomState = null,
        object? maxSamples = null,
        string criterion = "squared_error")
    {
        _nEstimators = nEstimators;
        _maxDepth = maxDepth;
        _minSamplesSplit = minSamplesSplit;
        _minSamplesLeaf = minSamplesLeaf;
        _minWeightFractionLeaf = minWeightFractionLeaf;
        _maxFeatures = maxFeatures ?? 1.0;
        _minImpurityDecrease = minImpurityDecrease;
        _bootstrap = bootstrap;
        _randomState = randomState;
        _maxSamples = maxSamples;
        _criterion = criterion;
    }

    /// <summary>The fitted trees (in estimator order), or <c>null</c> before a fit.</summary>
    internal SklearnLikeTree[]? Trees => _trees;

    /// <summary>Number of trees.</summary>
    public int NEstimators => _nEstimators;

    private RandomForestRegressor CloneUnfitted() => new()
    {
        _nEstimators = _nEstimators,
        _criterion = _criterion,
        _maxDepth = _maxDepth,
        _minSamplesSplit = _minSamplesSplit,
        _minSamplesLeaf = _minSamplesLeaf,
        _minWeightFractionLeaf = _minWeightFractionLeaf,
        _maxFeatures = _maxFeatures,
        _maxLeafNodes = _maxLeafNodes,
        _minImpurityDecrease = _minImpurityDecrease,
        _bootstrap = _bootstrap,
        _randomState = _randomState,
        _maxSamples = _maxSamples,
    };

    private RandomForestRegressor() { }

    /// <inheritdoc />
    public IRegressor WithParams(IReadOnlyDictionary<string, object?> parameters)
    {
        var clone = CloneUnfitted();
        foreach (var (name, value) in parameters)
            clone.SetParam(name, value);
        return clone;
    }

    // The analogue of estimator.set_params(**params). Keys are scikit-learn parameter names.
    private void SetParam(string name, object? value)
    {
        switch (name)
        {
            case "n_estimators": _nEstimators = AsInt(value); break;
            case "criterion": _criterion = (string)value!; break;
            case "max_depth": _maxDepth = value is null ? null : AsInt(value); break;
            case "min_samples_split": _minSamplesSplit = value!; break;
            case "min_samples_leaf": _minSamplesLeaf = value!; break;
            case "min_weight_fraction_leaf": _minWeightFractionLeaf = AsDouble(value); break;
            case "max_features": _maxFeatures = value; break;
            case "max_leaf_nodes": _maxLeafNodes = value is null ? null : AsInt(value); break;
            case "min_impurity_decrease": _minImpurityDecrease = AsDouble(value); break;
            case "bootstrap": _bootstrap = (bool)value!; break;
            case "random_state": _randomState = value is null ? null : AsInt(value); break;
            case "max_samples": _maxSamples = value; break;
            default:
                throw new ValueError($"Invalid parameter '{name}' for estimator RandomForestRegressor.");
        }
    }

    /// <inheritdoc />
    public void Fit(double[,] x, double[] y)
    {
        if (_criterion != "squared_error")
            throw new NotSupportedException(
                $"RandomForestRegressor: only criterion='squared_error' is ported (got '{_criterion}').");
        if (_maxLeafNodes is not null)
            throw new NotSupportedException(
                "RandomForestRegressor: max_leaf_nodes (the best-first builder) is not ported; leave it null.");

        int nSamples = y.Length;
        int nFeatures = x.GetLength(1);
        _nFeatures = nFeatures;

        // scikit-learn casts X to float32 (DTYPE) before building trees.
        var xf = ToFloat32(x);

        // Resolve the tree hyperparameters exactly as BaseDecisionTree._fit does (per n_samples).
        int maxDepth = _maxDepth ?? SklearnRandom.RandRMax;  // np.iinfo(np.int32).max
        int minSamplesLeaf = ResolveMinSamplesLeaf(_minSamplesLeaf, nSamples);
        int minSamplesSplit = ResolveMinSamplesSplit(_minSamplesSplit, nSamples, minSamplesLeaf);
        int maxFeatures = ResolveMaxFeatures(_maxFeatures, nFeatures);

        int nSamplesBootstrap = _bootstrap ? GetNSamplesBootstrap(nSamples, _maxSamples) : nSamples;

        // tree seeds: check_random_state(random_state).randint(np.iinfo(int32).max, size=n_estimators),
        // one per tree via _make_estimator (equivalent to a single vectorized draw for RandomState).
        var forestRng = new LazyRandomState(_randomState).Rng;
        int[] treeSeeds = forestRng
            .randint(0, SklearnRandom.RandRMax, new Shape(_nEstimators), typeof(int)).ToArray<int>();

        var trees = new SklearnLikeTree[_nEstimators];
        for (int t = 0; t < _nEstimators; t++)
        {
            int treeSeed = treeSeeds[t];

            double[] sampleWeight;
            if (_bootstrap)
            {
                // _generate_sample_indices: fresh RandomState(tree_seed).randint(0, n_samples, n_bootstrap),
                // then sample_weight_tree = bincount(indices, minlength=n_samples).
                int[] indices = new LazyRandomState(treeSeed).Rng
                    .randint(0, nSamples, new Shape(nSamplesBootstrap), typeof(int)).ToArray<int>();
                sampleWeight = new double[nSamples];
                foreach (int i in indices)
                    sampleWeight[i] += 1.0;
            }
            else
            {
                // bootstrap=False: the tree is fit on all samples with unit weights (sample_weight=None).
                sampleWeight = new double[nSamples];
                for (int i = 0; i < nSamples; i++)
                    sampleWeight[i] = 1.0;
            }

            // min_weight_leaf = min_weight_fraction_leaf * sum(sample_weight) (== * n for the None path).
            double sumWeight = 0.0;
            foreach (double w in sampleWeight)
                sumWeight += w;
            double minWeightLeaf = _minWeightFractionLeaf * sumWeight;

            // splitter our_rand_r seed: a *fresh* RandomState(tree_seed).randint(0, RAND_R_MAX).
            uint randRSeed = (uint)(long)new LazyRandomState(treeSeed).Rng.randint(0, SklearnRandom.RandRMax);

            trees[t] = DepthFirstTreeBuilder.Build(
                xf, y, sampleWeight, nFeatures,
                maxDepth: maxDepth, minSamplesSplit: minSamplesSplit, minSamplesLeaf: minSamplesLeaf,
                minWeightLeaf: minWeightLeaf, minImpurityDecrease: _minImpurityDecrease,
                randRSeed: randRSeed, maxFeatures: maxFeatures);
        }

        _trees = trees;
    }

    /// <inheritdoc />
    public double[] Predict(double[,] x)
    {
        if (_trees is null)
            throw new InvalidOperationException("RandomForestRegressor is not fitted. Call Fit first.");

        int m = x.GetLength(0);
        var xf = ToFloat32(x);
        var yHat = new double[m];

        // ForestRegressor.predict: y_hat = Σ_tree tree.predict(X) (estimator order), then /= n_estimators.
        foreach (var tree in _trees)
            for (int i = 0; i < m; i++)
                yHat[i] += PredictTree(tree, xf, i);

        for (int i = 0; i < m; i++)
            yHat[i] /= _trees.Length;

        return yHat;
    }

    /// <inheritdoc />
    public double Score(double[,] x, double[] y) => Metrics.R2Score(y, Predict(x));

    // sklearn.tree._tree.Tree.predict traversal: X is float32, the threshold float64 (float32 <= float64
    // promotes to float64), returning the leaf's stored node value.
    private static double PredictTree(SklearnLikeTree tree, float[,] xf, int row)
    {
        int node = 0;
        while (tree.ChildrenLeft[node] != -1)
        {
            int feature = tree.Feature[node];
            if ((double)xf[row, feature] <= tree.Threshold[node])
                node = tree.ChildrenLeft[node];
            else
                node = tree.ChildrenRight[node];
        }
        return tree.Value[node];
    }

    private static float[,] ToFloat32(double[,] x)
    {
        int rows = x.GetLength(0), cols = x.GetLength(1);
        var xf = new float[rows, cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                xf[i, j] = (float)x[i, j];  // np.float32(x): round-to-nearest-even, matches numpy.
        return xf;
    }

    // ---- scikit-learn hyperparameter resolution (BaseDecisionTree._fit / _get_n_samples_bootstrap) ----

    private static int ResolveMinSamplesLeaf(object v, int nSamples)
        => IsIntegral(v) ? AsInt(v) : (int)Math.Ceiling(AsDouble(v) * nSamples);

    private static int ResolveMinSamplesSplit(object v, int nSamples, int minSamplesLeaf)
    {
        int mss = IsIntegral(v)
            ? AsInt(v)
            : Math.Max(2, (int)Math.Ceiling(AsDouble(v) * nSamples));
        return Math.Max(mss, 2 * minSamplesLeaf);
    }

    private static int ResolveMaxFeatures(object? v, int nFeatures)
    {
        switch (v)
        {
            case null:
                return nFeatures;
            case string s when s == "sqrt":
                return Math.Max(1, (int)Math.Sqrt(nFeatures));
            case string s when s == "log2":
                return Math.Max(1, (int)Math.Log2(nFeatures));
            case string s:
                throw new ValueError($"Invalid value for max_features: '{s}'.");
            default:
                if (IsIntegral(v))
                    return AsInt(v);
                double f = AsDouble(v);
                return f > 0.0 ? Math.Max(1, (int)(f * nFeatures)) : 0;
        }
    }

    private static int GetNSamplesBootstrap(int nSamples, object? maxSamples)
    {
        if (maxSamples is null)
            return nSamples;
        if (IsIntegral(maxSamples))
            return AsInt(maxSamples);
        double f = AsDouble(maxSamples);
        return Math.Max((int)(f * nSamples), 1);
    }

    private static bool IsIntegral(object? v) => v is int or long or short or byte or sbyte or uint or ushort;

    private static int AsInt(object? v) => v switch
    {
        int i => i,
        long l => (int)l,
        short s => s,
        byte b => b,
        sbyte sb => sb,
        uint u => (int)u,
        ushort us => us,
        double d => (int)d,
        float f => (int)f,
        _ => throw new ValueError($"Expected an integer parameter value, got {v?.GetType().Name ?? "null"}."),
    };

    private static double AsDouble(object? v) => v switch
    {
        double d => d,
        float f => f,
        int i => i,
        long l => l,
        _ => throw new ValueError($"Expected a numeric parameter value, got {v?.GetType().Name ?? "null"}."),
    };
}
