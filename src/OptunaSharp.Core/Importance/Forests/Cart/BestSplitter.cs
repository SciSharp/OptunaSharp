using System;

namespace OptunaSharp.Importance.Forests.Internal.Cart;

/// <summary>
///     Port of <c>sklearn/tree/_splitter.pyx: Splitter.init/node_reset</c> + <c>node_split_best</c>
///     for the dense, no-missing, no-monotonic regression path. Draws features without replacement
///     with a Fisher–Yates shuffle over <see cref="SklearnRandom.RandInt" />, tracks constant
///     features across parent/child nodes, and selects the split maximizing the MSE proxy impurity
///     improvement (strict <c>&gt;</c>, so ties keep the first-evaluated feature — hence the RNG
///     matters even with <c>max_features = n_features</c>).
/// </summary>
internal sealed class BestSplitter
{
    internal readonly MseRegressionCriterion Criterion = new();
    internal readonly DensePartitioner Partitioner = new();

    internal int[] Samples = null!;
    private int[] _features = null!;
    private int[] _constantFeatures = null!;
    private float[] _featureValues = null!;
    private int _nFeatures;
    private int _maxFeatures;
    private readonly int _minSamplesLeaf;
    private readonly double _minWeightLeaf;
    private double _weightedNSamples;
    private uint _randRState;

    internal int NSamples { get; private set; }

    internal BestSplitter(int minSamplesLeaf, double minWeightLeaf)
    {
        _minSamplesLeaf = minSamplesLeaf;
        _minWeightLeaf = minWeightLeaf;
    }

    /// <summary><c>Splitter.init</c> — seed <c>our_rand_r</c>, gather positively-weighted samples.</summary>
    /// <param name="maxFeatures">
    ///     The number of features to consider per split — scikit-learn's resolved <c>max_features</c>.
    ///     Pass <c>&lt; 0</c> (or <c>nFeatures</c>) to consider all features, the fANOVA/MDI configuration
    ///     (<c>max_features=1.0</c>); a smaller value reproduces a general
    ///     <c>RandomForestRegressor</c>'s per-node feature subsampling (the Fisher–Yates draw over
    ///     <see cref="SklearnRandom.RandInt" /> is consumed identically either way, so the RNG stays in
    ///     lock-step with scikit-learn).
    /// </param>
    internal void Init(float[,] x, double[] y, double[] sampleWeight, int nFeatures, uint randRSeed, int maxFeatures = -1)
    {
        _randRState = randRSeed;
        int nTotal = y.Length;

        Samples = new int[nTotal];
        int j = 0;
        double weightedNSamples = 0.0;
        for (int i = 0; i < nTotal; i++)
        {
            if (sampleWeight[i] != 0.0)
            {
                Samples[j] = i;
                j++;
            }
            weightedNSamples += sampleWeight[i];
        }
        NSamples = j;
        _weightedNSamples = weightedNSamples;

        _nFeatures = nFeatures;
        // Optuna's fANOVA/MDI leaves RandomForestRegressor's default max_features=1.0 -> all features;
        // a general forest passes its resolved max_features (see the Init doc).
        _maxFeatures = maxFeatures < 0 ? nFeatures : maxFeatures;
        _features = new int[nFeatures];
        for (int i = 0; i < nFeatures; i++)
            _features[i] = i;
        _featureValues = new float[nTotal];
        _constantFeatures = new int[nFeatures];

        Criterion.Y = y;
        Criterion.SampleWeight = sampleWeight;
        Criterion.SampleIndices = Samples;
        Partitioner.X = x;
        Partitioner.Samples = Samples;
        Partitioner.FeatureValues = _featureValues;
    }

    /// <summary><c>Splitter.node_reset</c>.</summary>
    internal double NodeReset(int start, int end)
    {
        Criterion.Init(Criterion.Y, Criterion.SampleWeight, _weightedNSamples, Samples, start, end);
        return Criterion.WeightedNNodeSamples;
    }

    internal double NodeImpurity() => Criterion.NodeImpurity();
    internal double NodeValue() => Criterion.NodeValue();

    private static void SwapFeatures(int[] f, int a, int b) => (f[a], f[b]) = (f[b], f[a]);

    /// <summary><c>node_split_best</c>. Returns the best split; updates <paramref name="nConstantFeatures" />.</summary>
    internal SplitRecord NodeSplit(double impurity, ref int nConstantFeatures)
    {
        int start = Criterion.Start;
        int end = Criterion.End;

        var best = new SplitRecord
        {
            ImpurityLeft = double.PositiveInfinity,
            ImpurityRight = double.PositiveInfinity,
            Pos = end,
            Feature = 0,
            Threshold = 0.0,
            Improvement = double.NegativeInfinity,
            MissingGoToLeft = false,
            NMissing = 0,
        };
        var current = new SplitRecord();
        double bestProxy = double.NegativeInfinity;

        Partitioner.InitNodeSplit(start, end);

        int fi = _nFeatures;
        int pPrev = 0, p = 0;
        int nVisited = 0, nFoundConstants = 0, nDrawnConstants = 0;
        int nKnownConstants = nConstantFeatures;
        int nTotalConstants = nKnownConstants;

        while (fi > nTotalConstants &&
               (nVisited < _maxFeatures || nVisited <= nFoundConstants + nDrawnConstants))
        {
            nVisited += 1;

            int fj = SklearnRandom.RandInt(nDrawnConstants, fi - nFoundConstants, ref _randRState);

            if (fj < nKnownConstants)
            {
                SwapFeatures(_features, nDrawnConstants, fj);
                nDrawnConstants++;
                continue;
            }

            fj += nFoundConstants;
            current.Feature = _features[fj];
            Partitioner.SortSamplesAndFeatureValues(current.Feature);
            int nMissing = Partitioner.NMissing;
            int endNonMissing = end - nMissing;

            if (endNonMissing == start ||
                (_featureValues[endNonMissing - 1] <= _featureValues[start] + DensePartitioner.FeatureThreshold && nMissing == 0))
            {
                SwapFeatures(_features, fj, nTotalConstants);
                nFoundConstants++;
                nTotalConstants++;
                continue;
            }

            fi -= 1;
            SwapFeatures(_features, fi, fj);
            Criterion.NMissing = 0;

            Criterion.Reset();
            p = start;
            while (p < endNonMissing)
            {
                Partitioner.NextP(ref pPrev, ref p);
                if (p >= endNonMissing)
                    continue;

                int nLeft = p - start;
                int nRight = endNonMissing - p;
                if (nLeft < _minSamplesLeaf || nRight < _minSamplesLeaf)
                    continue;

                current.Pos = p;
                Criterion.Update(current.Pos);

                if (Criterion.WeightedNLeft < _minWeightLeaf || Criterion.WeightedNRight < _minWeightLeaf)
                    continue;

                double currentProxy = Criterion.ProxyImpurityImprovement();
                if (currentProxy > bestProxy)
                {
                    bestProxy = currentProxy;
                    // sum of halves avoids an infinite intermediate value
                    current.Threshold = _featureValues[pPrev] / 2.0 + _featureValues[p] / 2.0;
                    if (current.Threshold == _featureValues[p] ||
                        double.IsPositiveInfinity(current.Threshold) ||
                        double.IsNegativeInfinity(current.Threshold))
                    {
                        current.Threshold = _featureValues[pPrev];
                    }
                    current.NMissing = 0;
                    current.MissingGoToLeft = nLeft > nRight;  // n_missing == 0
                    best = current;
                }
            }
        }

        if (best.Pos < end)
        {
            Partitioner.PartitionSamplesFinal(best.Feature, best.Threshold, best.NMissing);
            Criterion.NMissing = 0;
            Criterion.Reset();
            Criterion.Update(best.Pos);
            Criterion.ChildrenImpurity(out best.ImpurityLeft, out best.ImpurityRight);
            best.Improvement = Criterion.ImpurityImprovement(impurity, best.ImpurityLeft, best.ImpurityRight);
        }

        // Preserve the original order of known constant features for sibling/child nodes.
        Array.Copy(_constantFeatures, 0, _features, 0, nKnownConstants);
        Array.Copy(_features, nKnownConstants, _constantFeatures, nKnownConstants, nFoundConstants);

        nConstantFeatures = nTotalConstants;
        return best;
    }
}
