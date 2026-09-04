using System.Collections.Generic;

namespace OptunaSharp.Importance.Forests.Internal.Cart;

/// <summary>
///     The growing node arrays — port of <c>sklearn/tree/_tree.pyx: Tree</c> (<c>_add_node</c> and
///     the <c>Node</c> struct fields) for a single-output regression tree. Emits a
///     <see cref="SklearnLikeTree" /> carrying the split structure (for fANOVA) plus the stored
///     per-node impurity and weighted counts (for exact MDI).
/// </summary>
internal sealed class CartNodeTree
{
    internal const int TreeLeaf = -1;        // sklearn TREE_LEAF
    internal const int TreeUndefined = -2;   // sklearn TREE_UNDEFINED

    private readonly int _nFeatures;
    private readonly List<int> _feature = new(), _childrenLeft = new(), _childrenRight = new(), _nNodeSamples = new();
    private readonly List<double> _threshold = new(), _impurity = new(), _weightedNNodeSamples = new(), _value = new();

    internal CartNodeTree(int nFeatures) => _nFeatures = nFeatures;

    internal int NodeCount => _feature.Count;

    /// <summary><c>Tree._add_node</c>. Returns the new node id.</summary>
    internal int AddNode(int parent, bool isLeft, bool isLeaf, int feature, double threshold, double impurity,
                         int nNodeSamples, double weightedNNodeSamples)
    {
        int nodeId = NodeCount;
        _feature.Add(0); _threshold.Add(0.0); _childrenLeft.Add(0); _childrenRight.Add(0);
        _impurity.Add(impurity); _nNodeSamples.Add(nNodeSamples); _weightedNNodeSamples.Add(weightedNNodeSamples);
        _value.Add(0.0);

        if (parent != TreeUndefined)
        {
            if (isLeft) _childrenLeft[parent] = nodeId;
            else _childrenRight[parent] = nodeId;
        }

        if (isLeaf)
        {
            _childrenLeft[nodeId] = TreeLeaf;
            _childrenRight[nodeId] = TreeLeaf;
            _feature[nodeId] = TreeUndefined;
            _threshold[nodeId] = TreeUndefined;
        }
        else
        {
            _feature[nodeId] = feature;
            _threshold[nodeId] = threshold;
        }

        return nodeId;
    }

    internal void SetValue(int nodeId, double value) => _value[nodeId] = value;

    internal SklearnLikeTree ToSklearnLikeTree()
        => new(_nFeatures, _feature.ToArray(), _childrenLeft.ToArray(), _childrenRight.ToArray(),
               _threshold.ToArray(), _value.ToArray(), _impurity.ToArray(), _weightedNNodeSamples.ToArray());
}

/// <summary>
///     Port of <c>sklearn/tree/_tree.pyx: DepthFirstTreeBuilder.build</c> — the non-recursive,
///     LIFO-stack depth-first tree construction (right child pushed before left, so the left child
///     takes the next node id: scikit-learn's pre-order numbering).
/// </summary>
internal static class DepthFirstTreeBuilder
{
    internal const double Epsilon = 2.220446049250313e-16;  // np.finfo(double).eps

    private struct StackRecord
    {
        internal int Start, End, Depth, Parent;
        internal bool IsLeft;
        internal double Impurity;
        internal int NConstantFeatures;
    }

    internal static SklearnLikeTree Build(
        float[,] x, double[] y, double[] sampleWeight, int nFeatures,
        int maxDepth, int minSamplesSplit, int minSamplesLeaf, double minWeightLeaf, double minImpurityDecrease,
        uint randRSeed)
    {
        var tree = new CartNodeTree(nFeatures);
        var splitter = new BestSplitter(minSamplesLeaf, minWeightLeaf);
        splitter.Init(x, y, sampleWeight, nFeatures, randRSeed);

        int nNodeSamples = splitter.NSamples;
        bool first = true;

        var stack = new Stack<StackRecord>();
        stack.Push(new StackRecord
        {
            Start = 0, End = nNodeSamples, Depth = 0, Parent = CartNodeTree.TreeUndefined,
            IsLeft = false, Impurity = double.PositiveInfinity, NConstantFeatures = 0,
        });

        while (stack.Count > 0)
        {
            var rec = stack.Pop();
            int start = rec.Start, end = rec.End, depth = rec.Depth, parent = rec.Parent;
            bool isLeft = rec.IsLeft;
            double parentImpurity = rec.Impurity;
            int nConstantFeatures = rec.NConstantFeatures;

            nNodeSamples = end - start;
            double weightedNNodeSamples = splitter.NodeReset(start, end);

            bool isLeaf = depth >= maxDepth ||
                          nNodeSamples < minSamplesSplit ||
                          nNodeSamples < 2 * minSamplesLeaf ||
                          weightedNNodeSamples < 2 * minWeightLeaf;

            if (first)
            {
                parentImpurity = splitter.NodeImpurity();
                first = false;
            }

            isLeaf = isLeaf || parentImpurity <= Epsilon;

            SplitRecord split = default;
            if (!isLeaf)
            {
                split = splitter.NodeSplit(parentImpurity, ref nConstantFeatures);
                isLeaf = isLeaf || split.Pos >= end || (split.Improvement + Epsilon < minImpurityDecrease);
            }

            int nodeId = tree.AddNode(parent, isLeft, isLeaf, split.Feature, split.Threshold, parentImpurity,
                                      nNodeSamples, weightedNNodeSamples);
            tree.SetValue(nodeId, splitter.NodeValue());

            if (!isLeaf)
            {
                // Push right child first so the left child pops first and takes the next node id.
                stack.Push(new StackRecord
                {
                    Start = split.Pos, End = end, Depth = depth + 1, Parent = nodeId,
                    IsLeft = false, Impurity = split.ImpurityRight, NConstantFeatures = nConstantFeatures,
                });
                stack.Push(new StackRecord
                {
                    Start = start, End = split.Pos, Depth = depth + 1, Parent = nodeId,
                    IsLeft = true, Impurity = split.ImpurityLeft, NConstantFeatures = nConstantFeatures,
                });
            }
        }

        return tree.ToSklearnLikeTree();
    }
}
