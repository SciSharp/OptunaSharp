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
    /// <summary>The child-index sentinel identifying a leaf node.</summary>
    internal const int TreeLeaf = -1;        // sklearn TREE_LEAF
    /// <summary>The sentinel for fields that have no meaning on a leaf or root.</summary>
    internal const int TreeUndefined = -2;   // sklearn TREE_UNDEFINED

    private readonly int _nFeatures;
    private readonly List<int> _feature = new(), _childrenLeft = new(), _childrenRight = new(), _nNodeSamples = new();
    private readonly List<double> _threshold = new(), _impurity = new(), _weightedNNodeSamples = new(), _value = new();

    /// <summary>Initializes an append-only node builder for the specified number of feature columns.</summary>
    /// <param name="nFeatures">The number of feature columns in the training matrix.</param>
    internal CartNodeTree(int nFeatures) => _nFeatures = nFeatures;

    /// <summary>The number of nodes appended to the tree builder.</summary>
    internal int NodeCount => _feature.Count;

    /// <summary><c>Tree._add_node</c>. Returns the new node id.</summary>
    /// <param name="parent">The parent node index, or the root sentinel when creating the first node.</param>
    /// <param name="isLeft">Whether the new node is the parent's left child.</param>
    /// <param name="isLeaf">Whether the node terminates prediction without a further split.</param>
    /// <param name="feature">The zero-based feature column whose splits or importance are requested.</param>
    /// <param name="threshold">The incumbent objective value that a candidate must improve upon, in the model's maximization convention.</param>
    /// <param name="impurity">The weighted target variance at the parent node.</param>
    /// <param name="nNodeSamples">The number of training rows reaching this node.</param>
    /// <param name="weightedNNodeSamples">The total sample weight reaching this node.</param>
    /// <returns>The zero-based index assigned to the appended tree node.</returns>
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

    /// <summary>Stores the weighted-mean prediction associated with a completed node.</summary>
    /// <param name="nodeId">The zero-based node index to update.</param>
    /// <param name="value">The value to convert, format, or compare using the operation's stated convention.</param>
    internal void SetValue(int nodeId, double value) => _value[nodeId] = value;

    /// <summary>Returns a snapshot of the constructed nodes in the flat sklearn-compatible tree representation.</summary>
    /// <returns>A snapshot of the constructed nodes in the flat sklearn-compatible tree representation.</returns>
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
    /// <summary>The double-precision rounding allowance used when deciding whether a split improves impurity enough.</summary>
    internal const double Epsilon = 2.220446049250313e-16;  // np.finfo(double).eps

    private struct StackRecord
    {
        /// <summary>The pending node's half-open sample range, tree depth, and parent index.</summary>
        internal int Start, End, Depth, Parent;
        /// <summary>Whether the pending node should be attached to its parent's left child slot.</summary>
        internal bool IsLeft;
        /// <summary>The target variance already computed for the pending node.</summary>
        internal double Impurity;
        /// <summary>The count of feature columns known to be constant on the ancestor path.</summary>
        internal int NConstantFeatures;
    }

    /// <summary>Returns the regression tree produced by depth-first best-split expansion.</summary>
    /// <param name="x">Feature values with one sample per row and one feature per column.</param>
    /// <param name="y">Target values aligned with the feature rows.</param>
    /// <param name="sampleWeight">The non-null weight of every original training row, including bootstrap multiplicities.</param>
    /// <param name="nFeatures">The number of feature columns in the training matrix.</param>
    /// <param name="maxDepth">The maximum tree depth; the root has depth zero.</param>
    /// <param name="minSamplesSplit">The minimum number of samples required to attempt splitting a node.</param>
    /// <param name="minSamplesLeaf">The minimum number of samples required in either child after a split.</param>
    /// <param name="minWeightLeaf">The minimum sample-weight mass required in either child.</param>
    /// <param name="minImpurityDecrease">The minimum weighted impurity improvement required to accept a split.</param>
    /// <param name="randRSeed">The state used to seed the splitter's scikit-learn-compatible random generator.</param>
    /// <param name="maxFeatures">The number of features considered at each split; negative selects all features where supported.</param>
    /// <returns>The regression tree produced by depth-first best-split expansion.</returns>
    internal static SklearnLikeTree Build(
        float[,] x, double[] y, double[] sampleWeight, int nFeatures,
        int maxDepth, int minSamplesSplit, int minSamplesLeaf, double minWeightLeaf, double minImpurityDecrease,
        uint randRSeed, int maxFeatures = -1)
    {
        var tree = new CartNodeTree(nFeatures);
        var splitter = new BestSplitter(minSamplesLeaf, minWeightLeaf);
        splitter.Init(x, y, sampleWeight, nFeatures, randRSeed, maxFeatures);

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
