namespace OptunaSharp.Importance.Forests.Internal;

/// <summary>
///     A decision tree in the flat, array-of-nodes representation that scikit-learn's
///     <c>sklearn.tree._tree.Tree</c> exposes and that <see cref="FanovaTree" /> (a port of
///     <c>optuna/importance/_fanova/_tree.py</c>) and <see cref="Mdi" /> read. It is deliberately
///     backend-agnostic: it can be built by hand (the port of Optuna's own <c>test_tree.py</c> mock
///     tree), produced bit-for-bit by the native scikit-learn CART port
///     (<c>OptunaSharp.Importance.Forests.Internal.Cart</c>), or adapted from an ML.NET
///     <c>RegressionTree</c> in the <c>OptunaSharp.Importance.Forests</c> satellite package.
///
///     <para>Nodes are numbered <c>0..NodeCount-1</c> with node 0 the root, and every parent has a
///     smaller index than its children (pre-order) — the ordering the fANOVA precompute loops and
///     MDI rely on. A leaf has <see cref="Feature" /> <c>&lt; 0</c> and <see cref="ChildrenLeft" /> /
///     <see cref="ChildrenRight" /> <c>== -1</c>.</para>
/// </summary>
internal sealed class SklearnLikeTree
{
    /// <summary>Number of features the tree was built over (= <c>search_spaces.shape[0]</c>).</summary>
    internal int NFeatures { get; }

    /// <summary>Split feature per node; <c>&lt; 0</c> for a leaf (<c>sklearn.tree._tree.Tree.feature</c>).</summary>
    internal int[] Feature { get; }

    /// <summary>Split threshold per node; unused for a leaf (<c>Tree.threshold</c>).</summary>
    internal double[] Threshold { get; }

    /// <summary>Left-child node index per node; <c>-1</c> for a leaf (<c>Tree.children_left</c>).</summary>
    internal int[] ChildrenLeft { get; }

    /// <summary>Right-child node index per node; <c>-1</c> for a leaf (<c>Tree.children_right</c>).</summary>
    internal int[] ChildrenRight { get; }

    /// <summary>
    ///     Prediction value per node. Only leaf values are read by fANOVA (the flattened
    ///     <c>Tree.value[node].reshape(-1)[0]</c>); internal-node entries are ignored there.
    /// </summary>
    internal double[] Value { get; }

    /// <summary>
    ///     Per-node impurity (<c>Tree.impurity</c>). Populated by the native CART (the exact
    ///     criterion value) and by the ML.NET adapter (routed training-set variance); consumed by
    ///     <see cref="Mdi" />. <c>null</c> for hand-built trees that carry no statistics.
    /// </summary>
    internal double[]? Impurity { get; }

    /// <summary>
    ///     Per-node weighted sample count (<c>Tree.weighted_n_node_samples</c>). Populated alongside
    ///     <see cref="Impurity" />; consumed by <see cref="Mdi" />.
    /// </summary>
    internal double[]? WeightedNNodeSamples { get; }

    /// <summary><c>Tree.node_count</c>.</summary>
    internal int NodeCount => Feature.Length;

    internal SklearnLikeTree(
        int nFeatures, int[] feature, int[] childrenLeft, int[] childrenRight, double[] threshold, double[] value,
        double[]? impurity = null, double[]? weightedNNodeSamples = null)
    {
        NFeatures = nFeatures;
        Feature = feature;
        ChildrenLeft = childrenLeft;
        ChildrenRight = childrenRight;
        Threshold = threshold;
        Value = value;
        Impurity = impurity;
        WeightedNNodeSamples = weightedNNodeSamples;
    }
}
