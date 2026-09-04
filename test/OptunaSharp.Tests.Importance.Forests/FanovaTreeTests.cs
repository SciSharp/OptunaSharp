using System.Collections.Generic;
using AwesomeAssertions;
using OptunaSharp.Importance.Forests.Internal;

namespace OptunaSharp.Tests.Importance.Forests;

/// <summary>
///     Port of Optuna's <c>tests/importance_tests/fanova_tests/test_tree.py</c>. The fANOVA tree
///     algorithm (<see cref="FanovaTree" />, a port of <c>_fanova/_tree.py</c>) is exercised on the
///     exact hand-built mock tree that Optuna's own test uses — so this validates the port with
///     <b>no random forest involved</b>. Expected values are the authoritative outputs of Optuna's
///     <c>_FanovaTree</c> on the same tree (captured from the pinned optuna 5.0.0.dev). The tree
///     path never touches <c>erf</c>/LAPACK, so agreement is to machine precision.
/// </summary>
[TestClass]
public class FanovaTreeTests
{
    // The mock sklearn tree from test_tree.py:
    //   n_features=3, node_count=5, feature=[1,2,-1,-1,-1],
    //   children_left=[1,2,-1,-1,-1], children_right=[4,3,-1,-1,-1],
    //   value=[-1,-1,0.1,0.2,0.5], threshold=[0.5,1.5,-1,-1,-1],
    //   search_spaces=[[0,1],[0,1],[0,2]].
    private static SklearnLikeTree MakeSklearnTree() => new(
        nFeatures: 3,
        feature: new[] { 1, 2, -1, -1, -1 },
        childrenLeft: new[] { 1, 2, -1, -1, -1 },
        childrenRight: new[] { 4, 3, -1, -1, -1 },
        threshold: new[] { 0.5, 1.5, -1.0, -1.0, -1.0 },
        value: new[] { -1.0, -1.0, 0.1, 0.2, 0.5 });

    private static readonly double[,] SearchSpaces =
        { { 0.0, 1.0 }, { 0.0, 1.0 }, { 0.0, 2.0 } };

    private static FanovaTree MakeTree() => new(MakeSklearnTree(), SearchSpaces);

    private const double Tol = 1e-12;

    [TestMethod]
    public void Variance()
    {
        MakeTree().Variance.Should().BeApproximately(0.036093749999999994, Tol);
    }

    [TestMethod]
    public void Statistics()
    {
        // Per node: (weighted value, sum of weights).
        var expected = new[]
        {
            (0.3125, 2.0),
            (0.125, 1.0),
            (0.1, 0.75),
            (0.2, 0.25),
            (0.5, 1.0),
        };
        var stats = MakeTree().Statistics;
        for (int node = 0; node < expected.Length; node++)
        {
            stats[node, 0].Should().BeApproximately(expected[node].Item1, Tol, $"value of node {node}");
            stats[node, 1].Should().BeApproximately(expected[node].Item2, Tol, $"weight of node {node}");
        }
    }

    [TestMethod]
    public void SplitMidpoints()
    {
        var tree = MakeTree();
        tree.SplitMidpoints(0).Should().Equal(0.5);
        tree.SplitMidpoints(1).Should().Equal(0.25, 0.75);
        tree.SplitMidpoints(2).Should().Equal(0.75, 1.75);
    }

    [TestMethod]
    public void SplitSizes()
    {
        var tree = MakeTree();
        tree.SplitSizes(0).Should().Equal(1.0);
        tree.SplitSizes(1).Should().Equal(0.5, 0.5);
        tree.SplitSizes(2).Should().Equal(1.5, 0.5);
    }

    [TestMethod]
    public void SubtreeActiveFeatures()
    {
        var expected = new[]
        {
            new[] { false, true, true },
            new[] { false, false, true },
            new[] { false, false, false },
            new[] { false, false, false },
            new[] { false, false, false },
        };
        var saf = MakeTree().SubtreeActiveFeatures;
        for (int node = 0; node < expected.Length; node++)
            for (int f = 0; f < 3; f++)
                saf[node, f].Should().Be(expected[node][f], $"subtree_active[{node},{f}]");
    }

    [TestMethod]
    public void TreeAttrs()
    {
        var tree = MakeTree();
        tree.NFeaturesInternal.Should().Be(3);
        tree.NNodesInternal.Should().Be(5);

        var sk = MakeSklearnTree();
        // leaf iff feature < 0.
        (sk.Feature[0] < 0).Should().BeFalse();
        (sk.Feature[1] < 0).Should().BeFalse();
        (sk.Feature[2] < 0).Should().BeTrue();
        (sk.Feature[3] < 0).Should().BeTrue();
        (sk.Feature[4] < 0).Should().BeTrue();

        sk.ChildrenLeft.Should().Equal(1, 2, -1, -1, -1);
        sk.ChildrenRight.Should().Equal(4, 3, -1, -1, -1);
        sk.Value.Should().Equal(-1.0, -1.0, 0.1, 0.2, 0.5);
        sk.Threshold.Should().Equal(0.5, 1.5, -1.0, -1.0, -1.0);
        sk.Feature[0].Should().Be(1);
        sk.Feature[1].Should().Be(2);
    }

    [TestMethod]
    public void GetNodeSubspaces()
    {
        var tree = MakeTree();

        // Splitting on the second feature, first node.
        tree.GetNodeLeftChildSubspaces(0, SearchSpaces).Should().BeEquivalentTo(
            new[,] { { 0.0, 1.0 }, { 0.0, 0.5 }, { 0.0, 2.0 } });
        tree.GetNodeRightChildSubspaces(0, SearchSpaces).Should().BeEquivalentTo(
            new[,] { { 0.0, 1.0 }, { 0.5, 1.0 }, { 0.0, 2.0 } });

        // Splitting on the third feature, second node.
        tree.GetNodeLeftChildSubspaces(1, SearchSpaces).Should().BeEquivalentTo(
            new[,] { { 0.0, 1.0 }, { 0.0, 1.0 }, { 0.0, 1.5 } });
        tree.GetNodeRightChildSubspaces(1, SearchSpaces).Should().BeEquivalentTo(
            new[,] { { 0.0, 1.0 }, { 0.0, 1.0 }, { 1.5, 2.0 } });

        // The input search space must not be mutated.
        SearchSpaces.Should().BeEquivalentTo(new[,] { { 0.0, 1.0 }, { 0.0, 1.0 }, { 0.0, 2.0 } });
    }

    [TestMethod]
    public void GetMarginalVariance()
    {
        var tree = MakeTree();
        var cases = new Dictionary<long[], double>
        {
            [new long[] { 0 }] = 0.0,
            [new long[] { 1 }] = 0.03515625,
            [new long[] { 2 }] = 0.0004687499999999998,
            [new long[] { 0, 1 }] = 0.03515625,
            [new long[] { 0, 2 }] = 0.0004687499999999998,
            [new long[] { 1, 2 }] = 0.036093749999999994,
            [new long[] { 0, 1, 2 }] = 0.036093749999999994,
        };
        foreach (var (features, expected) in cases)
            tree.GetMarginalVariance(features).Should().BeApproximately(
                expected, Tol, $"marginal variance of [{string.Join(",", features)}]");
    }

    [TestMethod]
    public void GetMarginalizedStatistics()
    {
        var tree = MakeTree();
        double n = double.NaN;
        // (feature vector) -> (value, weight), from Optuna's _get_marginalized_statistics.
        var cases = new (double[] Fv, double Value, double Weight)[]
        {
            (new[] { 0.5, n, n }, 0.3125, 2.0),
            (new[] { n, 0.25, n }, 0.125, 2.0),
            (new[] { n, 0.75, n }, 0.5, 2.0),
            (new[] { n, n, 0.75 }, 0.3, 1.0),
            (new[] { n, n, 1.75 }, 0.35, 1.0),
            (new[] { 0.5, 0.25, n }, 0.125, 2.0),
            (new[] { 0.5, 0.75, n }, 0.5, 2.0),
            (new[] { 0.5, n, 0.75 }, 0.3, 1.0),
            (new[] { 0.5, n, 1.75 }, 0.35, 1.0),
            (new[] { n, 0.25, 0.75 }, 0.1, 1.0),
            (new[] { n, 0.25, 1.75 }, 0.2, 1.0),
            (new[] { n, 0.75, 0.75 }, 0.5, 1.0),
            (new[] { n, 0.75, 1.75 }, 0.5, 1.0),
            (new[] { 0.5, 0.25, 0.75 }, 0.1, 1.0),
            (new[] { 0.5, 0.25, 1.75 }, 0.2, 1.0),
            (new[] { 0.5, 0.75, 0.75 }, 0.5, 1.0),
            (new[] { 0.5, 0.75, 1.75 }, 0.5, 1.0),
        };
        foreach (var (fv, value, weight) in cases)
        {
            var (v, w) = tree.GetMarginalizedStatistics(fv);
            v.Should().BeApproximately(value, Tol, $"value for [{Describe(fv)}]");
            w.Should().BeApproximately(weight, Tol, $"weight for [{Describe(fv)}]");
        }
    }

    private static string Describe(double[] fv)
    {
        var parts = new List<string>();
        foreach (double x in fv)
            parts.Add(double.IsNaN(x) ? "nan" : x.ToString(System.Globalization.CultureInfo.InvariantCulture));
        return string.Join(",", parts);
    }
}
