using System;

namespace OptunaSharp.Importance.Forests.Internal.Cart;

/// <summary>
///     Port of <c>sklearn/tree/_criterion.pyx: RegressionCriterion + MSE</c> for the single-output
///     regression case (<c>n_outputs == 1</c>, the shape a fANOVA/MDI forest uses). Tracks the
///     weighted sums that let the impurity of the current split's left/right children be evaluated in
///     linear time. All accumulation is float64 in scikit-learn's exact order; the <c>**2.0</c> terms
///     compile to <c>pow(x, 2.0)</c> in Cython, hence <see cref="Math.Pow" /> (not <c>x*x</c>) here —
///     required for bit-exactness.
/// </summary>
internal sealed class MseRegressionCriterion
{
    internal double[] Y = null!;                 // y[sample] (n_outputs == 1)
    internal double[]? SampleWeight;             // bootstrap counts; null => all ones
    internal int[] SampleIndices = null!;        // the shared, in-place-permuted samples array

    internal int Start, End, Pos, NNodeSamples, NMissing;
    internal double WeightedNSamples, WeightedNNodeSamples, WeightedNLeft, WeightedNRight;
    internal double SqSumTotal, SumTotal, SumLeft, SumRight;

    /// <summary>Initialize the criterion on <c>sample_indices[start:end]</c> (<c>init</c> + <c>reset</c>).</summary>
    internal void Init(double[] y, double[]? sampleWeight, double weightedNSamples, int[] sampleIndices, int start, int end)
    {
        Y = y;
        SampleWeight = sampleWeight;
        SampleIndices = sampleIndices;
        Start = start;
        End = end;
        NNodeSamples = end - start;
        WeightedNSamples = weightedNSamples;
        WeightedNNodeSamples = 0.0;

        SqSumTotal = 0.0;
        SumTotal = 0.0;
        double w = 1.0;
        for (int p = start; p < end; p++)
        {
            int i = sampleIndices[p];
            if (sampleWeight != null)
                w = sampleWeight[i];

            double yik = y[i];
            double wYik = w * yik;
            SumTotal += wYik;
            SqSumTotal += wYik * yik;
            WeightedNNodeSamples += w;
        }

        Reset();
    }

    /// <summary><c>reset</c> — pos=start; all mass to the right child (no missing).</summary>
    internal void Reset()
    {
        Pos = Start;
        SumLeft = 0.0;
        SumRight = SumTotal;
        WeightedNLeft = 0.0;
        WeightedNRight = WeightedNNodeSamples;
    }

    /// <summary><c>reverse_reset</c> — pos=end; all mass to the left child (no missing).</summary>
    internal void ReverseReset()
    {
        Pos = End;
        SumRight = 0.0;
        SumLeft = SumTotal;
        WeightedNRight = 0.0;
        WeightedNLeft = WeightedNNodeSamples;
    }

    /// <summary><c>update(new_pos)</c> — move <c>sample_indices[pos:new_pos]</c> to the left child.</summary>
    internal void Update(int newPos)
    {
        int pos = Pos;
        int endNonMissing = End - NMissing;
        double w = 1.0;

        // Update from whichever side needs fewer operations (sklearn's exact branch).
        if ((newPos - pos) <= (endNonMissing - newPos))
        {
            for (int p = pos; p < newPos; p++)
            {
                int i = SampleIndices[p];
                if (SampleWeight != null)
                    w = SampleWeight[i];
                SumLeft += w * Y[i];
                WeightedNLeft += w;
            }
        }
        else
        {
            ReverseReset();
            for (int p = endNonMissing - 1; p >= newPos; p--)
            {
                int i = SampleIndices[p];
                if (SampleWeight != null)
                    w = SampleWeight[i];
                SumLeft -= w * Y[i];
                WeightedNLeft -= w;
            }
        }

        WeightedNRight = WeightedNNodeSamples - WeightedNLeft;
        SumRight = SumTotal - SumLeft;
        Pos = newPos;
    }

    /// <summary><c>MSE.node_impurity</c>.</summary>
    internal double NodeImpurity()
    {
        double impurity = SqSumTotal / WeightedNNodeSamples;
        impurity -= Math.Pow(SumTotal / WeightedNNodeSamples, 2.0);
        return impurity;
    }

    /// <summary><c>MSE.proxy_impurity_improvement</c>.</summary>
    internal double ProxyImpurityImprovement()
    {
        double proxyLeft = SumLeft * SumLeft;
        double proxyRight = SumRight * SumRight;
        return proxyLeft / WeightedNLeft + proxyRight / WeightedNRight;
    }

    /// <summary><c>MSE.children_impurity</c> (no missing).</summary>
    internal void ChildrenImpurity(out double impurityLeft, out double impurityRight)
    {
        double sqSumLeft = 0.0;
        double w = 1.0;
        for (int p = Start; p < Pos; p++)
        {
            int i = SampleIndices[p];
            if (SampleWeight != null)
                w = SampleWeight[i];
            double yik = Y[i];
            sqSumLeft += w * yik * yik;
        }

        double sqSumRight = SqSumTotal - sqSumLeft;
        impurityLeft = sqSumLeft / WeightedNLeft;
        impurityRight = sqSumRight / WeightedNRight;
        impurityLeft -= Math.Pow(SumLeft / WeightedNLeft, 2.0);
        impurityRight -= Math.Pow(SumRight / WeightedNRight, 2.0);
    }

    /// <summary><c>Criterion.impurity_improvement</c>.</summary>
    internal double ImpurityImprovement(double impurityParent, double impurityLeft, double impurityRight)
        => (WeightedNNodeSamples / WeightedNSamples) *
           (impurityParent
            - (WeightedNRight / WeightedNNodeSamples * impurityRight)
            - (WeightedNLeft / WeightedNNodeSamples * impurityLeft));

    /// <summary><c>RegressionCriterion.node_value</c> — the node's mean prediction.</summary>
    internal double NodeValue() => SumTotal / WeightedNNodeSamples;
}
