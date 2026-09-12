using System;

namespace OptunaSharp.Importance.Forests.Internal.Cart;

/// <summary>Port of <c>sklearn/tree/_splitter.pxd: SplitRecord</c> (regression fields).</summary>
internal struct SplitRecord
{
    /// <summary>The zero-based feature index selected for the split.</summary>
    internal int Feature;
    /// <summary>The exclusive endpoint of the left child in the shared sample-index array.</summary>
    internal int Pos;
    /// <summary>The number of missing feature values associated with the split.</summary>
    internal int NMissing;
    /// <summary>The feature cutoff; values at or below it route left.</summary>
    internal double Threshold;
    /// <summary>The weighted decrease in impurity achieved by the split.</summary>
    internal double Improvement;
    /// <summary>The weighted target variance in the left child.</summary>
    internal double ImpurityLeft;
    /// <summary>The weighted target variance in the right child.</summary>
    internal double ImpurityRight;
    /// <summary>Whether a missing feature value follows the left child during prediction.</summary>
    internal bool MissingGoToLeft;
}

/// <summary>
///     Port of <c>sklearn/tree/_partitioner.pyx: DensePartitioner</c> (the dense, no-missing path
///     the fANOVA/MDI forest uses) plus the module-level <c>sort</c> (introsort with median-of-3
///     three-way partition and a heapsort fallback). Feature values are copied into a float32 buffer
///     — matching scikit-learn casting X to float32 — and sorted with the sample indices in
///     lock-step. The sort is verified bit-exact (sample permutation included) against sklearn's
///     <c>_py_sort</c> hook.
/// </summary>
internal sealed class DensePartitioner
{
    /// <summary>The tolerance below which adjacent float32 feature values are treated as the same split candidate.</summary>
    internal const float FeatureThreshold = 1e-7f;

    /// <summary>The shared float32 training feature matrix, indexed by original sample row and feature column.</summary>
    internal float[,] X = null!;          // float32 (n_samples, n_features)
    /// <summary>The shared permutation of original sample-row indices.</summary>
    internal int[] Samples = null!;       // the shared, in-place-permuted samples array
    /// <summary>The reusable buffer of values for the feature currently being sorted.</summary>
    internal float[] FeatureValues = null!;
    /// <summary>The current half-open node range and its missing-value count in the shared sample array.</summary>
    internal int Start, End, NMissing;

    /// <summary>Selects the node's half-open sample range and clears its missing-value count.</summary>
    /// <param name="start">The inclusive start of the interval or slice.</param>
    /// <param name="end">The exclusive endpoint of the sample-index slice.</param>
    internal void InitNodeSplit(int start, int end)
    {
        Start = start;
        End = end;
        NMissing = 0;
    }

    /// <summary><c>sort_samples_and_feature_values</c> (no missing).</summary>
    /// <param name="feature">The zero-based feature column whose splits or importance are requested.</param>
    internal void SortSamplesAndFeatureValues(int feature)
    {
        for (int i = Start; i < End; i++)
            FeatureValues[i] = X[Samples[i], feature];
        Sort(FeatureValues, Samples, Start, End - Start);
        NMissing = 0;
    }

    /// <summary><c>next_p</c> — advance past feature values within FEATURE_THRESHOLD of each other.</summary>
    /// <param name="pPrev">Receives the final index of the current group of nearly equal feature values.</param>
    /// <param name="p">The bit position or cursor used by the operation.</param>
    internal void NextP(ref int pPrev, ref int p)
    {
        int endNonMissing = End - NMissing;
        while (p + 1 < endNonMissing && FeatureValues[p + 1] <= FeatureValues[p] + FeatureThreshold)
            p++;
        pPrev = p;
        p++;
    }

    /// <summary><c>partition_samples_final</c> (no missing).</summary>
    /// <param name="bestThresholdFeature">The feature column chosen by the best split.</param>
    /// <param name="bestThreshold">The split threshold; values at or below it go to the left child.</param>
    /// <param name="bestNMissing">The number of missing feature values kept outside the ordinary partition range.</param>
    internal void PartitionSamplesFinal(int bestThresholdFeature, double bestThreshold, int bestNMissing)
    {
        int start = Start;
        int p = start;
        int end = End - 1;
        int partitionEnd = end - bestNMissing;

        while (p < partitionEnd)
        {
            if (X[Samples[p], bestThresholdFeature] <= bestThreshold)
                p++;
            else
            {
                (Samples[p], Samples[partitionEnd]) = (Samples[partitionEnd], Samples[p]);
                partitionEnd--;
            }
        }
    }

    // ---- introsort (sklearn/tree/_partitioner.pyx: sort / introsort / median3 / heapsort) ----

    /// <summary>Sorts feature values and sample indices together using the scikit-learn-compatible introsort path.</summary>
    /// <param name="featureValues">The float32 feature-value buffer sorted together with its sample indices.</param>
    /// <param name="samples">The original sample indices reordered in place together with their feature values.</param>
    /// <param name="baseIdx">The starting index of the slice being sorted.</param>
    /// <param name="n">The number of elements or points to process.</param>
    private static void Sort(float[] featureValues, int[] samples, int baseIdx, int n)
    {
        if (n == 0)
            return;
        int maxd = 2 * (int)Math.Log2(n);
        Introsort(featureValues, samples, baseIdx, n, maxd);
    }

    /// <summary>Exchanges feature values and their associated original-sample indices together.</summary>
    /// <param name="fv">The feature values sorted in place together with their sample indices.</param>
    /// <param name="s">The original sample-row indices permuted together with the feature values.</param>
    /// <param name="i">The first zero-based position involved in the operation.</param>
    /// <param name="j">The second zero-based position involved in the operation.</param>
    private static void Swap(float[] fv, int[] s, int i, int j)
    {
        (fv[i], fv[j]) = (fv[j], fv[i]);
        (s[i], s[j]) = (s[j], s[i]);
    }

    /// <summary>Returns the median of the first, middle, and last values used as the sort pivot.</summary>
    /// <param name="fv">The feature values sorted in place together with their sample indices.</param>
    /// <param name="b">The second numerical operand; its dimensions must agree with the first operand.</param>
    /// <param name="n">The number of elements or points to process.</param>
    /// <returns>The median of the first, middle, and last values used as the sort pivot.</returns>
    private static float Median3(float[] fv, int b, int n)
    {
        float a = fv[b], mid = fv[b + n / 2], c = fv[b + n - 1];
        if (a < mid) { if (mid < c) return mid; else if (a < c) return c; else return a; }
        else if (mid < c) { if (a < c) return a; else return c; }
        else return mid;
    }

    /// <summary>Partitions feature values with median-of-three pivots, falling back to heapsort when the recursion budget is exhausted.</summary>
    /// <param name="fv">The feature values sorted in place together with their sample indices.</param>
    /// <param name="s">The original sample-row indices permuted together with the feature values.</param>
    /// <param name="b">The second numerical operand; its dimensions must agree with the first operand.</param>
    /// <param name="n">The number of elements or points to process.</param>
    /// <param name="maxd">The remaining introsort recursion budget before heapsort is used.</param>
    private static void Introsort(float[] fv, int[] s, int b, int n, int maxd)
    {
        while (n > 1)
        {
            if (maxd <= 0) { Heapsort(fv, s, b, n); return; }
            maxd -= 1;

            float pivot = Median3(fv, b, n);

            int i = 0, l = 0, r = n;
            while (i < r)
            {
                if (fv[b + i] < pivot) { Swap(fv, s, b + i, b + l); i++; l++; }
                else if (fv[b + i] > pivot) { r--; Swap(fv, s, b + i, b + r); }
                else i++;
            }

            Introsort(fv, s, b, l, maxd);
            b += r;
            n -= r;
        }
    }

    /// <summary>Restores the max-heap invariant while keeping feature values paired with their sample indices.</summary>
    /// <param name="fv">The feature values sorted in place together with their sample indices.</param>
    /// <param name="s">The original sample-row indices permuted together with the feature values.</param>
    /// <param name="b">The second numerical operand; its dimensions must agree with the first operand.</param>
    /// <param name="start">The inclusive start of the interval or slice.</param>
    /// <param name="end">The exclusive endpoint of the sample-index slice.</param>
    private static void SiftDown(float[] fv, int[] s, int b, int start, int end)
    {
        int root = start;
        while (true)
        {
            int child = root * 2 + 1;
            int maxind = root;
            if (child < end && fv[b + maxind] < fv[b + child]) maxind = child;
            if (child + 1 < end && fv[b + maxind] < fv[b + child + 1]) maxind = child + 1;
            if (maxind == root) break;
            Swap(fv, s, b + root, b + maxind);
            root = maxind;
        }
    }

    /// <summary>Sorts the remaining partition with bounded worst-case work when quicksort becomes too deep.</summary>
    /// <param name="fv">The feature values sorted in place together with their sample indices.</param>
    /// <param name="s">The original sample-row indices permuted together with the feature values.</param>
    /// <param name="b">The second numerical operand; its dimensions must agree with the first operand.</param>
    /// <param name="n">The number of elements or points to process.</param>
    private static void Heapsort(float[] fv, int[] s, int b, int n)
    {
        int start = (n - 2) / 2;
        int end = n;
        while (true)
        {
            SiftDown(fv, s, b, start, end);
            if (start == 0) break;
            start--;
        }
        end = n - 1;
        while (end > 0)
        {
            Swap(fv, s, b, b + end);
            SiftDown(fv, s, b, 0, end);
            end--;
        }
    }
}
