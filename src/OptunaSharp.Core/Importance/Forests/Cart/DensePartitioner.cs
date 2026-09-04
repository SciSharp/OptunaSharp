using System;

namespace OptunaSharp.Importance.Forests.Internal.Cart;

/// <summary>Port of <c>sklearn/tree/_splitter.pxd: SplitRecord</c> (regression fields).</summary>
internal struct SplitRecord
{
    internal int Feature;
    internal int Pos;
    internal int NMissing;
    internal double Threshold;
    internal double Improvement;
    internal double ImpurityLeft;
    internal double ImpurityRight;
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
    internal const float FeatureThreshold = 1e-7f;

    internal float[,] X = null!;          // float32 (n_samples, n_features)
    internal int[] Samples = null!;       // the shared, in-place-permuted samples array
    internal float[] FeatureValues = null!;
    internal int Start, End, NMissing;

    internal void InitNodeSplit(int start, int end)
    {
        Start = start;
        End = end;
        NMissing = 0;
    }

    /// <summary><c>sort_samples_and_feature_values</c> (no missing).</summary>
    internal void SortSamplesAndFeatureValues(int feature)
    {
        for (int i = Start; i < End; i++)
            FeatureValues[i] = X[Samples[i], feature];
        Sort(FeatureValues, Samples, Start, End - Start);
        NMissing = 0;
    }

    /// <summary><c>next_p</c> — advance past feature values within FEATURE_THRESHOLD of each other.</summary>
    internal void NextP(ref int pPrev, ref int p)
    {
        int endNonMissing = End - NMissing;
        while (p + 1 < endNonMissing && FeatureValues[p + 1] <= FeatureValues[p] + FeatureThreshold)
            p++;
        pPrev = p;
        p++;
    }

    /// <summary><c>partition_samples_final</c> (no missing).</summary>
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

    private static void Sort(float[] featureValues, int[] samples, int baseIdx, int n)
    {
        if (n == 0)
            return;
        int maxd = 2 * (int)Math.Log2(n);
        Introsort(featureValues, samples, baseIdx, n, maxd);
    }

    private static void Swap(float[] fv, int[] s, int i, int j)
    {
        (fv[i], fv[j]) = (fv[j], fv[i]);
        (s[i], s[j]) = (s[j], s[i]);
    }

    private static float Median3(float[] fv, int b, int n)
    {
        float a = fv[b], mid = fv[b + n / 2], c = fv[b + n - 1];
        if (a < mid) { if (mid < c) return mid; else if (a < c) return c; else return a; }
        else if (mid < c) { if (a < c) return a; else return c; }
        else return mid;
    }

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
