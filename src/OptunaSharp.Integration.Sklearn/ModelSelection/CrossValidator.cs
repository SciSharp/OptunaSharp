using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Samplers;

namespace OptunaSharp.Integration.Sklearn;

/// <summary>
///     Base class for the cross-validation splitters — the .NET analogue of scikit-learn's
///     <c>sklearn.model_selection.BaseCrossValidator</c>. A splitter turns a sample count into a sequence
///     of (train, test) index arrays and reports how many splits it produces.
/// </summary>
public abstract class BaseCrossValidator
{
    /// <summary>
    ///     The number of splitting iterations — scikit-learn's <c>get_n_splits(...)</c>.
    /// </summary>
    public abstract int GetNumSplits(int nSamples);

    /// <summary>
    ///     Generate the (train, test) index arrays. Both arrays are in <b>ascending</b> order (scikit-learn
    ///     yields <c>indices[mask]</c> over <c>arange(n)</c>, so ordering is ascending even when the splitter
    ///     shuffles which samples land in each fold).
    /// </summary>
    public abstract IEnumerable<(int[] Train, int[] Test)> Split(int nSamples);
}

/// <summary>
///     K-Folds cross-validator — a byte-exact port of <c>sklearn.model_selection.KFold</c>. Splits the
///     samples into <see cref="NSplits" /> consecutive folds (the first <c>n % k</c> folds get one extra
///     sample); each fold serves once as the validation set. With <see cref="Shuffle" /> the fold
///     membership is drawn from <c>check_random_state(random_state).shuffle(arange(n))</c> — NumSharp's
///     MT19937 <c>shuffle</c>, bit-identical to NumPy — but the emitted train/test indices are still
///     ascending.
/// </summary>
public sealed class KFold : BaseCrossValidator
{
    /// <summary>Number of folds. At least 2.</summary>
    public int NSplits { get; }

    /// <summary>Whether the samples are shuffled before being split into folds.</summary>
    public bool Shuffle { get; }

    /// <summary>Seed for the shuffle RNG (only meaningful when <see cref="Shuffle" /> is <c>true</c>).</summary>
    public int? RandomState { get; }

    /// <param name="nSplits">Number of folds. Must be at least 2. Defaults to <c>5</c>.</param>
    /// <param name="shuffle">Whether to shuffle the samples before splitting. Defaults to <c>false</c>.</param>
    /// <param name="randomState">
    ///     Seed for the shuffle RNG. Must be <c>null</c> unless <paramref name="shuffle" /> is <c>true</c>
    ///     (matching scikit-learn's guard).
    /// </param>
    public KFold(int nSplits = 5, bool shuffle = false, int? randomState = null)
    {
        if (nSplits <= 1)
            throw new ValueError(
                "k-fold cross-validation requires at least one train/test split by setting n_splits=2 " +
                $"or more, got n_splits={nSplits}.");

        if (!shuffle && randomState is not null)
            throw new ValueError(
                "Setting a random_state has no effect since shuffle is False. You should leave " +
                "random_state to its default (None), or set shuffle=True.");

        NSplits = nSplits;
        Shuffle = shuffle;
        RandomState = randomState;
    }

    /// <inheritdoc />
    public override int GetNumSplits(int nSamples) => NSplits;

    /// <inheritdoc />
    public override IEnumerable<(int[] Train, int[] Test)> Split(int nSamples)
    {
        if (NSplits > nSamples)
            throw new ValueError(
                $"Cannot have number of splits n_splits={NSplits} greater than the number of samples: " +
                $"n_samples={nSamples}.");

        // indices = arange(n); if shuffle: check_random_state(random_state).shuffle(indices)
        int[] indices;
        if (Shuffle)
        {
            var arr = np.arange(nSamples);
            new LazyRandomState(RandomState).Rng.shuffle(arr);
            indices = arr.ToArray<long>().Select(v => (int)v).ToArray();
        }
        else
        {
            indices = new int[nSamples];
            for (int i = 0; i < nSamples; i++)
                indices[i] = i;
        }

        // fold_sizes = full(n_splits, n // k); fold_sizes[: n % k] += 1
        var foldSizes = new int[NSplits];
        for (int i = 0; i < NSplits; i++)
            foldSizes[i] = nSamples / NSplits;
        for (int i = 0; i < nSamples % NSplits; i++)
            foldSizes[i] += 1;

        int current = 0;
        foreach (int foldSize in foldSizes)
        {
            int start = current, stop = current + foldSize;

            // test_mask[indices[start:stop]] = True, then train/test = arange(n)[~mask] / [mask] (ascending).
            var testMask = new bool[nSamples];
            for (int p = start; p < stop; p++)
                testMask[indices[p]] = true;

            var train = new List<int>(nSamples - foldSize);
            var test = new List<int>(foldSize);
            for (int i = 0; i < nSamples; i++)
            {
                if (testMask[i]) test.Add(i);
                else train.Add(i);
            }

            yield return (train.ToArray(), test.ToArray());
            current = stop;
        }
    }
}

/// <summary>
///     A cross-validator over a fixed, caller-supplied list of (train, test) index splits — the analogue
///     of scikit-learn's <c>check_cv</c> wrapping an iterable of index pairs (<c>_CVIterableWrapper</c>).
/// </summary>
public sealed class PredefinedSplits : BaseCrossValidator
{
    private readonly (int[] Train, int[] Test)[] _splits;

    public PredefinedSplits(IEnumerable<(int[] Train, int[] Test)> splits)
        => _splits = splits?.ToArray() ?? throw new ArgumentNullException(nameof(splits));

    public override int GetNumSplits(int nSamples) => _splits.Length;

    public override IEnumerable<(int[] Train, int[] Test)> Split(int nSamples) => _splits;
}

/// <summary>
///     Port of <c>sklearn.model_selection.check_cv</c> for the regression path (<c>classifier=false</c>):
///     builds a <see cref="BaseCrossValidator" /> from the flexible <c>cv</c> argument. <c>null</c> → 5-fold
///     <see cref="KFold" />; an <see cref="int" /> → that many folds; an existing
///     <see cref="BaseCrossValidator" /> passes through; an iterable of index pairs is wrapped as
///     <see cref="PredefinedSplits" />.
///
///     <para>The classifier branch (StratifiedKFold when <c>y</c> is binary/multiclass) is not part of the
///     regression-focused first cut.</para>
/// </summary>
public static partial class CrossValidation
{
    public static BaseCrossValidator CheckCv(object? cv = null)
    {
        switch (cv)
        {
            case null:
                return new KFold(5);
            case BaseCrossValidator validator:
                return validator;
            case int n:
                return new KFold(n);
            case IEnumerable<(int[] Train, int[] Test)> iterable:
                return new PredefinedSplits(iterable);
            default:
                throw new ValueError(
                    "Expected `cv` as an integer, a cross-validation object, or an iterable yielding " +
                    $"(train, test) splits as arrays of indices. Got {cv}.");
        }
    }
}
