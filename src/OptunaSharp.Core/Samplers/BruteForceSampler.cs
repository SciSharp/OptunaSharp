using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers;

/// <summary>
///     Sampler that performs exhaustive search over the define-by-run search space.
///     Port of <c>optuna/samplers/_brute_force.py: BruteForceSampler</c> (experimental since 3.1.0).
///
///     <para>This sampler performs exhaustive search on the defined search space. Each suggestion
///     builds a tree over the trials observed so far — each internal node is a parameter and its
///     children are keyed by the parameter's candidate values (internal repr); a root-to-leaf path
///     is one complete <c>params</c> — then samples an unexplored branch, blending exact-uniform
///     with flat-uniform weighting to prevent starvation of unexplored branches. It stops the study
///     (internally invoking <see cref="Study.Stop" />) once the space is exhausted.</para>
/// </summary>
/// <example>
///     <code>
///     double Objective(Trial trial)
///     {
///         var c = trial.SuggestCategorical("c", new CategoricalChoice[] { "float", "int" }).Value as string;
///         if (c == "float")
///             return trial.SuggestFloat("x", 1, 3, step: 0.5);
///         long a = trial.SuggestInt("a", 1, 3);
///         long b = trial.SuggestInt("b", a, 3);
///         return a + b;
///     }
///
///     var study = Optuna.CreateStudy(sampler: new BruteForceSampler());
///     study.Optimize(Objective);
///     </code>
/// </example>
/// <remarks>
///     The defined search space must be finite. Therefore, when using
///     <see cref="FloatDistribution" /> or <see cref="Trial.SuggestFloat(string, double, double, double?, bool)" />,
///     <c>step = null</c> is not allowed (a <see cref="ValueError" /> is raised). The sampler may fail
///     to try the entire search space when the suggestion ranges or parameters are changed in the
///     same <see cref="Study" />.
/// </remarks>
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.BruteForceSampler.html">Official Optuna BruteForceSampler reference.</seealso>
public class BruteForceSampler : BaseSampler
{
    private readonly LazyRandomState _rng;
    private readonly bool _avoidPrematureStop;

    /// <param name="seed">
    ///     A seed to fix the order of trials as the search order is randomly shuffled. Please note
    ///     that it is not recommended using this option in distributed optimization settings since
    ///     this option cannot ensure the order of trials and may increase the number of duplicate
    ///     suggestions during distributed optimization.
    /// </param>
    /// <param name="avoidPrematureStop">
    ///     If <c>true</c>, the sampler performs a strict exhaustive search. Please note that enabling
    ///     this option may increase the likelihood of duplicate sampling. When this option is not
    ///     enabled (default), the sampler applies a looser criterion for determining when to stop the
    ///     search, which may result in incomplete coverage of the search space.
    /// </param>
    /// <summary>Configures traversal of finite parameter choices discovered while the objective runs.</summary>
    public BruteForceSampler(int? seed = null, bool avoidPrematureStop = false)
    {
        _rng = new LazyRandomState(seed);
        _avoidPrematureStop = avoidPrematureStop;
    }

    /// <inheritdoc />
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
        => new();

    /// <inheritdoc />
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
        => new();

    /// <inheritdoc />
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        bool excludeRunning = !_avoidPrematureStop;
        var (trials, currentIdx) = GetNonWaitingTrialsAndCurrentTrialIndex(study, trial.Number);
        trials.RemoveAt(currentIdx);
        var tree = new TreeNode();
        var cArgs = ArgsForDistribution(paramDistribution);
        tree.Expand(paramName, cArgs);
        // Populating must happen after the initialization above to prevent `tree` from being
        // initialized as an empty graph, which is created with n_jobs > 1 where we get
        // trials[i].params = {} for some i.
        PopulateTree(tree, trials, trial.Params);
        double paramVal;
        if (tree.IsAnyExpandable(excludeRunning))
        {
            paramVal = tree.SampleChild(_rng.Rng, excludeRunning);
        }
        else
        {
            var choices = EnumerateCandidates(cArgs);
            // numpy's choice(choices) with no size/replace/p reduces to choices[randint(0, len)].
            paramVal = choices[(int)_rng.Rng.randint(0, choices.Count)];
        }

        return paramDistribution.ToExternalRepr(paramVal);
    }

    /// <inheritdoc />
    /// <exception cref="NumSharp.RuntimeError">The search tree is exhausted and Study.Stop is invoked outside Study.Optimize.</exception>
    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
    {
        bool excludeRunning = !_avoidPrematureStop;
        var (trials, currentIdx) = GetNonWaitingTrialsAndCurrentTrialIndex(study, trial.Number);
        // Set current trial as complete.
        trials[currentIdx] = FrozenTrial.CreateTrial(
            state: state, values: values, @params: trial.ParamsDict, distributions: trial.DistributionsDict);

        // This routine checks whether any existing branch of the current trial is expandable
        // (unexplored). For example, if `params={"a": 1, "b": 2, "c": 3}`, we first check whether
        // all possible params represented by `{"a": 1, "b": 2} | X` exist. If there is any
        // unexplored `X` based on the search space, we still need to evaluate such `X`, so the study
        // does not have to be stopped yet. This search is much faster compared to a full tree build
        // because there are much fewer trials among all that take `{"a": 1, "b": 2}`. If such `X` is
        // already exhaustively searched, the condition is relaxed to `{"a": 1} | X`.
        var @params = new OrderedDict<string, object?>(trial.ParamsDict);
        var paramNames = new List<string>(trial.ParamsDict.Keys);
        for (int i = paramNames.Count - 1; i >= 0; i--)
        {
            @params.Remove(paramNames[i]);
            var tree = new TreeNode();
            PopulateTree(tree, trials, @params);
            if (tree.IsAnyExpandable(excludeRunning))
                return;
        }

        study.Stop();
    }

    /// <summary>Python <c>_get_non_waiting_trials_and_current_trial_index</c>.</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="currentTrialNumber">Public number identifying the current trial within the history.</param>
    /// <returns>Non-waiting trial snapshots and the position of the current trial.</returns>
    /// <exception cref="System.InvalidOperationException">The current trial is absent from non-waiting storage history.</exception>
    private static (List<FrozenTrial> Trials, int CurrentIndex) GetNonWaitingTrialsAndCurrentTrialIndex(
        Study study, int currentTrialNumber)
    {
        // We directly query the storage to get trials here instead of study.GetTrials, since some
        // pruners such as HyperbandPruner use the study transformed to filter trials.
        var states = new[] { TrialState.Complete, TrialState.Pruned, TrialState.Running, TrialState.Fail };
        var trials = study.Storage.GetAllTrials(study.StudyId, deepcopy: false, states: states);
        // `trials` is fetched by shallow copy, so removal or element replacement are safe operations.
        for (int i = 1; i <= trials.Count; i++)
        {
            // The current trial can be found at the later part for almost all cases.
            var t = trials[trials.Count - i];
            if (t.Number == currentTrialNumber)
                return (trials, trials.Count - i);
        }

        throw new InvalidOperationException("Should not reach");
    }

    /// <summary>Python <c>BruteForceSampler._populate_tree</c> — populate <paramref name="tree" /> under
    /// <paramref name="params" /> from the given trials.</summary>
    /// <param name="tree">Search tree being populated from observed objective paths.</param>
    /// <param name="trials">Trial history available to the operation, in its existing order.</param>
    /// <param name="params">External parameter values recorded along the objective path.</param>
    private static void PopulateTree(
        TreeNode tree, IReadOnlyList<FrozenTrial> trials, IReadOnlyDictionary<string, object?> @params)
    {
        // name -> choices when categorical, else null (mirrors Optuna's cat_internal_repr_cache,
        // where the else branch stores an empty dict that reads as "not categorical").
        var catCache = new Dictionary<string, IReadOnlyList<CategoricalChoice>?>();

        var nonNanParams = new List<KeyValuePair<string, object?>>();
        var nanParamNames = new List<string>();
        foreach (var kv in @params)
        {
            if (IsNan(kv.Value))
                nanParamNames.Add(kv.Key);
            else
                nonNanParams.Add(kv);
        }

        List<(string Name, ChoicesArgs Args, double Value)> GetTrialPath(FrozenTrial trial)
        {
            var trialPath = new List<(string, ChoicesArgs, double)>();
            var trialParams = trial.Params;
            foreach (var (name, dist) in trial.Distributions)
            {
                if (@params.ContainsKey(name))
                    continue;
                if (!catCache.TryGetValue(name, out var choicesOrNull))
                {
                    choicesOrNull = dist is CategoricalDistribution cd ? cd.Choices : null;
                    catCache[name] = choicesOrNull;
                }

                if (choicesOrNull is not null)
                {
                    var paramVal = trialParams[name];
                    // most likely param_val is nan when the cache lookup misses.
                    double value = CategoricalInternalRepr(choicesOrNull, paramVal) ?? dist.ToInternalRepr(paramVal);
                    trialPath.Add((name, new ChoicesArgs(0, choicesOrNull.Count - 1, 1, isInt: true), value));
                }
                else
                {
                    ChoicesArgs args;
                    if (dist is IntDistribution id)
                        args = new ChoicesArgs(id.Low, id.High, id.Step, isInt: true);
                    else
                    {
                        var fd = (FloatDistribution)dist;
                        args = new ChoicesArgs(fd.Low, fd.High, fd.Step, isInt: false);
                    }

                    PyFloat.TryConvert(trialParams[name], out double numeric);
                    trialPath.Add((name, args, numeric));
                }
            }

            return trialPath;
        }

        foreach (var trial in trials)
        {
            if (@params.Count > 0)
            {
                var trialParams = trial.Params;
                // nonnan_params_items <= trial_params.items() (Python dict_items subset).
                bool isSubset = true;
                foreach (var kv in nonNanParams)
                {
                    if (!trialParams.TryGetValue(kv.Key, out var existing) || !PyJson.ValueEquals(kv.Value, existing))
                    {
                        isSubset = false;
                        break;
                    }
                }
                if (!isSubset)
                    continue;

                // all(_is_nan(trial_params.get(p)) for p in nan_param_names).
                bool allNan = true;
                foreach (var p in nanParamNames)
                {
                    trialParams.TryGetValue(p, out var existing);  // dict.get(p) -> null when absent.
                    if (!IsNan(existing))
                    {
                        allNan = false;
                        break;
                    }
                }
                if (!allNan)
                    continue;
            }

            var leaf = tree.AddPath(GetTrialPath(trial));
            if (leaf is not null)
            {
                // The parameters are on the defined grid.
                if (trial.State.IsFinished())
                    leaf.SetLeaf();
                else
                    leaf.SetRunning();
            }
        }
    }

    /// <summary>The <c>choices_args</c> triple for the current trial's distribution.</summary>
    /// <param name="paramDistribution">Distribution defining the parameter domain and external representation.</param>
    /// <returns>A finite-choice progression descriptor for the supported distribution.</returns>
    /// <exception cref="System.InvalidOperationException">The distribution is not one of the supported finite distribution types.</exception>
    private static ChoicesArgs ArgsForDistribution(BaseDistribution paramDistribution) => paramDistribution switch
    {
        CategoricalDistribution cd => new ChoicesArgs(0, cd.Choices.Count - 1, 1, isInt: true),
        IntDistribution id => new ChoicesArgs(id.Low, id.High, id.Step, isInt: true),
        FloatDistribution fd => new ChoicesArgs(fd.Low, fd.High, fd.Step, isInt: false),
        _ => throw new InvalidOperationException("Should not reach."),
    };

    /// <summary>
    ///     Python <c>{c: i for i, c in enumerate(dist.choices)}.get(param_val)</c>: the (last) index
    ///     of a choice Python-equal to <paramref name="paramVal" />, or <c>null</c> when none match
    ///     (e.g. <paramref name="paramVal" /> is NaN, since <c>nan != nan</c>).
    /// </summary>
    /// <param name="choices">Categorical choices in distribution order.</param>
    /// <param name="paramVal">External categorical value whose last equal choice identifies the dictionary-backed branch.</param>
    /// <returns>The last matching categorical choice index, or null when the value is absent; duplicate-equal choices follow Python dictionary overwrite order.</returns>
    private static double? CategoricalInternalRepr(IReadOnlyList<CategoricalChoice> choices, object? paramVal)
    {
        object? raw = paramVal is CategoricalChoice cc ? cc.Value : paramVal;
        int? found = null;
        for (int i = 0; i < choices.Count; i++)
        {
            // Equal keys collapse in Python's dict comprehension (last write wins), so keep the last.
            if (PyJson.ValueEquals(choices[i].Value, raw))
                found = i;
        }

        return found;
    }

    /// <summary>Python <c>_is_nan</c>: <c>isinstance(v, Real) and math.isnan(float(v))</c>.</summary>
    /// <param name="v">Boxed value to inspect for floating-point NaN.</param>
    /// <returns>True when the boxed value is a floating-point NaN.</returns>
    private static bool IsNan(object? v)
    {
        object? x = v is CategoricalChoice cc ? cc.Value : v;
        return x is double d && double.IsNaN(d) || x is float f && float.IsNaN(f);
    }

    /// <summary>
    ///     Port of <c>optuna/samplers/_brute_force.py: _enumerate_candidates</c>. The ordered,
    ///     uniformly-spaced set of candidate values for one parameter (the children keys of a node).
    /// </summary>
    /// <exception cref="ValueError">When <c>step</c> is <c>null</c> (the search space would be infinite).</exception>
    /// <param name="args">Finite numerical progression descriptor.</param>
    /// <returns>All internal numerical choices of the finite progression.</returns>
    /// <exception cref="System.FormatException">A floating endpoint or step is NaN or infinity and cannot be represented as a decimal.</exception>
    /// <exception cref="System.OverflowException">A floating endpoint, step, or progression addition exceeds the decimal representation range.</exception>
    internal static List<double> EnumerateCandidates(ChoicesArgs args)
    {
        if (args.Step is null)
            throw new ValueError(
                "FloatDistribution.step must be given for BruteForceSampler" +
                " (otherwise, the search space will be infinite).");

        var result = new List<double>();
        if (args.IsInt)
        {
            // range(low, high + 1, step)
            long low = (long)args.Low, high = (long)args.High, step = (long)args.Step.Value;
            for (long v = low; v <= high; v += step)
                result.Add(v);
        }
        else
        {
            // Decimal arithmetic on the shortest reprs, exactly as Python does.
            decimal low = decimal.Parse(PythonRepr.FloatRepr(args.Low), NumberStyles.Float, CultureInfo.InvariantCulture);
            decimal high = decimal.Parse(PythonRepr.FloatRepr(args.High), NumberStyles.Float, CultureInfo.InvariantCulture);
            decimal step = decimal.Parse(PythonRepr.FloatRepr(args.Step.Value), NumberStyles.Float, CultureInfo.InvariantCulture);
            while (low <= high)
            {
                // Python's `Decimal` yields +0.0 at a zero crossing, but .NET's `decimal` keeps a
                // negative sign there (`-0.5m + 0.5m` -> -0.0 on cast). `+ 0.0` maps -0.0 to +0.0
                // and is a no-op for every other value, restoring bit-parity (and consistent
                // double keys, since .NET treats -0.0 and +0.0 as distinct dictionary keys). The conversion itself
                // must be correctly rounded like Python's float(Decimal); (double)decimal put 7% of candidates one
                // ULP off (QA-11).
                result.Add(PyNumbers.DecimalToDouble(low) + 0.0);
                low += step;
            }
        }

        return result;
    }

    /// <summary>
    ///     Python-repr of an enumerated candidate sequence for the search-space-mismatch error.
    ///     The reference prints <c>choices_old = list(self.children)</c> (a list, <c>[...]</c>) but
    ///     <c>choices_new = _enumerate_candidates(...)</c> (a tuple, <c>(...)</c>), so the two sides
    ///     bracket differently.
    /// </summary>
    /// <param name="args">Finite numerical progression descriptor.</param>
    /// <param name="asTuple">Whether diagnostics use tuple delimiters instead of list delimiters.</param>
    /// <returns>A Python-style list or tuple representation of the progression.</returns>
    private static string FormatCandidates(ChoicesArgs args, bool asTuple)
    {
        var candidates = EnumerateCandidates(args);
        var parts = (args.IsInt
            ? candidates.Select(v => ((long)v).ToString(CultureInfo.InvariantCulture))
            : candidates.Select(PythonRepr.FloatRepr)).ToList();
        if (!asTuple)
            return "[" + string.Join(", ", parts) + "]";
        // Python tuple repr: `()`, `(a,)`, `(a, b, ...)`.
        return parts.Count == 1 ? $"({parts[0]},)" : "(" + string.Join(", ", parts) + ")";
    }

    /// <summary>Python <c>!=</c> on the optional <c>choices_args</c> tuples (<c>None == None</c>).</summary>
    /// <param name="a">First finite-choice progression descriptor.</param>
    /// <param name="b">Second finite-choice progression descriptor.</param>
    /// <returns>True if both nullable progression descriptors agree.</returns>
    private static bool ArgsEqual(ChoicesArgs? a, ChoicesArgs? b)
        => a is null || b is null ? a is null && b is null : a.Value.Equals(b.Value);

    /// <summary>
    ///     NumPy's <c>np.add.reduce</c> pairwise summation for a contiguous float64 array — the exact
    ///     accumulation order of <c>ndarray.sum()</c>, so weight normalization stays bit-identical.
    /// </summary>
    /// <param name="a">First numerical array operand.</param>
    /// <param name="offset">Zero-based start of the active vector slice.</param>
    /// <param name="n">Number of entries in the active vector slice.</param>
    /// <returns>The accumulated value using the reference pairwise reduction order.</returns>
    private static double PairwiseSum(double[] a, int offset, int n)
    {
        if (n < 8)
        {
            double res = 0.0;
            for (int i = 0; i < n; i++)
                res += a[offset + i];
            return res;
        }

        if (n <= 128)
        {
            double r0 = a[offset], r1 = a[offset + 1], r2 = a[offset + 2], r3 = a[offset + 3];
            double r4 = a[offset + 4], r5 = a[offset + 5], r6 = a[offset + 6], r7 = a[offset + 7];
            int i;
            for (i = 8; i < n - n % 8; i += 8)
            {
                r0 += a[offset + i];
                r1 += a[offset + i + 1];
                r2 += a[offset + i + 2];
                r3 += a[offset + i + 3];
                r4 += a[offset + i + 4];
                r5 += a[offset + i + 5];
                r6 += a[offset + i + 6];
                r7 += a[offset + i + 7];
            }

            double res = (r0 + r1 + (r2 + r3)) + (r4 + r5 + (r6 + r7));
            for (; i < n; i++)
                res += a[offset + i];
            return res;
        }

        // divide by two but avoid non-multiples of unroll factor
        int n2 = n / 2;
        n2 -= n2 % 8;
        return PairwiseSum(a, offset, n2) + PairwiseSum(a, offset + n2, n - n2);
    }

    /// <summary>NumPy <c>searchsorted(cdf, v, side='right')</c> for an ascending array (bisect right).</summary>
    /// <param name="a">First numerical array operand.</param>
    /// <param name="v">Scalar operand used in the comparison or numerical transformation.</param>
    /// <returns>The insertion position after all entries equal to the target.</returns>
    private static int BisectRight(double[] a, double v)
    {
        int lo = 0, hi = a.Length;
        while (lo < hi)
        {
            int mid = (lo + hi) >> 1;
            if (v < a[mid])
                hi = mid;
            else
                lo = mid + 1;
        }

        return lo;
    }

    /// <summary>
    ///     A <c>(low, high, step)</c> triple driving one parameter's candidate enumeration.
    ///     <see cref="IsInt" /> records whether the triple is integer-typed (Python's
    ///     <c>isinstance(low, int) and ...</c>) — it selects <c>range</c> vs. <c>Decimal</c>
    ///     enumeration. Equality is by numeric value only, matching Python tuple <c>==</c>.
    /// </summary>
    internal readonly struct ChoicesArgs : IEquatable<ChoicesArgs>
    {
        /// <summary>Lower endpoint of the represented numerical domain.</summary>
        internal readonly double Low;
        /// <summary>Upper endpoint of the represented numerical domain.</summary>
        internal readonly double High;
        /// <summary>Quantization spacing; null or zero denotes an unquantized domain where supported.</summary>
        internal readonly double? Step;
        /// <summary>Whether finite choices are converted back to integer values.</summary>
        internal readonly bool IsInt;

        /// <summary>Stores the arithmetic progression describing a finite numerical search-space branch.</summary>
        /// <param name="low">Lower endpoint of the numerical domain.</param>
        /// <param name="high">Upper endpoint of the numerical domain.</param>
        /// <param name="step">Positive spacing between adjacent quantized numerical values.</param>
        /// <param name="isInt">Whether finite numerical values are represented externally as integers.</param>
        internal ChoicesArgs(double low, double high, double? step, bool isInt)
        {
            Low = low;
            High = high;
            Step = step;
            IsInt = isInt;
        }

        /// <summary>Tests whether finite-choice progression descriptors have matching endpoints, spacing, and numeric kind.</summary>
        /// <param name="other">Second finite-choice progression descriptor.</param>
        /// <returns>True when all finite-choice descriptor fields compare equal.</returns>
        public bool Equals(ChoicesArgs other)
        {
            bool stepEqual = Step is null && other.Step is null
                || Step is not null && other.Step is not null && Step.Value == other.Step.Value;
            return Low == other.Low && High == other.High && stepEqual;
        }

        /// <inheritdoc />
        public override bool Equals(object? obj) => obj is ChoicesArgs other && Equals(other);

        /// <inheritdoc />
        public override int GetHashCode() => HashCode.Combine(Low, High, Step);
    }

    /// <summary>Python <c>_UnexpandedTreeNode</c> / <c>_TreeNode</c> common surface.</summary>
    private abstract class TreeNodeBase
    {
        // `is_running` is exposed on both node kinds; only running `_TreeNode`s ever set it.
        /// <summary>Whether the branch is currently assigned to an unfinished evaluation.</summary>
        internal bool IsRunning;

        /// <summary>Tests whether a search-tree subtree still contains eligible unexplored choices.</summary>
        /// <param name="excludeRunning">Whether subtrees assigned to running evaluations are excluded from traversal.</param>
        /// <returns>True when at least one eligible unexplored descendant exists.</returns>
        internal abstract bool IsAnyExpandable(bool excludeRunning);

        /// <summary>Counts eligible unexplored choices below a search-tree node.</summary>
        /// <param name="excludeRunning">Whether subtrees assigned to running evaluations are excluded from traversal.</param>
        /// <returns>The number of eligible unexplored descendants.</returns>
        internal abstract int CountUnexpanded(bool excludeRunning);
    }

    /// <summary>
    ///     Port of <c>_UnexpandedTreeNode</c>. A shared sentinel that avoids the initialization
    ///     overhead of a full <see cref="TreeNode" /> until the branch is actually expanded.
    /// </summary>
    private sealed class UnexpandedTreeNode : TreeNodeBase
    {
        /// <summary>Shared sentinel representing a child whose search-tree path has not been expanded.</summary>
        internal static readonly UnexpandedTreeNode Instance = new();

        /// <inheritdoc />
        internal override bool IsAnyExpandable(bool excludeRunning) => true;

        /// <inheritdoc />
        internal override int CountUnexpanded(bool excludeRunning) => 1;
    }

    /// <summary>
    ///     Port of <c>_TreeNode</c>. A node in the brute-force search tree. It is in one of four
    ///     states: <c>Running</c> (<see cref="Children" /> null, <see cref="TreeNodeBase.IsRunning" />
    ///     true), <c>Unexpanded</c> (null, false), <c>Leaf</c> (empty), or <c>Internal</c> (non-empty).
    ///     A path from the root to a terminal node is one complete <c>params</c>.
    /// </summary>
    private sealed class TreeNode : TreeNodeBase
    {
        /// <summary>Parameter assigned to this search-tree branch, or null for a terminal path.</summary>
        internal string? ParamName;
        // null => Running or Unexpanded; empty => Leaf; non-empty => Internal.
        /// <summary>Expanded children indexed by encoded parameter value, or null before expansion.</summary>
        internal OrderedDict<double, TreeNodeBase>? Children;
        /// <summary>Finite numerical progression associated with the expanded branch.</summary>
        internal ChoicesArgs? ChoicesArgsValue;

        /// <summary>Rejects a branch whose parameter name or finite choice progression changed during traversal.</summary>
        /// <param name="paramName">Name of the parameter being suggested or looked up.</param>
        /// <param name="choicesArgs">Finite numerical progression assigned to the branch.</param>
        /// <exception cref="NumSharp.ValueError">An expanded branch is revisited with a different parameter name or choice progression.</exception>
        private void ValidateSearchSpaceConsistency(string? paramName, ChoicesArgs? choicesArgs)
        {
            if (ParamName != paramName)
                throw new ValueError($"param_name mismatch: {ParamName ?? "None"} != {paramName ?? "None"}");
            if (!ArgsEqual(choicesArgs, ChoicesArgsValue))
            {
                string choicesOld = FormatCandidates(ChoicesArgsValue!.Value, asTuple: false);
                string choicesNew = FormatCandidates(choicesArgs!.Value, asTuple: true);
                throw new ValueError($"search_space mismatch in {paramName ?? "None"}: {choicesOld} != {choicesNew}");
            }
        }

        /// <summary>Materializes a branch with one child per finite numerical choice.</summary>
        /// <param name="paramName">Name of the parameter being suggested or looked up.</param>
        /// <param name="choicesArgs">Finite numerical progression assigned to the branch.</param>
        /// <exception cref="NumSharp.ValueError">An expanded branch has incompatible metadata, or its numerical choices are not finite.</exception>
        internal void Expand(string? paramName, ChoicesArgs choicesArgs)
        {
            // If the node is unexpanded, expand it. Otherwise, check compatibility.
            if (Children is null)
            {
                ParamName = paramName;
                var choices = EnumerateCandidates(choicesArgs);
                Children = new OrderedDict<double, TreeNodeBase>(choices.Count);
                foreach (var value in choices)
                    Children[value] = UnexpandedTreeNode.Instance;
                ChoicesArgsValue = choicesArgs;
            }
            else
            {
                ValidateSearchSpaceConsistency(paramName, choicesArgs);
            }
        }

        /// <summary>Marks a search-tree branch as assigned to a running evaluation.</summary>
        internal void SetRunning() => IsRunning = true;

        /// <summary>Marks a search-tree path as a finished objective evaluation.</summary>
        internal void SetLeaf()
        {
            if (Children is not null)
                ValidateSearchSpaceConsistency(null, null);
            Children = new OrderedDict<double, TreeNodeBase>();
        }

        /// <summary>Add a path (the suggested parameters of one trial) to the tree.</summary>
        /// <param name="trialPath">Ordered parameter names, progression descriptors, and internal values of the objective path.</param>
        /// <returns>The resulting path node, or null if an already-finished prefix prevents expansion.</returns>
        internal TreeNode? AddPath(List<(string Name, ChoicesArgs Args, double Value)> trialPath)
        {
            var currentNode = this;
            foreach (var (paramName, choicesArgs, value) in trialPath)
            {
                currentNode.Expand(paramName, choicesArgs);
                var children = currentNode.Children;
                if (children is null || children.Count == 0)  // children is empty or None.
                    return null;
                if (!children.TryGetValue(value, out var nextNode))
                    return null;
                if (ReferenceEquals(nextNode, UnexpandedTreeNode.Instance))
                {
                    nextNode = new TreeNode();
                    children[value] = nextNode;
                }

                currentNode = (TreeNode)nextNode;
            }

            return currentNode;
        }

        /// <inheritdoc />
        internal override bool IsAnyExpandable(bool excludeRunning)
        {
            if (Children is null)
                return !excludeRunning || !IsRunning;
            foreach (var child in Children.Values)
            {
                if (child.IsAnyExpandable(excludeRunning))
                    return true;
            }

            return false;
        }

        /// <inheritdoc />
        internal override int CountUnexpanded(bool excludeRunning)
        {
            // Count the number of unexpanded nodes in the subtree.
            if (Children is null)
                return excludeRunning && IsRunning ? 0 : 1;
            int total = 0;
            foreach (var child in Children.Values)
                total += child.CountUnexpanded(excludeRunning);
            return total;
        }

        /// <summary>Chooses a child with probability proportional to its eligible unexplored descendants.</summary>
        /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
        /// <param name="excludeRunning">Whether subtrees assigned to running evaluations are excluded from traversal.</param>
        /// <returns>The internal parameter value identifying the selected child.</returns>
        internal double SampleChild(NumPyRandom rng, bool excludeRunning)
        {
            var keys = new List<double>(Children!.Keys);
            var children = new List<TreeNodeBase>(Children!.Values);
            int n = children.Count;

            var unexpandedCounts = new double[n];
            for (int i = 0; i < n; i++)
                unexpandedCounts[i] = children[i].CountUnexpanded(excludeRunning);

            // Blend exact uniform sampling with flat uniform sampling to prevent starvation of
            // unexplored branches.
            const double alpha = 0.5;
            double countsSum = PairwiseSum(unexpandedCounts, 0, n);
            var weightsOrig = new double[n];
            for (int i = 0; i < n; i++)
                weightsOrig[i] = unexpandedCounts[i] / countsSum;

            var weightsFlat = new double[n];
            for (int i = 0; i < n; i++)
                weightsFlat[i] = unexpandedCounts[i] > 0 ? 1.0 : 0.0;
            double flatSum = PairwiseSum(weightsFlat, 0, n);
            for (int i = 0; i < n; i++)
                weightsFlat[i] /= flatSum;

            var weights = new double[n];
            for (int i = 0; i < n; i++)
                weights[i] = (1.0 - alpha) * weightsOrig[i] + alpha * weightsFlat[i];

            bool anyNonRunningPositive = false;
            for (int i = 0; i < n; i++)
            {
                if (!children[i].IsRunning && weights[i] > 0)
                {
                    anyNonRunningPositive = true;
                    break;
                }
            }
            if (anyNonRunningPositive)
            {
                // Prioritize picking non-running and unexpanded nodes.
                for (int i = 0; i < n; i++)
                {
                    if (children[i].IsRunning)
                        weights[i] = 0.0;
                }
            }

            double weightsSum = PairwiseSum(weights, 0, n);
            for (int i = 0; i < n; i++)
                weights[i] /= weightsSum;

            // rng.choice(list(children.keys()), p=weights).item(): cdf = weights.cumsum();
            // cdf /= cdf[-1]; idx = cdf.searchsorted(random_sample(), side='right').
            var cdf = new double[n];
            double acc = 0.0;
            for (int i = 0; i < n; i++)
            {
                acc += weights[i];
                cdf[i] = acc;
            }
            double last = cdf[n - 1];
            for (int i = 0; i < n; i++)
                cdf[i] /= last;

            double u = rng.random_sample(1L).ToArray<double>()[0];
            return keys[BisectRight(cdf, u)];
        }
    }
}
