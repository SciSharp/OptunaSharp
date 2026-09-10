// JANUS multi-objective CMIX injection engine — the reusable core of `JanusMooCmixSampler`, ported from
// the candidate-generation half of the authors' reference `janus_optuna/janus_moo_cmix.py`
// (`_sample_cmix`, `_build_transforms`, `_to_norm`, `_from_norm_vector`, `_scalar_value`,
// `_choose_center`, `_inverse_metric_cov`). Kept host-agnostic (it only needs the study, the relative
// search space, and its own RNG) so a NSGA-II, SMS-EMOA, or AGE-MOEA-2 host can all drive it.
//
// Reference: JANUS — arXiv:2608.22862 (2026); https://github.com/hongyuanyu/JANUS (Apache-2.0).
//
// Documented divergences (principled, none change the geometry):
//   • RNG = NumSharp MT19937 `RandomState`, not the reference's numpy `default_rng` (PCG64) — the
//     project's "never System.Random" discipline. So the CMIX fire decision and the Gaussian draw are
//     behaviorally faithful but not byte-identical to the reference; a PCG64/`multivariate_normal`
//     byte-match is a deferred stretch goal (Plan 05). The pure-numpy metric fits ARE byte-comparable
//     (see `JanusMooMetric` / `JanusMooMetricParityTests`).
//   • `multivariate_normal(center, cov)` = `center + L·z`, `L = chol(cov)` (the covariance is SPD after
//     the reference's `+1e-8·I` floor), not numpy's default SVD factorization.

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers.Janus;

/// <summary>
///     The JANUS MOO CMIX candidate generator: past a warm-up it occasionally replaces a host child with
///     an inverse-curvature Gaussian sample estimated from the recent window (component residuals, or a
///     diagonal-quadratic scalar fallback), centered on a crowding-ranked point. Held by a MOO host
///     sampler (<see cref="JanusMooCmixSampler" />) and consulted from its <c>SampleRelative</c>.
/// </summary>
internal sealed class JanusMooCmix
{
    private const double Eps = 1e-12;

    private readonly double _cmixProb;
    private readonly int _cmixWindow;
    private readonly double _cmixRidge;
    private readonly double _cmixLambda;
    private readonly double _cmixSigma;
    private readonly int _cmixStartTrial;
    private readonly string _componentAttr;
    private readonly string _scalarFallback;
    private readonly NumPyRandom _rng;

    private string _lastMode = "none";

    internal JanusMooCmix(
        int populationSize, int? seed, double cmixProb, string cmixWindow, double cmixRidge,
        double cmixLambda, double cmixSigma, int cmixStartGeneration, int? cmixStartTrial,
        string componentAttr, string scalarFallback)
    {
        _cmixProb = cmixProb;
        _cmixWindow = string.Equals(cmixWindow, "auto", StringComparison.OrdinalIgnoreCase)
            ? 3 * populationSize
            : (int.TryParse(cmixWindow, out int w) ? w : 3 * populationSize);
        _cmixRidge = cmixRidge;
        _cmixLambda = cmixLambda;
        _cmixSigma = cmixSigma;
        _cmixStartTrial = cmixStartTrial is null
            ? Math.Max(_cmixWindow, Math.Max(0, cmixStartGeneration) * populationSize)
            : Math.Max(0, cmixStartTrial.Value);
        _componentAttr = componentAttr;
        _scalarFallback = (scalarFallback ?? "diag_quadratic").ToLowerInvariant();

        _rng = np.random.RandomState();
        // Seed the CMIX stream from the given seed (a separate MT19937 from the host's, so the host's
        // bit-exact trajectory is untouched); an absent seed draws one from MT19937 (never System.Random).
        _rng.seed((uint)(seed ?? (int)new LazyRandomState().Rng.randint(1, int.MaxValue)));
    }

    internal int StartTrial => _cmixStartTrial;
    internal string LastMode => _lastMode;

    /// <summary>The reference's per-trial fire decision: <c>cmix_rng.random() &lt; cmix_prob</c>.</summary>
    internal bool DecideFire() => _rng.uniform(0.0, 1.0).ToArray<double>()[0] < _cmixProb;

    /// <summary>
    ///     The host-agnostic CMIX gate (the reference's <c>sample_relative</c> tail): past the warm-up, and
    ///     only in a &gt;1-D relative space, replace the host child <paramref name="hostParams" /> with a CMIX
    ///     sample with probability <c>cmix_prob</c> (a failed draw keeps the host child). Each MOO host
    ///     sampler calls this with the child its own <c>base.SampleRelative</c> produced.
    /// </summary>
    internal OrderedDict<string, object?> Apply(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace,
        OrderedDict<string, object?> hostParams)
    {
        if (trial.Number < _cmixStartTrial) return hostParams;
        if (searchSpace.Count > 1 && DecideFire())
        {
            var cmixParams = TrySample(study, searchSpace);
            if (cmixParams is not null && cmixParams.Count > 0) return cmixParams;
        }
        return hostParams;
    }

    /// <summary>
    ///     Port of <c>_sample_cmix</c>: harvest the recent window, fit the CMIX metric (component or the
    ///     diagonal-quadratic scalar fallback), pick a center, build the trace-normalized inverse-curvature
    ///     covariance, and draw a clipped normalized sample decoded back to params. <c>null</c> whenever any
    ///     stage fails (the host child is used instead).
    /// </summary>
    internal OrderedDict<string, object?>? TrySample(Study study, OrderedDict<string, BaseDistribution> searchSpace)
    {
        var paramNames = searchSpace.Keys.OrderBy(k => k, StringComparer.Ordinal).ToList();
        var transforms = BuildTransforms(searchSpace, paramNames);
        if (transforms is null) return null;

        var completed = study.GetTrialsInternal(deepcopy: false, states: null, useCache: true)
            .Where(t => t.State == TrialState.Complete && t.Values is not null)
            .ToList();

        var rows = new List<double[]>();
        var usedTrials = new List<FrozenTrial>();
        var components = new List<double[]>();
        bool hasComponents = true;

        int take = Math.Max(4, _cmixWindow);
        int start = Math.Max(0, completed.Count - take);
        for (int ti = start; ti < completed.Count; ti++)
        {
            var t = completed[ti];
            if (!paramNames.All(n => t.ParamsDict.ContainsKey(n))) continue;
            double[] row;
            try { row = paramNames.Select(n => ToNorm(t.ParamsDict[n], transforms[n])).ToArray(); }
            catch (Exception) { continue; }
            rows.Add(row);
            usedTrials.Add(t);
            if (!TryReadComponents(t, out double[] comp)) hasComponents = false;
            else if (hasComponents) components.Add(comp);
        }

        if (rows.Count < 4) return null;
        double[,] x = ToMatrix(rows);

        double[,]? metric = null;
        if (hasComponents && components.Count == rows.Count)
        {
            metric = JanusMooMetric.FitComponentMetric(x, ToMatrix(components), _cmixRidge);
            _lastMode = "component_losses";
        }
        if (metric is null && _scalarFallback is "diag_quadratic" or "diagq" or "quad")
        {
            double[] values = usedTrials.Select(t => ScalarValue(t)).ToArray();
            metric = JanusMooMetric.FitDiagQuadraticMetric(x, values, _cmixRidge);
            _lastMode = "diag_quadratic";
        }
        if (metric is null) { _lastMode = "none"; return null; }

        int dim = x.GetLength(1);
        double[]? center = ChooseCenter(usedTrials, paramNames, transforms) ?? ColumnMean(x);
        double[,]? cov = InverseMetricCov(metric, dim);
        if (cov is null) return null;

        for (int attempt = 0; attempt < 8; attempt++)
        {
            double[]? sample = MultivariateNormal(center, cov);
            if (sample is null) return null;
            for (int i = 0; i < dim; i++) sample[i] = Math.Min(Math.Max(sample[i], 0.0), 1.0);
            var pars = FromNormVector(sample, searchSpace, paramNames, transforms);
            if (pars is not null) return pars;
        }
        return null;
    }

    // ---- transforms & (de)normalization ----------------------------------------------------------

    private sealed class Transform
    {
        public double Low;
        public double High;
        public double? Step;
        public bool IsInt;
        public required BaseDistribution Dist;
    }

    private static Dictionary<string, Transform>? BuildTransforms(
        OrderedDict<string, BaseDistribution> searchSpace, List<string> paramNames)
    {
        var transforms = new Dictionary<string, Transform>();
        foreach (string name in paramNames)
        {
            var dist = searchSpace[name];
            if (dist is FloatDistribution fd)
            {
                double low = fd.Low, high = fd.High;
                if (!IsFinite(low) || !IsFinite(high) || high <= low) return null;
                transforms[name] = new Transform { Low = low, High = high, Step = fd.Step, IsInt = false, Dist = fd };
            }
            else if (dist is IntDistribution id)
            {
                double low = id.Low, high = id.High;
                if (!IsFinite(low) || !IsFinite(high) || high <= low) return null;
                transforms[name] = new Transform { Low = low, High = high, Step = id.Step, IsInt = true, Dist = id };
            }
            else
            {
                return null;
            }
        }
        return transforms;
    }

    private static double ToNorm(object? value, Transform t) => (Convert.ToDouble(value) - t.Low) / (t.High - t.Low);

    private static OrderedDict<string, object?>? FromNormVector(
        double[] sample, OrderedDict<string, BaseDistribution> searchSpace, List<string> paramNames,
        Dictionary<string, Transform> transforms)
    {
        var pars = new OrderedDict<string, object?>();
        for (int idx = 0; idx < paramNames.Count; idx++)
        {
            string name = paramNames[idx];
            var t = transforms[name];
            var dist = searchSpace[name];
            double value = t.Low + sample[idx] * (t.High - t.Low);
            if (t.Step is double step)
                value = t.Low + Math.Round((value - t.Low) / step, MidpointRounding.ToEven) * step;
            object boxed;
            if (t.IsInt)
            {
                long iv = (long)Math.Round(value, MidpointRounding.ToEven);
                iv = Math.Min(Math.Max(iv, (long)t.Low), (long)t.High);
                boxed = iv;
                value = iv;
            }
            else
            {
                value = Math.Min(Math.Max(value, t.Low), t.High);
                boxed = value;
            }
            if (!dist.Contains(dist.ToInternalRepr(boxed))) return null;
            pars[name] = boxed;
        }
        return pars;
    }

    // ---- center, covariance, sampling ------------------------------------------------------------

    private double ScalarValue(FrozenTrial trial)
    {
        var vals = trial.Values!;
        double s = 0;
        foreach (double v in vals) s += IsFinite(v) ? v : 1.0;
        return s / vals.Count;
    }

    private double[]? ChooseCenter(List<FrozenTrial> trials, List<string> paramNames, Dictionary<string, Transform> transforms)
    {
        var complete = trials.Where(t => paramNames.All(n => t.ParamsDict.ContainsKey(n))).ToList();
        if (complete.Count == 0) return null;

        var valueRows = complete.Where(t => t.Values is not null).Select(t => t.Values!.ToArray()).ToList();
        List<FrozenTrial> top;
        if (valueRows.Count == complete.Count && valueRows.Count > 0)
        {
            double[][] values = valueRows.ToArray();
            int[] frontIdx = JanusMooMetric.NondominatedIndices(values);
            var front = frontIdx.Select(i => complete[i]).ToList();
            if (front.Count > 1)
            {
                double[][] frontValues = frontIdx.Select(i => values[i]).ToArray();
                double[] cd = JanusMooMetric.CrowdingLike(frontValues);
                int[] order = DescendingStableArgSort(cd);
                int keep = Math.Max(1, Math.Min(5, front.Count));
                top = order.Take(keep).Select(i => front[i]).ToList();
            }
            else
            {
                top = front;
            }
        }
        else
        {
            top = new List<FrozenTrial>();
        }

        if (top.Count == 0)
        {
            var ranked = complete.OrderBy(t => ScalarValue(t)).ToList();
            int keep = Math.Max(1, Math.Min(5, ranked.Count));
            top = ranked.Take(keep).ToList();
        }

        int d = paramNames.Count;
        var center = new double[d];
        foreach (var t in top)
            for (int j = 0; j < d; j++) center[j] += ToNorm(t.ParamsDict[paramNames[j]], transforms[paramNames[j]]);
        for (int j = 0; j < d; j++) center[j] /= top.Count;
        return center;
    }

    private double[,]? InverseMetricCov(double[,] metric, int dim)
    {
        var a = new double[dim, dim];
        for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++) a[i, j] = metric[i, j];
        for (int i = 0; i < dim; i++) a[i, i] += _cmixLambda;
        double[,]? cov = Inverse(a);
        if (cov is null) return null;
        var sym = new double[dim, dim];
        for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++) sym[i, j] = 0.5 * (cov[i, j] + cov[j, i]);
        double trace = 0; for (int i = 0; i < dim; i++) trace += sym[i, i];
        if (!IsFinite(trace) || trace <= Eps) return null;
        double scale = Math.Pow(Math.Max(_cmixSigma, Eps), 2) * dim / trace;
        for (int i = 0; i < dim; i++) for (int j = 0; j < dim; j++) sym[i, j] *= scale;
        for (int i = 0; i < dim; i++) sym[i, i] += 1e-8;
        return sym;
    }

    private double[]? MultivariateNormal(double[] mean, double[,] cov)
    {
        int d = mean.Length;
        double[,] l;
        try { l = LinAlg.Cholesky(cov); }
        catch (Exception) { return null; }
        double[] z = _rng.randn(d).ToArray<double>();
        var sample = new double[d];
        for (int i = 0; i < d; i++)
        {
            double s = mean[i];
            for (int j = 0; j <= i; j++) s += l[i, j] * z[j];   // lower-triangular L
            sample[i] = s;
        }
        return sample;
    }

    // ---- small helpers ---------------------------------------------------------------------------

    private static bool TryReadComponents(FrozenTrial trial, out double[] components)
    {
        components = Array.Empty<double>();
        if (!trial.UserAttrs.TryGetValue(ComponentLosses.Key, out object? raw) || raw is null)
        {
            // Fall back to the shared reader (handles the legacy key / list-or-array forms).
            return ComponentLosses.TryRead(trial.UserAttrs, out components);
        }
        try
        {
            switch (raw)
            {
                case double[] da: components = (double[])da.Clone(); return components.Length > 0;
                case IEnumerable<double> ed: components = ed.ToArray(); return components.Length > 0;
                case System.Collections.IEnumerable en:
                    var list = new List<double>();
                    foreach (object? o in en) list.Add(Convert.ToDouble(o));
                    components = list.ToArray();
                    return components.Length > 0;
                default: return false;
            }
        }
        catch (Exception) { return false; }
    }

    private static double[,]? Inverse(double[,] a)
    {
        int n = a.GetLength(0);
        var inv = new double[n, n];
        try
        {
            for (int c = 0; c < n; c++)
            {
                var e = new double[n];
                e[c] = 1.0;
                double[] col = LinAlg.Solve(a, e);
                for (int i = 0; i < n; i++) inv[i, c] = col[i];
            }
        }
        catch (Exception) { return null; }
        return inv;
    }

    private static double[,] ToMatrix(List<double[]> rows)
    {
        int n = rows.Count, d = rows[0].Length;
        var m = new double[n, d];
        for (int i = 0; i < n; i++) for (int j = 0; j < d; j++) m[i, j] = rows[i][j];
        return m;
    }

    private static double[] ColumnMean(double[,] x)
    {
        int n = x.GetLength(0), d = x.GetLength(1);
        var mean = new double[d];
        for (int j = 0; j < d; j++) { double s = 0; for (int i = 0; i < n; i++) s += x[i, j]; mean[j] = s / n; }
        return mean;
    }

    /// <summary>Indices sorted by DESCENDING crowding (numpy <c>argsort(-cd, kind="mergesort")</c>): ascending
    ///     on <c>-cd</c>, ties by original index. <c>+∞</c> crowding (boundaries) sorts first.</summary>
    private static int[] DescendingStableArgSort(double[] cd)
    {
        int n = cd.Length;
        var idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Array.Sort(idx, (a, b) =>
        {
            int cmp = (-cd[a]).CompareTo(-cd[b]);
            return cmp != 0 ? cmp : a.CompareTo(b);
        });
        return idx;
    }

    private static bool IsFinite(double v) => !double.IsNaN(v) && !double.IsInfinity(v);
}
