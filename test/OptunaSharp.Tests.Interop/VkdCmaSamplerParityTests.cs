using System;
using System.Globalization;
using System.Text;
using AwesomeAssertions;
using OptunaSharp.Samplers.CmaEs;
using Python.Runtime;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Value-parity of the native <see cref="VkdCma" /> covariance model against its <b>live reference</b> —
///     Akimoto's own <c>GaussVkDSampler</c> in pycma (<c>cma.restricted_gaussian_sampler</c>), driven
///     in-process via pythonnet.
///
///     <para><b>What is compared, and why at the update/transform level rather than end-to-end.</b>
///     <see cref="VkdCma" /> ports <c>GaussVkDSampler</c>'s restricted-covariance model (<c>C = σ²·D(I +
///     Σⱼ Sⱼ VⱼVⱼᵀ)D</c>) and wraps it in a self-contained ES. A full multi-generation trajectory
///     comparison is <b>not</b> a sound bit-parity gate: the reference is a bare model driven by an
///     external ES, so the two runs are coupled only through discrete decisions — the fitness <b>sort
///     order</b>, the TPA injected-pair <b>rank recovery</b>, and the k-adaptation <b>grow/shrink
///     thresholds</b> — each of which can flip on a sub-ULP difference and then diverge to O(1). That is
///     inherent chaotic sensitivity of discrete choices, not a port defect. So this gate isolates the
///     <b>continuous</b> ported math and feeds it <b>identical inputs</b> on both sides:
///     <list type="bullet">
///         <item>the <c>transform</c> / <c>transform_inverse</c> whitening, on a crafted (D, V, S, σ) state
///             and a fixed z;</item>
///         <item>a single covariance <b>update</b> — the natural-gradient step (U-matrix build, the
///             <c>eigh</c> of <c>UᵀU</c> when N&gt;rankU else <c>UUᵀ</c>, the S/V extraction, the diagonal
///             update, and the determinant normalization) plus the evolution-path cumulation — from a
///             crafted state with <c>flg_injection = False</c> (so no TPA) and identical sample steps and
///             weights. The rank-mu update reads only the selected steps, so tail ordering is irrelevant
///             here; there is no RNG, no sort, no k-boundary.</item>
///     </list>
///     The eigendecomposition sign/row-order is gauge, so the covariance is compared through the
///     invariant <c>A = Σⱼ Sⱼ VⱼVⱼᵀ = Vᵀ diag(S) V</c> (plus D and pc directly). TPA (three scalar lines)
///     and k-adaptation timing are covered by the native convergence / rank-growth tests instead.</para>
///
///     <para><b>Tolerance.</b> With identical inputs the only difference is the sequential-vs-BLAS/pairwise
///     summation ULP (numpy's <c>@</c>/<c>eigh</c> vs the port's loops through the same OpenBLAS), the same
///     "value-parity ~1 ULP" tier as <see cref="JanusMetricParityTests" />; observed ≲1e-12, gated at
///     <see cref="Tol" />. Host-pinned to x64 (OpenBLAS DYNAMIC_ARCH + summation order). Self-skips
///     (Inconclusive) when Python or a pycma with <c>GaussVkDSampler</c> is unavailable.</para>
/// </summary>
[TestClass]
public class VkdCmaSamplerParityTests : OptunaInteropTestBase
{
    private const double Tol = 1e-9;

    private void RequirePycmaVkdOrInconclusive()
    {
        PythonSession.EnsureBitExactHostOrInconclusive();
        try
        {
            PyExec(
                "import numpy as np, math\n" +
                "from cma import restricted_gaussian_sampler as _rgs\n" +
                "assert hasattr(_rgs, 'GaussVkDSampler')\n" +
                "def _std_weights(lam):\n" +
                "    mu = lam // 2\n" +
                "    wp = np.array([math.log(mu + 1) - math.log(i + 1) for i in range(mu)])\n" +
                "    w = wp / wp.sum()\n" +
                "    fw = np.zeros(lam); fw[:mu] = w\n" +
                "    return fw\n" +
                // one covariance update from a crafted state, flg_injection=False (no TPA), kadapt=False
                "def _vkd_update(N, lam, k, ka, D, Vrows, Sact, pc, sigma, vecrows):\n" +
                "    sm = _rgs.GaussVkDSampler(N, kadapt=False, k_init=k)\n" +
                "    sm.D = np.array(D, dtype=float)\n" +
                "    sm.sigma = float(sigma)\n" +
                "    sm.pc = np.array(pc, dtype=float)\n" +
                "    sm.V = np.array(Vrows, dtype=float).reshape(k, N)\n" +
                "    S = np.zeros(N)\n" +
                "    if ka > 0: S[:ka] = np.array(Sact, dtype=float)\n" +
                "    sm.S = S\n" +
                "    sm.k = k; sm.k_active = ka\n" +
                "    sm.flg_injection = False\n" +
                "    fw = _std_weights(lam)\n" +
                "    vectors = np.vstack([np.zeros(N), np.array(vecrows, dtype=float).reshape(lam, N)])\n" +
                "    weights = np.concatenate([[0.0], fw])\n" +
                "    sm.update(vectors, weights)\n" +
                "    kk = int(sm.k_active)\n" +
                "    A = (sm.V[:kk].T @ (sm.S[:kk][:, None] * sm.V[:kk])) if kk > 0 else np.zeros((N, N))\n" +
                "    return {'D': np.asarray(sm.D, dtype=float), 'pc': np.asarray(sm.pc, dtype=float),\n" +
                "            'sigma': float(sm.sigma), 'A': np.asarray(A, dtype=float), 'ka': kk, 'k': int(sm.k)}\n" +
                // transform / transform_inverse of a fixed z under a crafted state
                "def _vkd_transform(N, ka, D, Vrows, Sact, sigma, z):\n" +
                "    sm = _rgs.GaussVkDSampler(N, kadapt=False, k_init=max(ka, 1))\n" +
                "    sm.D = np.array(D, dtype=float); sm.sigma = float(sigma)\n" +
                "    if ka > 0: sm.V = np.array(Vrows, dtype=float).reshape(ka, N)\n" +
                "    S = np.zeros(N)\n" +
                "    if ka > 0: S[:ka] = np.array(Sact, dtype=float)\n" +
                "    sm.S = S; sm.k = max(ka, 1); sm.k_active = ka\n" +
                "    zz = np.array(z, dtype=float)\n" +
                "    return {'t': np.asarray(sm.transform(zz), dtype=float),\n" +
                "            'ti': np.asarray(sm.transform_inverse(zz), dtype=float)}\n");
        }
        catch (PythonException ex)
        {
            Assert.Inconclusive($"pycma with GaussVkDSampler is not importable in the interop interpreter. {ex.Message}");
        }
    }

    // ---- covariance update parity: the natural-gradient step on identical crafted inputs ----------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void CovarianceUpdate_MatchesPycma_AcrossShapesAndRanks()
    {
        RequirePycmaVkdOrInconclusive();

        var rng = new Random(20260911);
        int compared = 0;
        // (N, lam, ka): mix N>rankU (the O(Nk²) eigh of UᵀU) and N≤rankU (the N×N eigh of UUᵀ).
        foreach (var (N, lam, ka) in new[]
                 {
                     (12, 8, 1), (12, 8, 3), (12, 10, 2), (16, 10, 4),   // N > rankU = ka+mu+1
                     (6, 8, 1), (7, 8, 2), (8, 8, 3),                    // N ≤ rankU
                 })
        {
            for (int trial = 0; trial < 3; trial++)
            {
                CompareOneUpdate(N, lam, ka, rng);
                compared++;
            }
        }
        compared.Should().Be(7 * 3);
    }

    // ---- transform / transform_inverse parity (the sampling whitening) ---------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Transform_MatchesPycma_AcrossStates()
    {
        RequirePycmaVkdOrInconclusive();

        var rng = new Random(31337);
        int compared = 0;
        foreach (var (N, ka) in new[] { (10, 0), (10, 1), (10, 3), (6, 2), (16, 5) })
        {
            for (int trial = 0; trial < 3; trial++)
            {
                CompareTransform(N, ka, rng);
                compared++;
            }
        }
        compared.Should().Be(5 * 3);
    }

    // ---- one update: craft identical state on both sides, apply, compare A / D / pc / sigma ------

    private void CompareOneUpdate(int N, int lam, int ka, Random rng)
    {
        double sigma = 1.0 + rng.NextDouble();
        double[] D = RandVec(rng, N, 0.4, 1.2);            // positive diagonal
        double[][] V = RandMat(rng, ka, N, -1.0, 1.0);     // any rows (need not be orthonormal for parity)
        double[] S = RandVec(rng, ka, 0.1, 3.0);           // positive eigenvalue corrections
        double[] pc = RandVec(rng, N, -0.6, 0.6);
        double[][] steps = RandMat(rng, lam, N, -1.0, 1.0);

        int k = ka;   // fixed rank (kAdapt=false); k == ka so V has exactly ka rows
        var opt = VkdCma.FromState(CraftState(N, lam, k, ka, sigma, D, V, S, pc));

        // one update via Tell: candidates = mean(0) + steps, value = rank so the sort keeps the given order
        var mean = opt.Mean;   // zeros
        var sols = new System.Collections.Generic.List<(double[] X, double Value)>();
        for (int i = 0; i < lam; i++)
        {
            var x = new double[N];
            for (int j = 0; j < N; j++) x[j] = mean[j] + steps[i][j];
            sols.Add((x, i));
        }
        opt.Tell(sols);

        var st = opt.ToState();
        double[][] aCs = AMatrix(st);   // A = Σⱼ Sⱼ VⱼVⱼᵀ over the active rank

        string ctx = $"N={N} lam={lam} ka={ka}";
        PyExec(
            $"r = _vkd_update({N}, {lam}, {k}, {ka}, {Lit(D)}, {Lit(V)}, {Lit(S)}, {Lit(pc)}, {R(sigma)}, {Lit(steps)})\n" +
            $"D_cs = np.array({Lit(st.D)}); pc_cs = np.array({Lit(st.Pc)}); A_cs = np.array({Lit(aCs)})\n" +
            $"sigma_cs = {R(st.Sigma)}\n" +
            "kamatch = (r['ka'] == " + st.KActive + ")\n" +
            "dD = float(np.max(np.abs(r['D'] - D_cs))); sD = float(np.max(np.abs(r['D'])))\n" +
            "dpc = float(np.max(np.abs(r['pc'] - pc_cs))); spc = float(np.max(np.abs(r['pc'])))\n" +
            "dA = float(np.max(np.abs(r['A'] - A_cs))); sA = float(np.max(np.abs(r['A'])))\n" +
            "dsig = float(abs(r['sigma'] - sigma_cs)); ssig = float(abs(r['sigma']))\n");

        PyBool("kamatch").Should().BeTrue($"active rank must agree ({ctx})");
        PyFloat("dD").Should().BeLessThan(Tol * (1 + PyFloat("sD")), $"D parity ({ctx})");
        PyFloat("dpc").Should().BeLessThan(Tol * (1 + PyFloat("spc")), $"pc parity ({ctx})");
        PyFloat("dA").Should().BeLessThan(Tol * (1 + PyFloat("sA")), $"covariance (VᵀSV) parity ({ctx})");
        PyFloat("dsig").Should().BeLessThan(Tol * (1 + PyFloat("ssig")), $"sigma unchanged parity ({ctx})");
    }

    private void CompareTransform(int N, int ka, Random rng)
    {
        double sigma = 0.5 + rng.NextDouble();
        double[] D = RandVec(rng, N, 0.4, 1.2);
        double[][] V = ka > 0 ? RandMat(rng, ka, N, -1.0, 1.0) : Array.Empty<double[]>();
        double[] S = ka > 0 ? RandVec(rng, ka, 0.1, 3.0) : Array.Empty<double>();
        double[] z = RandVec(rng, N, -1.5, 1.5);

        int k = Math.Max(ka, 1);
        var opt = VkdCma.FromState(CraftState(N, 8, k, ka, sigma, D, PadRows(V, k, N), S, new double[N]));
        double[] tCs = opt.TransformForParity(z);
        double[] tiCs = opt.TransformInverseForParity(z);

        string ctx = $"N={N} ka={ka}";
        PyExec(
            $"r = _vkd_transform({N}, {ka}, {Lit(D)}, {Lit(V)}, {Lit(S)}, {R(sigma)}, {Lit(z)})\n" +
            $"t_cs = np.array({Lit(tCs)}); ti_cs = np.array({Lit(tiCs)})\n" +
            "dt = float(np.max(np.abs(r['t'] - t_cs))); st = float(np.max(np.abs(r['t'])))\n" +
            "dti = float(np.max(np.abs(r['ti'] - ti_cs))); sti = float(np.max(np.abs(r['ti'])))\n");

        PyFloat("dt").Should().BeLessThan(Tol * (1 + PyFloat("st")), $"transform parity ({ctx})");
        PyFloat("dti").Should().BeLessThan(Tol * (1 + PyFloat("sti")), $"transform_inverse parity ({ctx})");
    }

    // ---- craft a VkdCma state with a controlled covariance model (flg_injection off, kAdapt off) --

    private static VkdCmaState CraftState(int N, int lam, int k, int ka, double sigma,
        double[] d, double[][] v, double[] sActive, double[] pc)
    {
        var s = new double[N];
        for (int j = 0; j < ka; j++) s[j] = sActive[j];
        return new VkdCmaState
        {
            NDim = N, Sigma0 = sigma, Popsize = lam, KInit = k, KAdapt = false, Kmin = 0, Kmax = k,
            Mean = new double[N], Sigma = sigma, D = (double[])d.Clone(),
            V = v, S = s, Pc = (double[])pc.Clone(), Dx = new double[N],
            K = k, KActive = ka, FlgInjection = false, Ps = 0, G = 0,
            LastLogSigma = Math.Log(sigma), LastLogD = new double[N], LastLogCondCorr = new double[N],
            ItrAfterKInc = 0, EmaLogSigmaM = new double[1], EmaLogDM = new double[N], EmaLogSM = new double[N],
        };
    }

    private static double[][] AMatrix(VkdCmaState st)
    {
        int N = st.NDim, ka = st.KActive;
        var a = new double[N][];
        for (int i = 0; i < N; i++) a[i] = new double[N];
        for (int j = 0; j < ka; j++)
        {
            double sj = st.S[j];
            double[] vj = st.V[j];
            for (int i = 0; i < N; i++)
                for (int l = 0; l < N; l++)
                    a[i][l] += sj * vj[i] * vj[l];
        }
        return a;
    }

    private static double[][] PadRows(double[][] v, int k, int n)
    {
        var r = new double[k][];
        for (int j = 0; j < k; j++) r[j] = j < v.Length ? v[j] : new double[n];
        return r;
    }

    // ---- deterministic random inputs -------------------------------------------------------------

    private static double[] RandVec(Random rng, int n, double lo, double hi)
    {
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = lo + (hi - lo) * rng.NextDouble();
        return a;
    }

    private static double[][] RandMat(Random rng, int rows, int cols, double lo, double hi)
    {
        var m = new double[rows][];
        for (int i = 0; i < rows; i++) m[i] = RandVec(rng, cols, lo, hi);
        return m;
    }

    // ---- python literals (round-trip exact) ------------------------------------------------------

    private static string R(double v) => v.ToString("R", CultureInfo.InvariantCulture);

    private static string Lit(double[] a)
    {
        var sb = new StringBuilder("[");
        for (int i = 0; i < a.Length; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(a[i].ToString("R", CultureInfo.InvariantCulture));
        }
        return sb.Append(']').ToString();
    }

    private static string Lit(double[][] a)
    {
        var sb = new StringBuilder("[");
        for (int i = 0; i < a.Length; i++)
        {
            if (i > 0) sb.Append(", ");
            sb.Append(Lit(a[i]));
        }
        return sb.Append(']').ToString();
    }
}
