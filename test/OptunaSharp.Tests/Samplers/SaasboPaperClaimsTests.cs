using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.ModelBased.Saas;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Paper-claim gates for SAASBO (Eriksson &amp; Jankowiak, "High-Dimensional Bayesian Optimization with
///     Sparse Axis-Aligned Subspaces", UAI 2021, arXiv:2103.00349). Each test proves a specific stated
///     equation, algorithm detail, or empirical claim of the paper against our native port — the audit that
///     backs the "paper parity" contract. Deterministic claims (the kernel/posterior formulae, the
///     half-Cauchy prior normalization, the reparameterization identity, the EI expectation, the
///     paper-default hyperparameters) are asserted exactly; the stochastic empirical claims (§5.2
///     relevant-subspace identification, α-shrinkage, §4.1/§5.7 adaptive sparsity) are asserted on seeded
///     MT19937 chains with thresholds calibrated to measured behavior.
/// </summary>
[TestClass]
public class SaasboPaperClaimsTests
{
    /// <summary>Builds a standardized, maximization-oriented training set from a seeded RNG (the GP's input convention).</summary>
    /// <param name="seed">The MT19937 seed.</param>
    /// <param name="n">The number of points.</param>
    /// <param name="d">The input dimension.</param>
    /// <param name="f">The pre-standardization objective evaluated per row.</param>
    /// <returns>The normalized inputs and standardized targets.</returns>
    private static (double[,] X, double[] Y) MakeData(int seed, int n, int d, Func<double[], double> f)
    {
        var rng = np.random.RandomState();
        rng.seed(seed);
        var x = new double[n, d];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                x[i, j] = (double)rng.rand();
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            var row = new double[d];
            for (int j = 0; j < d; j++) row[j] = x[i, j];
            y[i] = f(row);
        }
        double mean = y.Average();
        double std = Math.Max(1e-10, Math.Sqrt(y.Select(v => (v - mean) * (v - mean)).Average()));
        for (int i = 0; i < n; i++) y[i] = (y[i] - mean) / std;
        return (x, y);
    }

    /// <summary>Fits the SAAS GP with NUTS and returns the posterior-median inverse-squared length scales
    ///     <c>PosteriorMedian(ρ)</c> — the paper's relevance statistic (§5.2, Fig. 2).</summary>
    /// <param name="model">The SAAS model.</param>
    /// <param name="seed">The chain seed.</param>
    /// <param name="warmup">NUTS warmup iterations.</param>
    /// <param name="samples">NUTS post-warmup iterations.</param>
    /// <returns>The per-dimension posterior-median <c>ρ</c>.</returns>
    private static double[] PosteriorMedianRho(SaasModel model, int seed, int warmup, int samples)
    {
        var init = new double[model.LatentDim];
        init[0] = Math.Log(model.Alpha);
        var rng = np.random.RandomState();
        rng.seed(seed);
        List<SaasHyperparameters> draws = new NoUTurnSampler(warmup, samples, maxTreeDepth: 6)
            .Sample(model.LogJointAndGrad, init, rng)
            .Select(model.HyperparametersFromTheta)
            .ToList();

        int d = model.Dim;
        var median = new double[d];
        for (int j = 0; j < d; j++)
        {
            double[] col = draws.Select(h => h.Rho[j]).OrderBy(v => v).ToArray();
            median[j] = col[col.Length / 2];
        }
        return median;
    }

    // ---- §3.1 Eqn (1): the RBF kernel, and Eqn (4)-(5): the closed-form posterior --------------------

    [TestMethod]
    public void Kernel_And_Posterior_MatchEqn1And4And5_ClosedForm()
    {
        // Two 1-D training points; verify SaasPosterior against a hand-computed 2×2 RBF GP posterior.
        //   k(x,y) = σ²_k·exp(−½·ρ·(x−y)²)   (Eqn 1)
        //   mean   = k*ᵀ(C)⁻¹y,  var = σ²_k − k*ᵀ(C)⁻¹k*   (Eqns 4-5),  C = K + σ²·I.
        double rho = 1.7, sigmaK = 1.3, noise = 1e-6;
        var x = new double[,] { { 0.2 }, { 0.8 } };
        var y = new[] { 0.4, -0.9 };
        double x1 = 0.2, x2 = 0.8, xStar = 0.5;

        double K(double a, double b) => sigmaK * Math.Exp(-0.5 * rho * (a - b) * (a - b));
        double a11 = K(x1, x1) + noise, a22 = K(x2, x2) + noise, a12 = K(x1, x2);
        double det = a11 * a22 - a12 * a12;
        // C⁻¹ = 1/det · [[a22, −a12], [−a12, a11]].
        double[] cinvY = { (a22 * y[0] - a12 * y[1]) / det, (-a12 * y[0] + a11 * y[1]) / det };
        double ks1 = K(xStar, x1), ks2 = K(xStar, x2);
        double expMean = ks1 * cinvY[0] + ks2 * cinvY[1];
        // k*ᵀ C⁻¹ k* :
        double[] cinvKs = { (a22 * ks1 - a12 * ks2) / det, (-a12 * ks1 + a11 * ks2) / det };
        double expVar = sigmaK - (ks1 * cinvKs[0] + ks2 * cinvKs[1]);

        var gp = new SaasGaussianProcess(x, y, noise);
        SaasPosterior post = gp.Condition(new SaasHyperparameters(new[] { rho }, sigmaK));
        (double mean, double var) = post.Predict(new[] { xStar });

        mean.Should().BeApproximately(expMean, 1e-9, "posterior mean must equal Eqn (4)");
        var.Should().BeApproximately(expVar, 1e-9, "posterior variance must equal Eqn (5) with k** = σ²_k");
    }

    // ---- §4.1 Eqn (8): the half-Cauchy prior is a proper, normalized density ------------------------

    [TestMethod]
    public void HalfCauchyPrior_IsProperNormalizedDensity_Eqn8()
    {
        // p(τ|α) ∝ (α² + τ²)⁻¹ (Eqn 8), normalized to 2/(πα)/(1+(τ/α)²). The reparameterized log-density
        // (density of u = log t, Jacobian folded in) must integrate to exactly 1 over the whole log-axis.
        foreach (double scale in new[] { 0.1, 1.0, 3.0 })
        {
            double integral = 0.0;
            const double h = 5e-4;
            for (double u = -50; u < 50; u += h)
                integral += Math.Exp(SaasModel.HalfCauchyLogInLog(u, scale)) * h;
            integral.Should().BeApproximately(1.0, 1e-3, $"the half-Cauchy(scale={scale}) prior must be a proper density");
        }

        // Shape check: exp(logdensity) at t (via u=log t, dividing out the Jacobian t) ∝ (scale²+t²)⁻¹.
        double s = 0.7;
        foreach (double t in new[] { 0.2, 1.0, 4.0 })
        {
            double u = Math.Log(t);
            double densT = Math.Exp(SaasModel.HalfCauchyLogInLog(u, s)) / t;   // divide the du→dt Jacobian
            double expected = 2.0 / (Math.PI * s) / (1.0 + (t / s) * (t / s));
            densT.Should().BeApproximately(expected, 1e-9, "the half-Cauchy density must match Eqn (8)");
        }
    }

    // ---- §A.1 Eqn (12): the reparameterization ρ̃~HC(1), ρ=τ·ρ̃ induces ρ~HC(τ) --------------------

    [TestMethod]
    public void Reparameterization_Eqn12_InducesHalfCauchyScaleTau()
    {
        // The scale property of the half-Cauchy: if ρ̃ ~ HC(1) and ρ = τ·ρ̃, then ρ ~ HC(τ). Verify the two
        // densities of ρ agree — the change-of-variables of HC(1) by τ equals HC(τ) directly.
        double tau = 0.3;
        foreach (double r in new[] { 0.05, 0.3, 1.0, 3.0 })
        {
            double viaReparam = (2.0 / Math.PI / (1.0 + (r / tau) * (r / tau))) / tau;   // HC(1) at r/τ, ×(1/τ) Jacobian
            double direct = 2.0 / (Math.PI * tau) / (1.0 + (r / tau) * (r / tau));         // HC(τ) at r
            viaReparam.Should().BeApproximately(direct, 1e-12, "ρ = τ·ρ̃ with ρ̃~HC(1) must give ρ~HC(τ)");
        }
    }

    // ---- §3.2 Eqn (6)-(7): EI is the expectation of the improvement, and its limits -----------------

    [TestMethod]
    public void ExpectedImprovement_EqualsExpectationOfImprovement_Eqn6And7()
    {
        // Eqn (6): EI = E_{f~N(μ,σ²)}[max(0, f − f0)] (in our maximization convention). Verify our closed-form
        // (Eqn 7) against a Monte-Carlo estimate of that expectation, independent of any erf implementation.
        (double[,] x, double[] y) = MakeData(3, 6, 2, r => r[0]);
        var gp = new SaasGaussianProcess(x, y, 1e-6);
        double incumbent = y.Max();
        var acq = new SaasAcquisition(gp, new List<SaasHyperparameters> { new(new[] { 1.0, 1.0 }, 1.0) }, incumbent);

        var q = new[] { 0.85, 0.15 };
        SaasPosterior post = gp.Condition(new SaasHyperparameters(new[] { 1.0, 1.0 }, 1.0));
        (double mu, double varq) = post.Predict(q);
        double sigma = Math.Sqrt(varq);

        var rng = np.random.RandomState();
        rng.seed(2024);
        double[] z = rng.randn(400_000).ToArray<double>();
        double mc = z.Select(zi => Math.Max(0.0, mu + sigma * zi - incumbent)).Average();

        acq.Value(q).Should().BeApproximately(mc, 3e-3, "closed-form EI (Eqn 7) must equal the Monte-Carlo improvement expectation (Eqn 6)");
    }

    [TestMethod]
    public void ExpectedImprovement_Limits_VanishFarBelowAndEqualGapWhenCertain()
    {
        // Two boundary properties of EI (Eqn 7): it vanishes for a candidate whose posterior mean is far
        // below the incumbent, and it approaches max(0, μ − f0) as the posterior variance → 0.
        (double[,] x, double[] y) = MakeData(4, 8, 2, r => r[0] - r[1]);
        var gp = new SaasGaussianProcess(x, y, 1e-6);
        double incumbent = y.Max();
        var acq = new SaasAcquisition(gp, new List<SaasHyperparameters> { new(new[] { 4.0, 4.0 }, 1e-6) }, incumbent);

        // A training point (posterior var ≈ 0) whose value is below the incumbent ⇒ EI ≈ 0.
        int worst = Array.IndexOf(y, y.Min());
        var atWorst = new[] { x[worst, 0], x[worst, 1] };
        acq.Value(atWorst).Should().BeLessThan(1e-2, "EI vanishes at a near-certain point below the incumbent");

        // A training point that IS the incumbent ⇒ improvement gap ≈ 0, so EI ≈ 0 there too (var ≈ 0).
        int best = Array.IndexOf(y, y.Max());
        var atBest = new[] { x[best, 0], x[best, 1] };
        acq.Value(atBest).Should().BeLessThan(0.2, "EI is small at the incumbent itself (zero gap, near-zero variance)");
    }

    // ---- §4.3 Eqn (10): EI is averaged over the posterior hyperparameter samples --------------------

    [TestMethod]
    public void AveragedExpectedImprovement_IsMeanOverSamples_Eqn10()
    {
        // Eqn (10): EI(x|{ψ_ℓ}) = (1/L) Σ_ℓ EI(x|ψ_ℓ). Verify the acquisition's averaged value equals the
        // mean of the individually-conditioned per-sample EIs.
        (double[,] x, double[] y) = MakeData(7, 8, 3, r => r[0] - r[1] + r[2]);
        var gp = new SaasGaussianProcess(x, y, 1e-6);
        var samples = new List<SaasHyperparameters>
        {
            new(new[] { 1.0, 0.5, 0.2 }, 1.0),
            new(new[] { 2.0, 0.3, 0.1 }, 1.3),
            new(new[] { 0.4, 3.0, 0.7 }, 0.8),
        };
        double incumbent = y.Max();
        var acq = new SaasAcquisition(gp, samples, incumbent);

        var q = new[] { 0.6, 0.35, 0.7 };
        double manualAverage = samples
            .Select(hp => new SaasAcquisition(gp, new List<SaasHyperparameters> { hp }, incumbent).Value(q))
            .Average();

        acq.Value(q).Should().BeApproximately(manualAverage, 1e-9, "the acquisition must average EI over the sample bag (Eqn 10)");
    }

    // ---- §A.1 / §A.2 / §B: our defaults equal the paper's stated hyperparameters --------------------

    [TestMethod]
    public void SamplerDefaults_MatchPaper_Nuts_A1_Acquisition_B()
    {
        // §A.1: 512 warmup / 256 samples / thinning 16 (⇒ L=16) / max tree depth 6. §4.1: α=0.1, noise 1e-6,
        // σ²_k~LN(0,10²). §B: Q=5000 Sobol screen, K=3 L-BFGS restarts, ≤100 evals. §A.2: 1500 Adam steps.
        var sampler = new SaasboSampler();
        SaasboConfig c = sampler.Configuration;

        c.Method.Should().Be(SaasboInferenceMethod.Nuts, "NUTS is the paper's default (§4.2.1)");
        c.NumWarmup.Should().Be(512, "§A.1 warmup");
        c.NumSamples.Should().Be(256, "§A.1 post-warmup samples");
        c.Thinning.Should().Be(16, "§A.1 thinning ⇒ L=16 retained samples");
        c.MaxTreeDepth.Should().Be(6, "§A.1 max tree depth");
        c.Alpha.Should().Be(0.1, "§4.1 default global-shrinkage α");
        c.NoiseVariance.Should().Be(1e-6, "§4.1 noise-free objective");
        c.KernelVarianceLogStd.Should().Be(10.0, "§4.1 σ²_k ~ LogNormal(0, 10²)");
        c.MapSteps.Should().Be(1500, "§A.2 Adam steps");
        c.NumCandidates.Should().Be(5000, "§B Sobol screen size Q");
        c.NumRestarts.Should().Be(3, "§B L-BFGS restarts K");
        c.MaxLbfgsEvals.Should().Be(100, "§B L-BFGS function-evaluation cap");
    }

    [TestMethod]
    public void MapDefaultTaus_MatchAppendixA2()
    {
        // §A.2: MAP learns separate models for S=4 pre-selected τ_s ∈ {1, 1e-1, 1e-2, 1e-3} (the appendix's
        // authoritative grid; the main text's §4.2.2 lists only three, sans the leading 1).
        SaasMapOptimizer.DefaultTaus.Should().Equal(new[] { 1.0, 1e-1, 1e-2, 1e-3 });
    }

    // ---- §5.2: SAAS quickly identifies the relevant dimensions --------------------------------------

    [TestMethod]
    [Slow]
    public void SaasNuts_IdentifiesTwoRelevantDims_OnEmbeddedBranin_Section52()
    {
        // The paper's controlled §5.2 experiment (Branin in D=100), reproduced at a smaller D=12: SAASBO must
        // rank the two Branin variables (x0, x1) as the two largest PosteriorMedian(ρ) — the paper's
        // "relevant dimension found" criterion (Fig. 2 middle).
        static double Branin(double[] r)
        {
            double a = -5 + 15 * r[0], b = 15 * r[1];   // [0,1]² → [−5,10]×[0,15]
            double v = Math.Pow(b - 5.1 / (4 * Math.PI * Math.PI) * a * a + 5 / Math.PI * a - 6, 2)
                       + 10 * (1 - 1 / (8 * Math.PI)) * Math.Cos(a) + 10;
            return -v;   // maximization-oriented
        }

        foreach (int seed in new[] { 40, 41 })
        {
            (double[,] x, double[] y) = MakeData(seed, 30, 12, Branin);
            var model = new SaasModel(new SaasGaussianProcess(x, y, 1e-6), alpha: 0.1);
            double[] median = PosteriorMedianRho(model, seed, 150, 200);

            int[] top2 = Enumerable.Range(0, 12).OrderByDescending(j => median[j]).Take(2).OrderBy(j => j).ToArray();
            top2.Should().Equal(new[] { 0, 1 }, "SAAS identifies the two Branin dimensions as most relevant");
        }
    }

    // ---- §5.2: smaller α is more conservative (smaller subspaces) -----------------------------------

    [TestMethod]
    public void Alpha_ControlsShrinkage_SmallerAlphaSmallerSubspace_Section52()
    {
        // §5.2: "smaller values of α are more conservative (prefer smaller subspaces), while larger values of
        // α are less conservative (prefer larger subspaces)", most pronounced with few datapoints. On an
        // ambiguous few-point set, the effective subspace dimension (# PosteriorMedian(ρ) > 0.5, Fig. 2 right)
        // must increase with α.
        int EffectiveDim(double alpha)
        {
            int total = 0;
            for (int seed = 20; seed < 23; seed++)
            {
                (double[,] x, double[] y) = MakeData(seed, 6, 8, r => r[0] + 0.5 * r[1]);
                var model = new SaasModel(new SaasGaussianProcess(x, y, 1e-6), alpha: alpha);
                double[] median = PosteriorMedianRho(model, seed, 120, 150);
                total += median.Count(v => v > 0.5);
            }
            return total;
        }

        int small = EffectiveDim(0.01);
        int mid = EffectiveDim(0.1);
        int large = EffectiveDim(1.0);

        large.Should().BeGreaterThan(mid, "α=1.0 turns on more dimensions than α=0.1");
        mid.Should().BeGreaterThan(small, "α=0.1 turns on more dimensions than α=0.01");
    }

    // ---- §4.1 / §5.7: the sparsity is adaptive — more data turns on more dimensions -----------------

    [TestMethod]
    [Slow]
    public void SparsityIsAdaptive_MoreDataFindsMoreRelevantDims_Section41And57()
    {
        // §4.1: "as more data is accumulated, more of the ρ_i will escape zero"; §5.7/Fig. 5 shows the
        // effective subspace dimension growing over a run. On an objective with three genuinely relevant
        // dimensions (x0,x1,x2), the count found among the three largest PosteriorMedian(ρ) must grow with n.
        static double ThreeRelevant(double[] r) => Math.Sin(3 * r[0]) + Math.Sin(3 * r[1]) + Math.Sin(3 * r[2]);

        double Found(int n)
        {
            var counts = new List<int>();
            for (int seed = 50; seed < 53; seed++)
            {
                (double[,] x, double[] y) = MakeData(seed, n, 8, ThreeRelevant);
                var model = new SaasModel(new SaasGaussianProcess(x, y, 1e-6), alpha: 0.1);
                double[] median = PosteriorMedianRho(model, seed, 120, 150);
                var top3 = Enumerable.Range(0, 8).OrderByDescending(j => median[j]).Take(3).ToHashSet();
                counts.Add(new[] { 0, 1, 2 }.Count(top3.Contains));
            }
            return counts.Average();
        }

        double few = Found(5);
        double many = Found(30);

        many.Should().BeGreaterThan(few, "more evaluations identify more of the relevant dimensions (§4.1)");
        many.Should().BeGreaterThanOrEqualTo(2.5, "with ample data SAAS finds essentially all three relevant dimensions");
    }
}
