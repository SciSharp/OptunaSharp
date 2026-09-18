using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Samplers.ModelBased.Saas;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Unit tests for the beyond-core, paper-faithful SAASBO GP primitives (Eriksson &amp; Jankowiak 2021):
///     the RBF ARD GP with the SAAS prior (<see cref="SaasGaussianProcess" /> / <see cref="SaasModel" />),
///     the No-U-Turn sampler (<see cref="NoUTurnSampler" />), the MAP optimizer
///     (<see cref="SaasMapOptimizer" />), and the sample-averaged EI acquisition
///     (<see cref="SaasAcquisition" />). The analytic gradients are validated against finite differences
///     (the same self-check the Matérn GPRegressor uses in place of PyTorch autograd), NUTS is validated by
///     recovering the moments of a known Gaussian, and the load-bearing paper claim — that SAAS quickly
///     identifies the sparse relevant subspace (§5.2) — is asserted directly. SAASBO is a behavioral,
///     convergence-validated port (HMC is stochastic and not bit-exact vs NumPyro), so these gates assert
///     correctness properties rather than byte parity.
/// </summary>
[TestClass]
public class SaasGpPrimitivesTests
{
    /// <summary>Builds a small standardized, maximization-oriented training set from a seeded RNG.</summary>
    /// <param name="seed">The MT19937 seed.</param>
    /// <param name="n">The number of training points.</param>
    /// <param name="d">The input dimension.</param>
    /// <param name="f">The (pre-standardization) objective evaluated per row.</param>
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

    // ---- analytic gradients vs finite differences ------------------------------------------------

    [TestMethod]
    public void GpMarginalLogLikelihoodGradient_MatchesFiniteDifference()
    {
        (double[,] x, double[] y) = MakeData(0, 8, 4, r => Math.Sin(3 * r[0]) + 0.5 * r[1] - r[2]);
        var gp = new SaasGaussianProcess(x, y, 1e-6);
        var rho = new[] { 0.7, 1.3, 0.2, 2.1 };
        double kv = 1.4;

        (double _, double[] dRho, double dKv) = gp.MarginalLogLikelihoodAndGrad(new SaasHyperparameters(rho, kv));

        const double h = 1e-6;
        for (int j = 0; j < rho.Length; j++)
        {
            var rp = (double[])rho.Clone(); rp[j] += h;
            var rm = (double[])rho.Clone(); rm[j] -= h;
            double num = (gp.MarginalLogLikelihoodAndGrad(new SaasHyperparameters(rp, kv)).Mll
                          - gp.MarginalLogLikelihoodAndGrad(new SaasHyperparameters(rm, kv)).Mll) / (2 * h);
            dRho[j].Should().BeApproximately(num, 1e-4);
        }
        double numKv = (gp.MarginalLogLikelihoodAndGrad(new SaasHyperparameters(rho, kv + h)).Mll
                        - gp.MarginalLogLikelihoodAndGrad(new SaasHyperparameters(rho, kv - h)).Mll) / (2 * h);
        dKv.Should().BeApproximately(numKv, 1e-4);
    }

    [TestMethod]
    public void SaasLogJointGradient_MatchesFiniteDifference()
    {
        (double[,] x, double[] y) = MakeData(1, 7, 5, r => Math.Cos(2 * r[1]) - r[3]);
        var model = new SaasModel(new SaasGaussianProcess(x, y, 1e-6), alpha: 0.1);

        var rng = np.random.RandomState(); rng.seed(3);
        var theta = new double[model.LatentDim];
        for (int j = 0; j < theta.Length; j++) theta[j] = 0.3 * ((double)rng.rand() - 0.5);

        (double _, double[] grad) = model.LogJointAndGrad(theta);
        const double h = 1e-6;
        for (int j = 0; j < theta.Length; j++)
        {
            var tp = (double[])theta.Clone(); tp[j] += h;
            var tm = (double[])theta.Clone(); tm[j] -= h;
            double num = (model.LogJointAndGrad(tp).LogJoint - model.LogJointAndGrad(tm).LogJoint) / (2 * h);
            grad[j].Should().BeApproximately(num, 1e-4);
        }
    }

    [TestMethod]
    public void MapObjectiveGradient_MatchesFiniteDifference()
    {
        (double[,] x, double[] y) = MakeData(2, 6, 4, r => r[0] * r[0] - r[2]);
        var model = new SaasModel(new SaasGaussianProcess(x, y, 1e-6));
        const double tau = 0.1;

        var q = new double[model.Dim + 1];
        var rng = np.random.RandomState(); rng.seed(4);
        for (int j = 0; j < q.Length; j++) q[j] = 0.2 * ((double)rng.rand() - 0.5);

        (double _, double[] grad) = model.MapObjectiveAndGrad(q, tau);
        const double h = 1e-6;
        for (int j = 0; j < q.Length; j++)
        {
            var qp = (double[])q.Clone(); qp[j] += h;
            var qm = (double[])q.Clone(); qm[j] -= h;
            double num = (model.MapObjectiveAndGrad(qp, tau).Objective - model.MapObjectiveAndGrad(qm, tau).Objective) / (2 * h);
            grad[j].Should().BeApproximately(num, 1e-4);
        }
    }

    [TestMethod]
    public void HalfCauchyLogInLog_DerivativeMatchesNumerical()
    {
        // The reparameterized half-Cauchy log-density and its derivative underpin the SAAS prior gradient.
        foreach (double scale in new[] { 0.1, 1.0, 3.0 })
            foreach (double u in new[] { -2.0, -0.5, 0.0, 1.5 })
            {
                const double h = 1e-6;
                double num = (SaasModel.HalfCauchyLogInLog(u + h, scale) - SaasModel.HalfCauchyLogInLog(u - h, scale)) / (2 * h);
                SaasModel.HalfCauchyLogInLogDeriv(u, scale).Should().BeApproximately(num, 1e-5);
            }
    }

    // ---- posterior identities --------------------------------------------------------------------

    [TestMethod]
    public void Posterior_InterpolatesTrainingDataAtNoiseFloor()
    {
        // A noise-free GP interpolates: at a training input the posterior mean ≈ the target and variance ≈ 0.
        (double[,] x, double[] y) = MakeData(5, 6, 3, r => Math.Sin(4 * r[0]) + r[1]);
        var gp = new SaasGaussianProcess(x, y, 1e-6);
        SaasPosterior post = gp.Condition(new SaasHyperparameters(new[] { 1.0, 1.0, 1.0 }, 1.0));

        for (int i = 0; i < x.GetLength(0); i++)
        {
            var xi = new[] { x[i, 0], x[i, 1], x[i, 2] };
            (double mean, double var) = post.Predict(xi);
            mean.Should().BeApproximately(y[i], 1e-2);
            var.Should().BeLessThan(1e-2).And.BeGreaterThanOrEqualTo(0.0);
        }
    }

    [TestMethod]
    public void PosteriorWithGrad_MatchesFiniteDifference()
    {
        (double[,] x, double[] y) = MakeData(6, 7, 3, r => r[0] - r[1] * r[1] + Math.Sin(5 * r[2]));
        var gp = new SaasGaussianProcess(x, y, 1e-6);
        SaasPosterior post = gp.Condition(new SaasHyperparameters(new[] { 1.5, 0.8, 2.0 }, 1.2));

        var q = new[] { 0.42, 0.61, 0.27 };
        (double _, double _, double[] dMean, double[] dVar, bool _) = post.PredictWithGrad(q);
        const double h = 1e-6;
        for (int j = 0; j < q.Length; j++)
        {
            var qp = (double[])q.Clone(); qp[j] += h;
            var qm = (double[])q.Clone(); qm[j] -= h;
            (double mp, double vp) = post.Predict(qp);
            (double mm, double vm) = post.Predict(qm);
            dMean[j].Should().BeApproximately((mp - mm) / (2 * h), 1e-4);
            dVar[j].Should().BeApproximately((vp - vm) / (2 * h), 1e-4);
        }
    }

    // ---- NUTS correctness ------------------------------------------------------------------------

    [TestMethod]
    public void Nuts_RecoversDiagonalGaussianMoments()
    {
        // NUTS on N(mean, diag(var)) must recover the target means and variances — the sampler's ground-truth
        // correctness check, independent of the GP model it will later target.
        double[] targetVar = { 1.0, 4.0, 0.25 };
        double[] targetMean = { 0.5, -1.0, 2.0 };
        (double, double[]) Logp(double[] t)
        {
            double lp = 0; var g = new double[3];
            for (int k = 0; k < 3; k++)
            {
                double dd = t[k] - targetMean[k];
                lp += -0.5 * dd * dd / targetVar[k];
                g[k] = -dd / targetVar[k];
            }
            return (lp, g);
        }

        var rng = np.random.RandomState(); rng.seed(7);
        var draws = new NoUTurnSampler(numWarmup: 400, numSamples: 1500, maxTreeDepth: 8).Sample(Logp, new double[] { 0, 0, 0 }, rng);

        for (int k = 0; k < 3; k++)
        {
            double m = draws.Select(s => s[k]).Average();
            double v = draws.Select(s => (s[k] - m) * (s[k] - m)).Average();
            m.Should().BeApproximately(targetMean[k], 0.25);
            v.Should().BeApproximately(targetVar[k], 0.35 * targetVar[k] + 0.1);
        }
    }

    [TestMethod]
    public void SaasNuts_IdentifiesRelevantDimension()
    {
        // The paper's central §5.2 claim: SAAS quickly turns "off" irrelevant dimensions. On an objective
        // that depends only on x0, the posterior-median inverse length scale ρ0 must be the largest, and the
        // irrelevant ρ's near zero (the paper's "relevant dimension found" criterion).
        (double[,] x, double[] y) = MakeData(11, 25, 8, r => -(r[0] - 0.3) * (r[0] - 0.3));
        var model = new SaasModel(new SaasGaussianProcess(x, y, 1e-6), alpha: 0.1);

        var init = new double[model.LatentDim]; init[0] = Math.Log(0.1);
        var rng = np.random.RandomState(); rng.seed(11);
        var draws = new NoUTurnSampler(numWarmup: 150, numSamples: 200, maxTreeDepth: 6)
            .Sample(model.LogJointAndGrad, init, rng)
            .Select(model.HyperparametersFromTheta).ToList();

        int d = model.Dim;
        var medianRho = new double[d];
        for (int j = 0; j < d; j++)
        {
            double[] col = draws.Select(h => h.Rho[j]).OrderBy(v => v).ToArray();
            medianRho[j] = col[col.Length / 2];
        }

        int argmax = 0;
        for (int j = 1; j < d; j++) if (medianRho[j] > medianRho[argmax]) argmax = j;
        argmax.Should().Be(0, "SAAS should identify x0 as the relevant dimension");

        // The relevant ρ0 should dominate every irrelevant ρ by a wide margin (they are shrunk toward zero).
        for (int j = 1; j < d; j++)
            medianRho[0].Should().BeGreaterThan(medianRho[j] + 0.1);
    }

    // ---- MAP inference ---------------------------------------------------------------------------

    [TestMethod]
    public void Map_IdentifiesRelevantDimension()
    {
        // MAP (Adam over the τ grid + leave-one-out selection) also recovers the relevant subspace, though
        // the paper finds it less robust than NUTS.
        (double[,] x, double[] y) = MakeData(13, 25, 6, r => -(r[1] - 0.6) * (r[1] - 0.6));
        var model = new SaasModel(new SaasGaussianProcess(x, y, 1e-6));

        SaasHyperparameters hp = SaasMapOptimizer.Optimize(model, steps: 500);

        int argmax = 0;
        for (int j = 1; j < hp.Rho.Length; j++) if (hp.Rho[j] > hp.Rho[argmax]) argmax = j;
        argmax.Should().Be(1, "MAP should identify x1 as the relevant dimension");
    }

    [TestMethod]
    public void LeaveOneOut_IsFiniteAndPrefersFittedModel()
    {
        // The LOO predictive log likelihood MAP selects on must be finite and rank a data-consistent length
        // scale above a wildly mis-scaled one.
        (double[,] x, double[] y) = MakeData(14, 12, 3, r => Math.Sin(4 * r[0]));
        var gp = new SaasGaussianProcess(x, y, 1e-6);

        double looGood = gp.LeaveOneOutLogLikelihood(new SaasHyperparameters(new[] { 4.0, 0.01, 0.01 }, 1.0));
        double looBad = gp.LeaveOneOutLogLikelihood(new SaasHyperparameters(new[] { 0.001, 0.001, 0.001 }, 1.0));

        looGood.Should().NotBe(double.NegativeInfinity);
        double.IsNaN(looGood).Should().BeFalse();
        looGood.Should().BeGreaterThan(looBad, "a length scale that resolves the x0 structure should predict held-out points better");
    }

    // ---- acquisition -----------------------------------------------------------------------------

    [TestMethod]
    public void AveragedExpectedImprovement_IsNonNegativeAndGradientMatchesFiniteDifference()
    {
        (double[,] x, double[] y) = MakeData(9, 8, 3, r => r[0] - r[1]);
        var gp = new SaasGaussianProcess(x, y, 1e-6);
        var samples = new List<SaasHyperparameters>
        {
            new(new[] { 1.0, 0.5, 0.2 }, 1.0),
            new(new[] { 2.0, 0.3, 0.1 }, 1.3),
        };
        double incumbent = y.Max();
        var acq = new SaasAcquisition(gp, samples, incumbent);
        acq.SampleCount.Should().Be(2);

        var q = new[] { 0.55, 0.4, 0.7 };
        (double value, double[] grad) = acq.ValueAndGrad(q);
        value.Should().BeGreaterThanOrEqualTo(0.0);
        acq.Value(q).Should().BeApproximately(value, 1e-9);

        const double h = 1e-6;
        for (int j = 0; j < q.Length; j++)
        {
            var qp = (double[])q.Clone(); qp[j] += h;
            var qm = (double[])q.Clone(); qm[j] -= h;
            double num = (acq.Value(qp) - acq.Value(qm)) / (2 * h);
            grad[j].Should().BeApproximately(num, 1e-4);
        }
    }

    [TestMethod]
    public void AcquisitionMaximize_FindsAHighExpectedImprovementPoint()
    {
        // The Sobol-screen + L-BFGS-B maximizer must return a unit-box point whose averaged EI is at least as
        // large as the best of a fresh random screen — i.e. the optimization actually improves on sampling.
        (double[,] x, double[] y) = MakeData(10, 10, 3, r => -(r[0] - 0.2) * (r[0] - 0.2) - (r[1] - 0.8) * (r[1] - 0.8));
        var gp = new SaasGaussianProcess(x, y, 1e-6);
        var samples = new List<SaasHyperparameters> { new(new[] { 3.0, 3.0, 0.05 }, 1.0) };
        var acq = new SaasAcquisition(gp, samples, y.Max());

        double[] best = acq.Maximize(numCandidates: 512, numRestarts: 3, seed: 123);
        best.Should().OnlyContain(v => v >= 0.0 && v <= 1.0);

        var rng = np.random.RandomState(); rng.seed(999);
        double bestRandom = 0.0;
        for (int i = 0; i < 512; i++)
        {
            var p = new double[3];
            for (int j = 0; j < 3; j++) p[j] = (double)rng.rand();
            bestRandom = Math.Max(bestRandom, acq.Value(p));
        }
        acq.Value(best).Should().BeGreaterThanOrEqualTo(bestRandom - 1e-9);
    }
}
