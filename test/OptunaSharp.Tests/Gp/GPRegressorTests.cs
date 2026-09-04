using System;
using AwesomeAssertions;
using OptunaSharp.Gp;

namespace OptunaSharp.Tests.Gp;

/// <summary>
///     Unit coverage for the GP numeric core (PORTING_PLAN Phase 3.2 — the §7 autodiff decision
///     gate): the Matérn-5/2 kernel forward, the marginal-log-likelihood, the <see cref="Prior" />,
///     and the hand-derived analytic gradient in <see cref="GPRegressor.LossAndGrad" /> that
///     replaces PyTorch autograd. Forward values are pinned to figures produced by live Optuna
///     (torch); the gradient is checked against a central finite difference — an oracle-free proof
///     that the closed form is internally consistent. (Byte-tolerance parity of loss AND gradient vs
///     live autograd lives in the interop <c>GpFitParityTests</c>.)
/// </summary>
[TestClass]
public class GPRegressorTests
{
    private const double MinNoise = 1e-6;

    private static GPRegressor MakeCaseA()
        => new(
            new[] { false },
            new[,] { { 0.2 }, { 0.5 }, { 0.8 }, { 0.4 } },
            new[] { 1.0, -0.5, 0.3, -0.8 },
            new[] { 2.0 }, 1.5, 0.1);

    // ---- kernel forward -----------------------------------------------------------------------

    [TestMethod]
    public void Kernel_Numerical_MatchesOptuna()
    {
        double[,] k = MakeCaseA().Kernel();
        // kernel(x, x) == kernel_scale; off-diagonal from live Optuna.
        k[0, 0].Should().BeApproximately(1.5, 1e-12);
        k[0, 1].Should().BeApproximately(1.306205872197611, 1e-12);
        k[1, 0].Should().BeApproximately(k[0, 1], 1e-15, "the kernel is symmetric");
    }

    [TestMethod]
    public void Kernel_Categorical_IsHammingDistance()
    {
        // Two distinct categories at {0,1}; equal points give kernel_scale, distinct give the
        // Matérn value at Hamming distance 1 — verified against live Optuna.
        var gpr = new GPRegressor(
            new[] { true },
            new[,] { { 0.0 }, { 1.0 }, { 0.0 }, { 1.0 } },
            new[] { 0.5, -0.5, 0.4, -0.6 },
            new[] { 1.0 }, 1.0, 0.2);
        double[,] k = gpr.Kernel();
        k[0, 0].Should().BeApproximately(1.0, 1e-12);
        k[0, 2].Should().BeApproximately(1.0, 1e-12, "same category → distance 0");
        k[0, 1].Should().BeApproximately(0.5239941088318203, 1e-12, "different category → Hamming 1");
    }

    // ---- marginal log likelihood + prior ------------------------------------------------------

    [TestMethod]
    public void MarginalLogLikelihood_MatchesOptuna()
    {
        MakeCaseA().MarginalLogLikelihood().Should().BeApproximately(-4.6185483529016045, 1e-10);

        var caseB = new GPRegressor(
            new[] { true },
            new[,] { { 0.0 }, { 1.0 }, { 0.0 }, { 1.0 } },
            new[] { 0.5, -0.5, 0.4, -0.6 },
            new[] { 1.0 }, 1.0, 0.2);
        caseB.MarginalLogLikelihood().Should().BeApproximately(0.48910236694019626, 1e-10);

        var caseC = new GPRegressor(
            new[] { false, true },
            new[,] { { 0.1, 0.9 }, { 0.5, 0.2 }, { 0.8, 0.5 }, { 0.3, 0.7 }, { 0.6, 0.1 } },
            new[] { 0.3, -0.7, 0.9, -0.2, 0.5 },
            new[] { 1.3, 0.7 }, 2.2, 0.05);
        caseC.MarginalLogLikelihood().Should().BeApproximately(-1.8073250636556817, 1e-10);
    }

    [TestMethod]
    public void DefaultLogPrior_MatchesOptuna()
    {
        Prior.DefaultLogPrior(MakeCaseA()).Should().BeApproximately(-4.574793401191241, 1e-12);

        var caseC = new GPRegressor(
            new[] { false, true },
            new[,] { { 0.1, 0.9 }, { 0.5, 0.2 }, { 0.8, 0.5 }, { 0.3, 0.7 }, { 0.6, 0.1 } },
            new[] { 0.3, -0.7, 0.9, -0.2, 0.5 },
            new[] { 1.3, 0.7 }, 2.2, 0.05);
        Prior.DefaultLogPrior(caseC).Should().BeApproximately(-3.630896086771349, 1e-12);
    }

    // ---- hand-derived gradient: finite-difference self-check ----------------------------------

    [TestMethod]
    public void LossAndGrad_Gradient_MatchesFiniteDifference()
    {
        var gpr = new GPRegressor(
            new[] { false, false },
            new[,] { { 0.1, 0.9 }, { 0.5, 0.2 }, { 0.8, 0.5 }, { 0.3, 0.7 }, { 0.6, 0.1 }, { 0.2, 0.4 } },
            new[] { 0.3, -0.7, 0.9, -0.2, 0.5, -0.1 },
            new[] { 1.0, 1.0 }, 1.0, 1.0);

        // A few raw-parameter points spanning the fit trajectory's regime.
        double[][] points =
        {
            new[] { 0.0, 0.0, 0.0, 0.0 },
            new[] { 1.5, -0.7, 0.4, -3.0 },
            new[] { 2.2, 0.9, -0.3, -6.0 },
        };
        const double eps = 1e-6;
        foreach (double[] raw in points)
        {
            (double _, double[] analytic) = gpr.LossAndGrad(raw, MinNoise, false);
            for (int k = 0; k < raw.Length; k++)
            {
                double[] plus = (double[])raw.Clone();
                double[] minus = (double[])raw.Clone();
                plus[k] += eps;
                minus[k] -= eps;
                double fd = (gpr.LossAndGrad(plus, MinNoise, false).Loss
                             - gpr.LossAndGrad(minus, MinNoise, false).Loss) / (2 * eps);
                analytic[k].Should().BeApproximately(fd, 1e-5,
                    $"analytic ∂loss/∂raw[{k}] must match the central difference");
            }
        }
    }

    [TestMethod]
    public void LossAndGrad_Deterministic_NoiseGradientIsZero()
    {
        var gpr = MakeCaseA();
        (double _, double[] grad) = gpr.LossAndGrad(new[] { 0.5, 0.2, -3.0 }, MinNoise, true);
        grad[^1].Should().Be(0.0, "deterministic mode fixes the noise, so its raw gradient is exactly 0");
        // and the noise variance is pinned to minimum_noise regardless of the raw noise param.
        gpr.NoiseVar.Should().Be(MinNoise);
    }

    [TestMethod]
    public void Posterior_MatchesOptuna()
    {
        // Fit the 8-point 1-D scenario, then check the posterior mean/var at three query points
        // against live Optuna's gpr.posterior. The fit matches Optuna to ~1e-14, so the posterior
        // matches to a tolerance.
        var x = new double[,] { { 0.1 }, { 0.35 }, { 0.5 }, { 0.72 }, { 0.9 }, { 0.25 }, { 0.6 }, { 0.8 } };
        var y = new[]
        {
            0.7306317300743783, 1.268428212469242, 0.2936730029150271, -1.1446831397092652,
            -0.849223481972128, 1.418560466127369, -0.5033619930308116, -1.2140247968738114
        };
        GPRegressor gpr = GPRegressor.FitKernelParams(x, y, new[] { false }, 1e-6, false, null, 1e-2);

        (double[] mean, double[] var) = gpr.Posterior(new double[,] { { 0.15 }, { 0.55 }, { 0.95 } });
        var expMean = new[] { 1.0276146154309682, -0.11449233506753842, -0.6020268275607927 };
        var expVar = new[] { 0.002437885425645714, 0.0005376554800664568, 0.0069967899184824045 };
        for (int i = 0; i < 3; i++)
        {
            mean[i].Should().BeApproximately(expMean[i], 1e-9);
            var[i].Should().BeApproximately(expVar[i], 1e-9);
            var[i].Should().BeGreaterThanOrEqualTo(0.0, "variance is clamped to be non-negative");
        }
    }

    [TestMethod]
    public void LossAndGrad_IsReproducible()
    {
        var gpr = MakeCaseA();
        var raw = new[] { 0.7, -0.3, -4.0 };
        (double l1, double[] g1) = gpr.LossAndGrad(raw, MinNoise, false);
        (double l2, double[] g2) = gpr.LossAndGrad(raw, MinNoise, false);
        l1.Should().Be(l2);
        g1.Should().BeEquivalentTo(g2);
    }
}
