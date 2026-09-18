using System;
using AwesomeAssertions;
using OptunaSharp.Samplers.ModelBased.Gp;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Unit tests for the beyond-core, paper-faithful GP primitives REMBO is built on (Wang et al. 2016):
///     the DIRECT global optimizer (<see cref="Direct" />), the DIRECT + CMA-ES best-of optimizer
///     (<see cref="GlobalMinimizer" />), and the isotropic squared-exponential GP (<see cref="SquaredExponentialGp" />).
///     These are validated against known answers (analytic minima; GP interpolation identities), the same
///     way the paper's own experiments use the Branin function, so the REMBO rewire that sits on top can be
///     trusted.
/// </summary>
[TestClass]
public class ModelBasedGpPrimitivesTests
{
    // ---- DIRECT ----------------------------------------------------------------------------------

    [TestMethod]
    public void Direct_FindsShiftedSphereMinimum()
    {
        // Global min of (x0−0.3)² + (x1+1.2)² is 0 at (0.3, −1.2), interior to [−5, 5]².
        (double[] x, double f) = Direct.Minimize(
            p => (p[0] - 0.3) * (p[0] - 0.3) + (p[1] + 1.2) * (p[1] + 1.2),
            new[] { -5.0, -5.0 }, new[] { 5.0, 5.0 }, maxEvals: 4000);

        f.Should().BeLessThan(1e-4);
        x[0].Should().BeApproximately(0.3, 0.05);
        x[1].Should().BeApproximately(-1.2, 0.05);
    }

    [TestMethod]
    public void Direct_FindsBraninGlobalMinimum()
    {
        // Branin (the paper's own BBOB test function) has global minimum 0.397887; box x0∈[−5,10], x1∈[0,15].
        (double[] _, double f) = Direct.Minimize(
            Branin, new[] { -5.0, 0.0 }, new[] { 10.0, 15.0 }, maxEvals: 6000);
        f.Should().BeApproximately(0.397887, 1e-2);
    }

    [TestMethod]
    public void Direct_IsDeterministic()
    {
        // DIRECT uses no RNG, so repeated runs are bit-identical.
        (double[] x1, double f1) = Direct.Minimize(Branin, new[] { -5.0, 0.0 }, new[] { 10.0, 15.0 }, 3000);
        (double[] x2, double f2) = Direct.Minimize(Branin, new[] { -5.0, 0.0 }, new[] { 10.0, 15.0 }, 3000);
        f1.Should().Be(f2);
        x1.Should().Equal(x2);
    }

    [TestMethod]
    public void Direct_RejectsIllFormedBounds()
    {
        Action bad = () => Direct.Minimize(p => p[0], new[] { 1.0 }, new[] { 1.0 }, 100);
        bad.Should().Throw<ArgumentException>("an upper bound must strictly exceed its lower bound");
    }

    // ---- GlobalMinimizer (DIRECT + CMA-ES best-of) -----------------------------------------------

    [TestMethod]
    public void GlobalMinimizer_SolvesSphereAndIsSeedReproducible()
    {
        Func<double[], double> sphere = p => (p[0] - 0.3) * (p[0] - 0.3) + (p[1] + 1.2) * (p[1] + 1.2);
        (double[] xa, double fa) = GlobalMinimizer.Minimize(sphere, new[] { -5.0, -5.0 }, new[] { 5.0, 5.0 }, seed: 1, 2000, 2000);
        (double[] xb, double fb) = GlobalMinimizer.Minimize(sphere, new[] { -5.0, -5.0 }, new[] { 5.0, 5.0 }, seed: 1, 2000, 2000);

        fa.Should().BeLessThan(1e-4);
        fa.Should().Be(fb, "the CMA-ES half is a pure function of the seed and DIRECT is deterministic");
        xa.Should().Equal(xb);
    }

    [TestMethod]
    public void GlobalMinimizer_NeverWorseThanDirectAlone()
    {
        // Best-of two optimizers cannot do worse than the deterministic DIRECT half.
        (double[] _, double fDirect) = Direct.Minimize(Branin, new[] { -5.0, 0.0 }, new[] { 10.0, 15.0 }, 3000);
        (double[] _, double fGlobal) = GlobalMinimizer.Minimize(Branin, new[] { -5.0, 0.0 }, new[] { 10.0, 15.0 }, 5, 3000, 3000);
        fGlobal.Should().BeLessThanOrEqualTo(fDirect + 1e-12);
    }

    // ---- SquaredExponentialGp --------------------------------------------------------------------

    [TestMethod]
    public void SquaredExponentialGp_InterpolatesTrainingPoints()
    {
        // A noise-free GP (tiny nugget) reproduces the training targets at the training inputs, with a
        // vanishing posterior variance there.
        double[,] x = { { 0.1 }, { 0.5 }, { 0.9 } };
        double[] y = { -1.0, 2.0, -0.5 };
        var gp = SquaredExponentialGp.Fit(x, y, new[] { false }, lengthScale: 0.2, categoricalScale: 0.0);

        for (int i = 0; i < 3; i++)
        {
            (double mean, double var) = gp.Posterior(new[] { x[i, 0] });
            mean.Should().BeApproximately(y[i], 1e-3);
            var.Should().BeLessThan(1e-3);
        }
    }

    [TestMethod]
    public void SquaredExponentialGp_HasPositiveVarianceAwayFromData()
    {
        double[,] x = { { 0.1 }, { 0.9 } };
        double[] y = { -1.0, 1.0 };
        var gp = SquaredExponentialGp.Fit(x, y, new[] { false }, lengthScale: 0.2, categoricalScale: 0.0);
        (double _, double var) = gp.Posterior(new[] { 0.5 });
        var.Should().BeGreaterThan(0.0, "the posterior is uncertain between observations");
    }

    [TestMethod]
    public void SquaredExponentialGp_ConcentratedLogLikelihoodIsFiniteAndLengthScaleDependent()
    {
        double[,] x = { { 0.1 }, { 0.5 }, { 0.9 } };
        double[] y = { -1.0, 2.0, -0.5 };
        double a = SquaredExponentialGp.ConcentratedLogLikelihood(x, y, new[] { false }, 0.1, 0.0);
        double b = SquaredExponentialGp.ConcentratedLogLikelihood(x, y, new[] { false }, 2.0, 0.0);
        double.IsFinite(a).Should().BeTrue();
        double.IsFinite(b).Should().BeTrue();
        a.Should().NotBe(b, "the marginal likelihood must respond to the length scale so Algorithm 3 can optimize it");
        // A non-positive length scale is infeasible, not an exception.
        SquaredExponentialGp.ConcentratedLogLikelihood(x, y, new[] { false }, 0.0, 0.0)
            .Should().Be(double.NegativeInfinity);
    }

    [TestMethod]
    public void SquaredExponentialGp_CategoricalHammingKernelInterpolatesCategories()
    {
        // One categorical coordinate with two categories (coded 0 and 1). The §3.2 Hamming kernel must still
        // interpolate each observed category exactly.
        double[,] x = { { 0.0 }, { 1.0 } };
        double[] y = { 1.0, -1.0 };
        var gp = SquaredExponentialGp.Fit(x, y, new[] { true }, lengthScale: 1.0, categoricalScale: 1.0);
        gp.Posterior(new[] { 0.0 }).Mean.Should().BeApproximately(1.0, 1e-3);
        gp.Posterior(new[] { 1.0 }).Mean.Should().BeApproximately(-1.0, 1e-3);
    }

    /// <summary>The Branin-Hoo test function (global minimum 0.397887) used by the paper's BBOB experiments.</summary>
    /// <param name="p">The two-dimensional input point.</param>
    /// <returns>The Branin value at <paramref name="p" />.</returns>
    private static double Branin(double[] p)
    {
        double b = 5.1 / (4 * Math.PI * Math.PI), c = 5 / Math.PI, t = 1 / (8 * Math.PI);
        double x1 = p[0], x2 = p[1];
        return Math.Pow(x2 - b * x1 * x1 + c * x1 - 6, 2) + 10 * (1 - t) * Math.Cos(x1) + 10;
    }
}
