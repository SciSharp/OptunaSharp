using System;
using System.Collections.Generic;
using AwesomeAssertions;
using OptunaSharp.Distributions;
using OptunaSharp.Gp;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Gp;

/// <summary>
///     Unit coverage for the GP acquisition functions (PORTING_PLAN Phase 3.2 parts 3b/3e —
///     <c>optuna/_gp/acqf.py</c>): <c>standard_logei</c>/<c>logei</c>, <see cref="Erf.Erfcx" />, and
///     <see cref="LogEI" />'s batched evaluation plus the hand-derived gradient that replaces
///     autograd (part 3b); and the constrained single-objective acqfs <see cref="LogPI" /> and
///     <see cref="LogCEI" /> = LogEI + Σ LogPI (part 3e-2). Scalar values are pinned to live-Optuna
///     float64 figures; gradients are checked against a central finite difference (oracle-free), and
///     the LogCEI composition is asserted exact. Byte-tolerance parity vs live autograd lives in the
///     interop <c>GpAcqfParityTests</c>.
/// </summary>
[TestClass]
public class AcqfTests
{
    [TestMethod]
    public void Erfcx_MatchesReference()
    {
        Erf.Erfcx(2.5).Should().BeApproximately(0.2108063640611436, 1e-13);
        Erf.Erfcx(20.0).Should().BeApproximately(0.028174348741051323, 1e-15);
        // erfcx(-x) = 2 exp(x²) - erfcx(x).
        Erf.Erfcx(-1.5).Should().BeApproximately(2.0 * Math.Exp(2.25) - Erf.Erfcx(1.5), 1e-12);
    }

    [TestMethod]
    public void StandardLogEi_MatchesOptuna()
    {
        Acqf.StandardLogEi(3.0).Should().BeApproximately(1.0987396653277077, 1e-13);
        Acqf.StandardLogEi(0.5).Should().BeApproximately(-0.35982768374506374, 1e-13);
        Acqf.StandardLogEi(-1.0).Should().BeApproximately(-2.4851210257126413, 1e-13);
        Acqf.StandardLogEi(-5.0).Should().BeApproximately(-16.744301162660967, 1e-13);
        // z < -25 branch (erfcx form).
        Acqf.StandardLogEi(-30.0).Should().BeApproximately(-457.7246537605979, 1e-9);
        Acqf.StandardLogEi(-100.0).Should().BeApproximately(-5010.129578800247, 1e-8);
    }

    [TestMethod]
    public void LogEi_MatchesOptuna()
        => Acqf.LogEi(1.2, 0.3, 0.5).Should().BeApproximately(-0.3200331084864203, 1e-13);

    private static (GPRegressor Gpr, LogEI Acqf) FitScenario()
    {
        var x = new double[,] { { 0.1 }, { 0.35 }, { 0.5 }, { 0.72 }, { 0.9 }, { 0.25 }, { 0.6 }, { 0.8 } };
        var y = new[]
        {
            0.7306317300743783, 1.268428212469242, 0.2936730029150271, -1.1446831397092652,
            -0.849223481972128, 1.418560466127369, -0.5033619930308116, -1.2140247968738114
        };
        GPRegressor gpr = GPRegressor.FitKernelParams(x, y, new[] { false }, 1e-6, false, null, 1e-2);
        var space = new System.Collections.Generic.Dictionary<string, BaseDistribution> { ["x"] = new FloatDistribution(0.0, 1.0) };
        return (gpr, new LogEI(gpr, new GpSearchSpace(space), threshold: 1.418560466127369));
    }

    [TestMethod]
    public void EvalAcqfNoGrad_MatchesOptuna()
    {
        (_, LogEI acqf) = FitScenario();
        double[] vals = acqf.EvalAcqfNoGrad(new double[,] { { 0.15 }, { 0.45 } });
        // From live Optuna (the C# fit matches Optuna to ~1e-14, so eval matches to a tolerance).
        vals[0].Should().BeApproximately(-39.457491396443004, 1e-5);
        vals[1].Should().BeApproximately(-233.17606833707896, 1e-4);
    }

    [TestMethod]
    public void EvalAcqfWithGrad_Gradient_MatchesFiniteDifference()
    {
        (_, LogEI acqf) = FitScenario();
        const double eps = 1e-7;
        foreach (double xv in new[] { 0.15, 0.3, 0.45, 0.7 })
        {
            (double _, double[] grad) = acqf.EvalAcqfWithGrad(new[] { xv });
            double fPlus = acqf.EvalAcqfNoGrad(new double[,] { { xv + eps } })[0];
            double fMinus = acqf.EvalAcqfNoGrad(new double[,] { { xv - eps } })[0];
            double fd = (fPlus - fMinus) / (2 * eps);
            // The acqf is steep here, so compare relative to the magnitude.
            grad[0].Should().BeApproximately(fd, 1e-4 * (1 + Math.Abs(fd)),
                $"analytic ∂LogEI/∂x at {xv} must match the central difference");
        }
    }

    [TestMethod]
    public void EvalAcqf_NegInfThreshold_IsZero()
    {
        (GPRegressor gpr, _) = FitScenario();
        var space = new System.Collections.Generic.Dictionary<string, BaseDistribution> { ["x"] = new FloatDistribution(0.0, 1.0) };
        var acqf = new LogEI(gpr, new GpSearchSpace(space), threshold: double.NegativeInfinity);
        acqf.EvalAcqfNoGrad(new double[,] { { 0.2 }, { 0.6 } }).Should().Equal(0.0, 0.0);
        (double val, double[] grad) = acqf.EvalAcqfWithGrad(new[] { 0.4 });
        val.Should().Be(0.0);
        grad.Should().Equal(0.0);
    }

    // ---- LogPI / LogCEI (Phase 3.2 part 3e-2 — constrained single-objective) -------------------

    [TestMethod]
    public void LogPI_BatchMatchesSinglePoint()
    {
        (GPRegressor gpr, LogEI ei) = FitScenario();
        var pi = new LogPI(gpr, ei.SearchSpace, threshold: 0.3);
        double[] batch = pi.EvalAcqfNoGrad(new double[,] { { 0.15 }, { 0.45 }, { 0.7 } });
        batch[0].Should().Be(pi.EvalAcqfNoGrad(new[] { 0.15 }));
        batch[1].Should().Be(pi.EvalAcqfNoGrad(new[] { 0.45 }));
        batch[2].Should().Be(pi.EvalAcqfNoGrad(new[] { 0.7 }));
    }

    [TestMethod]
    public void LogPI_Gradient_MatchesFiniteDifference()
    {
        (GPRegressor gpr, LogEI ei) = FitScenario();
        var pi = new LogPI(gpr, ei.SearchSpace, threshold: 0.3);
        const double eps = 1e-7;
        foreach (double xv in new[] { 0.15, 0.3, 0.45, 0.7 })
        {
            (double _, double[] grad) = pi.EvalAcqfWithGrad(new[] { xv });
            double fPlus = pi.EvalAcqfNoGrad(new double[,] { { xv + eps } })[0];
            double fMinus = pi.EvalAcqfNoGrad(new double[,] { { xv - eps } })[0];
            double fd = (fPlus - fMinus) / (2 * eps);
            grad[0].Should().BeApproximately(fd, 1e-4 * (1 + Math.Abs(fd)),
                $"analytic ∂LogPI/∂x at {xv} must match the central difference");
        }
    }

    [TestMethod]
    public void LogCEI_EqualsLogEiPlusSumLogPi()
    {
        // LogCEI(x) = LogEI(x) + Σ LogPI_c(x) — the composition must be exact (same float ops).
        (GPRegressor gpr, LogEI ei) = FitScenario();
        GpSearchSpace space = ei.SearchSpace;
        var pi = new LogPI(gpr, space, threshold: 0.2);
        var cei = new LogCEI(gpr, space, threshold: 1.418560466127369, new[] { gpr }, new[] { 0.2 });
        foreach (double xv in new[] { 0.15, 0.45, 0.7 })
        {
            cei.EvalAcqfNoGrad(new[] { xv }).Should()
                .Be(ei.EvalAcqfNoGrad(new[] { xv }) + pi.EvalAcqfNoGrad(new[] { xv }));

            (double vEi, double[] gEi) = ei.EvalAcqfWithGrad(new[] { xv });
            (double vPi, double[] gPi) = pi.EvalAcqfWithGrad(new[] { xv });
            (double vCei, double[] gCei) = cei.EvalAcqfWithGrad(new[] { xv });
            vCei.Should().Be(vEi + vPi);
            gCei[0].Should().Be(gEi[0] + gPi[0]);
        }
    }

    [TestMethod]
    public void LogCEI_NegInfThreshold_IgnoresObjectiveTerm()
    {
        // No feasible trial ⇒ objective threshold −∞ ⇒ the LogEI term is 0, leaving only constraints.
        (GPRegressor gpr, LogEI ei) = FitScenario();
        GpSearchSpace space = ei.SearchSpace;
        var pi = new LogPI(gpr, space, threshold: 0.2);
        var cei = new LogCEI(gpr, space, threshold: double.NegativeInfinity, new[] { gpr }, new[] { 0.2 });
        cei.EvalAcqfNoGrad(new[] { 0.4 }).Should().Be(pi.EvalAcqfNoGrad(new[] { 0.4 }));
    }

    [TestMethod]
    public void LogCEI_EmptyConstraints_Throws()
    {
        (GPRegressor gpr, LogEI ei) = FitScenario();
        var act = () => new LogCEI(gpr, ei.SearchSpace, 0.0,
            System.Array.Empty<GPRegressor>(), System.Array.Empty<double>());
        act.Should().Throw<System.ArgumentException>();
    }

    // ---- LogEHVI (Phase 3.2 part 3e-4 — multi-objective) ---------------------------------------

    private static (LogEHVI Acqf, GpSearchSpace Space) FitEhviScenario()
    {
        var x = new double[,] { { 0.1 }, { 0.35 }, { 0.5 }, { 0.72 }, { 0.9 }, { 0.25 }, { 0.6 }, { 0.8 } };
        var y0 = new[] { 0.73, 1.26, 0.29, -1.14, -0.84, 1.41, -0.50, -1.21 };
        var y1 = new[] { -0.90, 0.40, 1.10, 0.20, -1.30, -0.10, 1.35, 0.55 };
        var gpr0 = GPRegressor.FitKernelParams(x, y0, new[] { false }, 1e-6, false, null, 1e-2);
        var gpr1 = GPRegressor.FitKernelParams(x, y1, new[] { false }, 1e-6, false, null, 1e-2);
        var space = new GpSearchSpace(new System.Collections.Generic.Dictionary<string, BaseDistribution>
            { ["x"] = new FloatDistribution(0.0, 1.0) });
        var yTrain = new double[8, 2];
        for (int i = 0; i < 8; i++) { yTrain[i, 0] = y0[i]; yTrain[i, 1] = y1[i]; }
        return (new LogEHVI(new[] { gpr0, gpr1 }, space, yTrain, nQmcSamples: 128, qmcSeed: 42), space);
    }

    [TestMethod]
    public void LogEHVI_BatchMatchesSinglePoint()
    {
        (LogEHVI acqf, _) = FitEhviScenario();
        double[] batch = acqf.EvalAcqfNoGrad(new double[,] { { 0.15 }, { 0.45 }, { 0.7 } });
        batch[0].Should().Be(acqf.EvalAcqfNoGrad(new[] { 0.15 }));
        batch[1].Should().Be(acqf.EvalAcqfNoGrad(new[] { 0.45 }));
        batch[2].Should().Be(acqf.EvalAcqfNoGrad(new[] { 0.7 }));
    }

    [TestMethod]
    public void LogEHVI_Gradient_MatchesFiniteDifference()
    {
        (LogEHVI acqf, _) = FitEhviScenario();
        const double eps = 1e-7;
        foreach (double xv in new[] { 0.15, 0.3, 0.45, 0.7, 0.85 })
        {
            (double _, double[] grad) = acqf.EvalAcqfWithGrad(new[] { xv });
            double fPlus = acqf.EvalAcqfNoGrad(new double[,] { { xv + eps } })[0];
            double fMinus = acqf.EvalAcqfNoGrad(new double[,] { { xv - eps } })[0];
            double fd = (fPlus - fMinus) / (2 * eps);
            grad[0].Should().BeApproximately(fd, 1e-4 * (1 + Math.Abs(fd)),
                $"analytic ∂LogEHVI/∂x at {xv} must match the central difference");
        }
    }

    [TestMethod]
    public void LogEHVI_ValueAndGrad_ForwardsAgree()
    {
        // The value returned by EvalAcqfWithGrad must equal EvalAcqfNoGrad at the same point.
        (LogEHVI acqf, _) = FitEhviScenario();
        foreach (double xv in new[] { 0.2, 0.55, 0.9 })
        {
            (double val, _) = acqf.EvalAcqfWithGrad(new[] { xv });
            val.Should().BeApproximately(acqf.EvalAcqfNoGrad(new[] { xv }), 1e-12);
        }
    }

    // ---- LogCEHVI (Phase 3.2 part 3e-5 — multi-objective constrained) --------------------------

    private static (GPRegressor[] Gprs, GpSearchSpace Space, double[,] YTrain, GPRegressor ConstraintGpr) FitCehviScenario()
    {
        var x = new double[,] { { 0.1 }, { 0.35 }, { 0.5 }, { 0.72 }, { 0.9 }, { 0.25 }, { 0.6 }, { 0.8 } };
        var y0 = new[] { 0.73, 1.26, 0.29, -1.14, -0.84, 1.41, -0.50, -1.21 };
        var y1 = new[] { -0.90, 0.40, 1.10, 0.20, -1.30, -0.10, 1.35, 0.55 };
        var yc = new[] { 0.30, -0.70, 0.90, -0.20, 1.10, -0.40, 0.15, -0.95 };
        var gpr0 = GPRegressor.FitKernelParams(x, y0, new[] { false }, 1e-6, false, null, 1e-2);
        var gpr1 = GPRegressor.FitKernelParams(x, y1, new[] { false }, 1e-6, false, null, 1e-2);
        var cgpr = GPRegressor.FitKernelParams(x, yc, new[] { false }, 1e-6, false, null, 1e-2);
        var space = new GpSearchSpace(new System.Collections.Generic.Dictionary<string, BaseDistribution>
            { ["x"] = new FloatDistribution(0.0, 1.0) });
        var yTrain = new double[8, 2];
        for (int i = 0; i < 8; i++) { yTrain[i, 0] = y0[i]; yTrain[i, 1] = y1[i]; }
        return (new[] { gpr0, gpr1 }, space, yTrain, cgpr);
    }

    [TestMethod]
    public void LogCEHVI_EqualsLogEhviPlusSumLogPi()
    {
        (GPRegressor[] gprs, GpSearchSpace space, double[,] yTrain, GPRegressor cgpr) = FitCehviScenario();
        var ehvi = new LogEHVI(gprs, space, yTrain, nQmcSamples: 128, qmcSeed: 42);
        var pi = new LogPI(cgpr, space, threshold: 0.1);
        var cehvi = new LogCEHVI(gprs, space, yTrain, 128, 42, new[] { cgpr }, new[] { 0.1 });
        foreach (double xv in new[] { 0.2, 0.55, 0.9 })
        {
            cehvi.EvalAcqfNoGrad(new[] { xv }).Should()
                .BeApproximately(ehvi.EvalAcqfNoGrad(new[] { xv }) + pi.EvalAcqfNoGrad(new[] { xv }), 1e-12);
        }
    }

    [TestMethod]
    public void LogCEHVI_NoFeasible_IsConstraintsOnly()
    {
        // Y_feasible = null (all infeasible) ⇒ the LogEHVI term is dropped, leaving only the constraints.
        (GPRegressor[] gprs, GpSearchSpace space, _, GPRegressor cgpr) = FitCehviScenario();
        var pi = new LogPI(cgpr, space, threshold: 0.1);
        var cehvi = new LogCEHVI(gprs, space, yFeasible: null, 128, 42, new[] { cgpr }, new[] { 0.1 });
        foreach (double xv in new[] { 0.3, 0.7 })
            cehvi.EvalAcqfNoGrad(new[] { xv }).Should().Be(pi.EvalAcqfNoGrad(new[] { xv }));
    }

    // ---- qLogEI + ConditionalGPRegressor (Phase 3.2 part 3e-6 — running trials) ----------------

    private static (QLogEI Acqf, GpSearchSpace Space) FitQLogEiScenario(double threshold = 0.5)
    {
        var x = new double[,] { { 0.1 }, { 0.35 }, { 0.5 }, { 0.72 }, { 0.9 }, { 0.25 }, { 0.6 }, { 0.8 } };
        var y = new[] { 0.73, 1.26, 0.29, -1.14, -0.84, 1.41, -0.50, -1.21 };
        var gpr = GPRegressor.FitKernelParams(x, y, new[] { false }, 1e-6, false, null, 1e-2);
        var space = new GpSearchSpace(new System.Collections.Generic.Dictionary<string, BaseDistribution>
            { ["x"] = new FloatDistribution(0.0, 1.0) });
        var xRunning = new double[,] { { 0.42 }, { 0.66 } };
        return (new QLogEI(gpr, space, threshold, nQmcSamples: 64, qmcSeed: 7, xRunning), space);
    }

    [TestMethod]
    public void QLogEI_BatchMatchesSinglePoint()
    {
        (QLogEI acqf, _) = FitQLogEiScenario();
        double[] batch = acqf.EvalAcqfNoGrad(new double[,] { { 0.15 }, { 0.45 }, { 0.7 } });
        batch[0].Should().Be(acqf.EvalAcqfNoGrad(new[] { 0.15 }));
        batch[1].Should().Be(acqf.EvalAcqfNoGrad(new[] { 0.45 }));
        batch[2].Should().Be(acqf.EvalAcqfNoGrad(new[] { 0.7 }));
    }

    [TestMethod]
    public void QLogEI_ValueAndGrad_ForwardsAgree()
    {
        (QLogEI acqf, _) = FitQLogEiScenario();
        foreach (double xv in new[] { 0.2, 0.55, 0.9 })
        {
            (double val, _) = acqf.EvalAcqfWithGrad(new[] { xv });
            val.Should().BeApproximately(acqf.EvalAcqfNoGrad(new[] { xv }), 1e-12);
        }
    }

    [TestMethod]
    public void QLogEI_Gradient_MatchesFiniteDifference()
    {
        // Low threshold ⇒ the query column frequently wins the per-sample max ⇒ non-trivial gradient.
        (QLogEI acqf, _) = FitQLogEiScenario(threshold: -1.5);
        const double eps = 1e-7;
        foreach (double xv in new[] { 0.15, 0.3, 0.45, 0.7, 0.85 })
        {
            (double _, double[] grad) = acqf.EvalAcqfWithGrad(new[] { xv });
            double fPlus = acqf.EvalAcqfNoGrad(new[] { xv + eps });
            double fMinus = acqf.EvalAcqfNoGrad(new[] { xv - eps });
            double fd = (fPlus - fMinus) / (2 * eps);
            grad[0].Should().BeApproximately(fd, 1e-4 * (1 + Math.Abs(fd)),
                $"analytic ∂qLogEI/∂x at {xv} must match the central difference");
        }
    }

    [TestMethod]
    public void QLogEI_NegInfThreshold_IsZero()
    {
        // No feasible trial ⇒ threshold −∞ ⇒ qLogEI is identically 0 (matching Optuna's all-zeros utility).
        (QLogEI acqf, _) = FitQLogEiScenario(threshold: double.NegativeInfinity);
        acqf.EvalAcqfNoGrad(new[] { 0.4 }).Should().Be(0.0);
        (double val, double[] grad) = acqf.EvalAcqfWithGrad(new[] { 0.4 });
        val.Should().Be(0.0);
        grad[0].Should().Be(0.0);
    }

    [TestMethod]
    public void QLogEI_IsDeterministic()
    {
        // Fixed QMC seed ⇒ the acqf is a deterministic function (re-constructing gives identical values).
        (QLogEI a1, _) = FitQLogEiScenario();
        (QLogEI a2, _) = FitQLogEiScenario();
        a1.EvalAcqfNoGrad(new[] { 0.33 }).Should().Be(a2.EvalAcqfNoGrad(new[] { 0.33 }));
    }

    // ---- qLogEHVI (Phase 3.2 part 3e-6 — multi-objective with running trials) ------------------

    private static QLogEHVI FitQLogEhviScenario()
    {
        var x = new double[,] { { 0.1 }, { 0.35 }, { 0.5 }, { 0.72 }, { 0.9 }, { 0.25 }, { 0.6 }, { 0.8 } };
        var y0 = new[] { 0.73, 1.26, 0.29, -1.14, -0.84, 1.41, -0.50, -1.21 };
        var y1 = new[] { -0.90, 0.40, 1.10, 0.20, -1.30, -0.10, 1.35, 0.55 };
        var gpr0 = GPRegressor.FitKernelParams(x, y0, new[] { false }, 1e-6, false, null, 1e-2);
        var gpr1 = GPRegressor.FitKernelParams(x, y1, new[] { false }, 1e-6, false, null, 1e-2);
        var space = new GpSearchSpace(new System.Collections.Generic.Dictionary<string, BaseDistribution>
            { ["x"] = new FloatDistribution(0.0, 1.0) });
        var yTrain = new double[8, 2];
        for (int i = 0; i < 8; i++) { yTrain[i, 0] = y0[i]; yTrain[i, 1] = y1[i]; }
        var xRunning = new double[,] { { 0.42 }, { 0.66 } };
        return new QLogEHVI(new[] { gpr0, gpr1 }, space, yTrain, nQmcSamples: 64, qmcSeed: 11, xRunning);
    }

    [TestMethod]
    public void QLogEHVI_BatchMatchesSinglePoint()
    {
        QLogEHVI acqf = FitQLogEhviScenario();
        double[] batch = acqf.EvalAcqfNoGrad(new double[,] { { 0.15 }, { 0.45 }, { 0.7 } });
        batch[0].Should().Be(acqf.EvalAcqfNoGrad(new[] { 0.15 }));
        batch[1].Should().Be(acqf.EvalAcqfNoGrad(new[] { 0.45 }));
        batch[2].Should().Be(acqf.EvalAcqfNoGrad(new[] { 0.7 }));
    }

    [TestMethod]
    public void QLogEHVI_ValueAndGrad_ForwardsAgree()
    {
        QLogEHVI acqf = FitQLogEhviScenario();
        foreach (double xv in new[] { 0.2, 0.55, 0.9 })
        {
            (double val, _) = acqf.EvalAcqfWithGrad(new[] { xv });
            val.Should().BeApproximately(acqf.EvalAcqfNoGrad(new[] { xv }), 1e-12);
        }
    }

    [TestMethod]
    public void QLogEHVI_Gradient_MatchesFiniteDifference()
    {
        QLogEHVI acqf = FitQLogEhviScenario();
        const double eps = 1e-7;
        foreach (double xv in new[] { 0.15, 0.3, 0.45, 0.7, 0.85 })
        {
            (double _, double[] grad) = acqf.EvalAcqfWithGrad(new[] { xv });
            double fPlus = acqf.EvalAcqfNoGrad(new[] { xv + eps });
            double fMinus = acqf.EvalAcqfNoGrad(new[] { xv - eps });
            double fd = (fPlus - fMinus) / (2 * eps);
            grad[0].Should().BeApproximately(fd, 1e-4 * (1 + Math.Abs(fd)),
                $"analytic ∂qLogEHVI/∂x at {xv} must match the central difference");
        }
    }

    // ---- qLogCEHVI (Optuna 5.0.0, #6804 — multi-objective constrained with running trials) ----------

    /// <summary>
    ///     The 1-D two-objective scenario of <see cref="FitCehviScenario" /> plus a second constraint GP and
    ///     two running trials. <paramref name="allInfeasible" /> selects the <c>Y_feasible = None</c> branch.
    /// </summary>
    /// <param name="allInfeasible">Whether to pass no feasible observations (the marginal-LogPI branch).</param>
    /// <returns>The acquisition function plus the pieces the identity tests rebuild independently.</returns>
    private static (QLogCEHVI Acqf, GPRegressor[] ConstraintGprs, double[] Thresholds, GpSearchSpace Space) FitQLogCehviScenario(
        bool allInfeasible = false)
    {
        (GPRegressor[] gprs, GpSearchSpace space, double[,] yTrain, GPRegressor cgpr0) = FitCehviScenario();
        var x = new double[,] { { 0.1 }, { 0.35 }, { 0.5 }, { 0.72 }, { 0.9 }, { 0.25 }, { 0.6 }, { 0.8 } };
        var yc1 = new[] { -0.20, 0.55, 0.10, -0.65, 0.40, 0.95, -0.30, 0.05 };
        var cgpr1 = GPRegressor.FitKernelParams(x, yc1, new[] { false }, 1e-6, false, null, 1e-2);
        var constraintGprs = new[] { cgpr0, cgpr1 };
        var thresholds = new[] { 0.1, -0.2 };

        // A strict subset of the trials stands in for the feasible rows (any rows are a valid baseline).
        double[,]? yFeasible = null;
        if (!allInfeasible)
        {
            int[] feasibleRows = { 0, 2, 4, 6 };
            yFeasible = new double[feasibleRows.Length, 2];
            for (int i = 0; i < feasibleRows.Length; i++)
                for (int o = 0; o < 2; o++)
                    yFeasible[i, o] = yTrain[feasibleRows[i], o];
        }
        var xRunning = new double[,] { { 0.42 }, { 0.66 } };
        var acqf = new QLogCEHVI(
            gprs, space, yFeasible, nQmcSamples: 64, qmcSeed: 13, constraintGprs, thresholds, xRunning);
        return (acqf, constraintGprs, thresholds, space);
    }

    /// <summary>
    ///     The batched <c>eval_acqf_no_grad</c> must equal the single-point evaluation row by row, bit for bit:
    ///     the optimizer screens candidates in batches and refines them one at a time, so any drift between
    ///     the two paths would make it rank a point differently from how it then scores it.
    /// </summary>
    [TestMethod]
    public void QLogCEHVI_BatchMatchesSinglePoint()
    {
        (QLogCEHVI acqf, _, _, _) = FitQLogCehviScenario();
        double[] batch = acqf.EvalAcqfNoGrad(new double[,] { { 0.15 }, { 0.45 }, { 0.7 } });
        batch[0].Should().Be(acqf.EvalAcqfNoGrad(new[] { 0.15 }));
        batch[1].Should().Be(acqf.EvalAcqfNoGrad(new[] { 0.45 }));
        batch[2].Should().Be(acqf.EvalAcqfNoGrad(new[] { 0.7 }));
    }

    /// <summary>
    ///     The value returned with the gradient must match the no-gradient value in both regimes, so the
    ///     L-BFGS-B ascent and the screening pass optimize the same function.
    /// </summary>
    /// <param name="allInfeasible">Whether to exercise the marginal-LogPI branch (<c>Y_feasible = None</c>) instead of the feasible-baseline branch.</param>
    [DataTestMethod]
    [DataRow(false)]
    [DataRow(true)]
    public void QLogCEHVI_ValueAndGrad_ForwardsAgree(bool allInfeasible)
    {
        (QLogCEHVI acqf, _, _, _) = FitQLogCehviScenario(allInfeasible);
        foreach (double xv in new[] { 0.2, 0.55, 0.9 })
        {
            (double val, _) = acqf.EvalAcqfWithGrad(new[] { xv });
            val.Should().BeApproximately(acqf.EvalAcqfNoGrad(new[] { xv }), 1e-12);
        }
    }

    /// <summary>
    ///     The hand-derived gradient (softmax over draws of the constraint log-sigmoid and log-HVI
    ///     derivatives) must agree with a central finite difference — the independent check that replaces
    ///     autograd for this port.
    /// </summary>
    /// <param name="allInfeasible">Whether to exercise the marginal-LogPI branch (<c>Y_feasible = None</c>) instead of the feasible-baseline branch.</param>
    [DataTestMethod]
    [DataRow(false)]
    [DataRow(true)]
    public void QLogCEHVI_Gradient_MatchesFiniteDifference(bool allInfeasible)
    {
        (QLogCEHVI acqf, _, _, _) = FitQLogCehviScenario(allInfeasible);
        const double eps = 1e-7;
        foreach (double xv in new[] { 0.15, 0.3, 0.45, 0.7, 0.85 })
        {
            (double _, double[] grad) = acqf.EvalAcqfWithGrad(new[] { xv });
            double fPlus = acqf.EvalAcqfNoGrad(new[] { xv + eps });
            double fMinus = acqf.EvalAcqfNoGrad(new[] { xv - eps });
            double fd = (fPlus - fMinus) / (2 * eps);
            grad[0].Should().BeApproximately(fd, 1e-4 * (1 + Math.Abs(fd)),
                $"analytic ∂qLogCEHVI/∂x at {xv} (allInfeasible={allInfeasible}) must match the central difference");
        }
    }

    /// <summary>
    ///     With no feasible trial the acqf must be exactly the sum of plain <see cref="LogPI" /> terms built
    ///     WITHOUT running trials — Optuna's tower-property reduction; a regression here would mean the
    ///     running trials leak into (or mutate the GPs of) the all-infeasible branch.
    /// </summary>
    [TestMethod]
    public void QLogCEHVI_NoFeasible_IsMarginalLogPiSum()
    {
        // Y_feasible = None ⇒ the tower property leaves Σ_c LogPI_c(x) with NO running-trial conditioning:
        // fresh LogPI objects built without running trials must reproduce the value exactly.
        (QLogCEHVI acqf, GPRegressor[] constraintGprs, double[] thresholds, GpSearchSpace space) =
            FitQLogCehviScenario(allInfeasible: true);
        foreach (double xv in new[] { 0.3, 0.7 })
        {
            double expected = 0.0;
            for (int c = 0; c < constraintGprs.Length; c++)
                expected += new LogPI(constraintGprs[c], space, thresholds[c]).EvalAcqfNoGrad(new[] { xv });
            acqf.EvalAcqfNoGrad(new[] { xv }).Should().Be(expected);
        }
    }

    /// <summary>
    ///     Rebuilding the acqf from the same inputs must give bit-identical values: every QMC stream (objectives
    ///     and constraints) is seeded, which is what keeps the sampler reproducible and parity-testable.
    /// </summary>
    [TestMethod]
    public void QLogCEHVI_IsDeterministic()
    {
        // Every QMC stream is seeded, so rebuilding the acqf gives bit-identical values.
        (QLogCEHVI a1, _, _, _) = FitQLogCehviScenario();
        (QLogCEHVI a2, _, _, _) = FitQLogCehviScenario();
        a1.EvalAcqfNoGrad(new[] { 0.33 }).Should().Be(a2.EvalAcqfNoGrad(new[] { 0.33 }));
    }

    /// <summary>
    ///     Port of Optuna 5.0.0's <c>tests/gp_tests/test_acqf.py::test_eval_q_logcehvi</c>: fixed-hyperparameter
    ///     GPs from <c>get_gpr</c>, objectives <c>[1, 2, 3] ⊕ additional_values</c>, the same values as
    ///     constraints (threshold 0, feasible rows where every value is ≤ 0), one running trial at
    ///     <c>(0.4, 0.6)</c>, <c>stabilizing_noise = 0</c>; <c>verify_eval_acqf</c> asserts one finite value
    ///     per query row and a finite gradient — for an unbatched and a batched query.
    /// </summary>
    /// <param name="caseIndex">Which of the four <c>parametrized_additional_values</c> cases to use.</param>
    [DataTestMethod]
    [DataRow(0)]
    [DataRow(1)]
    [DataRow(2)]
    [DataRow(3)]
    public void QLogCEHVI_EvalAcqf_PortOfOptunaTest(int caseIndex)
    {
        double[][,] additionalValuesCases =
        {
            new double[,] { { 0.2 }, { 0.3 }, { -0.4 } },
            new double[,] { { 0.2, 0.3, 0.4 }, { 0.2, 0.3, 0.4 }, { 0.2, 0.3, 0.4 } },
            new double[,] { { 0.2, 0.3, 0.4 }, { 0.2, 0.3, -0.4 }, { -0.2, -0.3, -0.4 } },
            new double[,] { { -0.2, -0.3, -0.4 }, { -0.2, -0.3, -0.4 }, { -0.2, -0.3, -0.4 } },
        };
        double[,] c = additionalValuesCases[caseIndex];
        int nRows = c.GetLength(0), nCols = c.GetLength(1);

        // Y = hstack([[1, 2, 3]ᵀ, additional_values]).
        int nObjectives = 1 + nCols;
        var y = new double[nRows, nObjectives];
        for (int i = 0; i < nRows; i++)
        {
            y[i, 0] = i + 1.0;
            for (int j = 0; j < nCols; j++)
                y[i, 1 + j] = c[i, j];
        }

        var objectiveGprs = new GPRegressor[nObjectives];
        for (int o = 0; o < nObjectives; o++)
            objectiveGprs[o] = GetGpr(Column(y, o));
        var constraintGprs = new GPRegressor[nCols];
        var thresholds = new double[nCols];
        for (int j = 0; j < nCols; j++)
            constraintGprs[j] = GetGpr(Column(c, j));

        // is_feasible = np.all(c <= 0, axis=1); Y_feasible = None when nothing is feasible.
        var feasible = new List<int>();
        for (int i = 0; i < nRows; i++)
        {
            bool all = true;
            for (int j = 0; j < nCols; j++) all &= c[i, j] <= 0;
            if (all) feasible.Add(i);
        }
        double[,]? yFeasible = null;
        if (feasible.Count > 0)
        {
            yFeasible = new double[feasible.Count, nObjectives];
            for (int k = 0; k < feasible.Count; k++)
                for (int o = 0; o < nObjectives; o++)
                    yFeasible[k, o] = y[feasible[k], o];
        }

        var space = new GpSearchSpace(new System.Collections.Generic.Dictionary<string, BaseDistribution>
        {
            ["a"] = new FloatDistribution(0.0, 1.0),
            ["b"] = new FloatDistribution(0.0, 1.0),
        });
        var acqf = new QLogCEHVI(
            objectiveGprs, space, yFeasible, nQmcSamples: 32, qmcSeed: 42, constraintGprs, thresholds,
            normalizedParamsOfRunningTrials: new double[,] { { 0.4, 0.6 } }, stabilizingNoise: 0.0);

        // Unbatched x = [0.15, 0.12] and batched [[0.15, 0.12], [0.0, 1.0]].
        double[,] batch = { { 0.15, 0.12 }, { 0.0, 1.0 } };
        double[] values = acqf.EvalAcqfNoGrad(batch);
        values.Should().HaveCount(2);
        values.Should().OnlyContain(v => double.IsFinite(v));
        foreach (double[] row in new[] { new[] { 0.15, 0.12 }, new[] { 0.0, 1.0 } })
        {
            (double value, double[] grad) = acqf.EvalAcqfWithGrad(row);
            double.IsFinite(value).Should().BeTrue();
            grad.Should().HaveCount(2).And.OnlyContain(g => double.IsFinite(g));
        }
    }

    /// <summary>
    ///     <c>get_gpr</c> from Optuna's <c>test_acqf.py</c>: three fixed 2-D training points and fixed
    ///     hyperparameters (inverse squared lengthscales <c>[2, 3]</c>, kernel scale 4, noise 0.1), no fit.
    /// </summary>
    /// <param name="yTrain">The three training targets.</param>
    /// <returns>A cached regressor ready for posterior queries.</returns>
    private static GPRegressor GetGpr(double[] yTrain)
    {
        var gpr = new GPRegressor(
            new[] { false, false },
            new double[,] { { 0.1, 0.2 }, { 0.2, 0.3 }, { 0.3, 0.1 } },
            yTrain,
            new[] { 2.0, 3.0 },
            4.0,
            0.1);
        gpr.CacheMatrix();
        return gpr;
    }

    /// <summary>Copies one column of a matrix.</summary>
    /// <param name="m">The source matrix.</param>
    /// <param name="j">The column index.</param>
    /// <returns>A new array holding column <paramref name="j" /> of <paramref name="m" />.</returns>
    private static double[] Column(double[,] m, int j)
    {
        var col = new double[m.GetLength(0)];
        for (int i = 0; i < col.Length; i++)
            col[i] = m[i, j];
        return col;
    }
}
