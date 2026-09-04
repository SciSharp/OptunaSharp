using System;
using System.Globalization;
using AwesomeAssertions;
using OptunaSharp.Gp;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Byte-exact parity of the GP-internal <see cref="GpSearchSpace" /> transforms (PORTING_PLAN
///     Phase 3.2 part 3) against <c>optuna/_gp/search_space.py</c>. The normalize / unnormalize /
///     round transforms are pure numpy (no <c>erf</c>, no LAPACK — just affine, <c>log</c>/<c>exp</c>,
///     <c>floor</c>), so OptunaSharp reproduces them bit-for-bit.
/// </summary>
[TestClass]
public class GpSearchSpaceParityTests : OptunaInteropTestBase
{
    private const string Helper =
        "import numpy as np\n" +
        "import optuna._gp.search_space as _ss\n" +
        "def _norm(v, st, lo, hi, step):\n" +
        "    return float(_ss._normalize_one_param(np.array([v]), _ss._ScaleType(st), (lo, hi), step)[0])\n" +
        "def _unnorm(v, st, lo, hi, step):\n" +
        "    return float(_ss._unnormalize_one_param(np.array([v]), _ss._ScaleType(st), (lo, hi), step)[0])\n" +
        "def _round(v, st, lo, hi, step):\n" +
        "    return float(_ss._round_one_normalized_param(np.array([v]), _ss._ScaleType(st), (lo, hi), step)[0])\n";

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void OneParamTransforms_ByteExact_Sweep()
    {
        PythonSession.EnsureOrInconclusive();
        PyExec(Helper);
        var rng = new Random(4242);
        for (int trial = 0; trial < 60; trial++)
        {
            bool log = rng.NextDouble() < 0.4;
            var st = log ? GpSearchSpace.ScaleType.Log : GpSearchSpace.ScaleType.Linear;
            double lo, hi;
            if (log) { lo = Math.Pow(10, rng.NextDouble() * 3 - 4); hi = lo * (2 + rng.NextDouble() * 100); }
            else { lo = rng.NextDouble() * 20 - 10; hi = lo + 0.5 + rng.NextDouble() * 20; }
            // Optuna disallows a step on log-scale distributions (they stay continuous), so a log
            // dimension always has step 0; only linear dimensions get a discretization step.
            double step = !log && rng.NextDouble() < 0.4 ? (hi - lo) / (2 + rng.Next(8)) : 0.0;

            double normVal = rng.NextDouble();                 // a normalized value in [0,1]
            double extVal = lo + rng.NextDouble() * (hi - lo); // an external value in-range

            double normCs = GpSearchSpace.NormalizeOneParam(extVal, st, (lo, hi), step);
            double unnormCs = GpSearchSpace.UnnormalizeOneParam(normVal, st, (lo, hi), step);
            double roundCs = GpSearchSpace.RoundOneNormalizedParam(normVal, st, (lo, hi), step);

            string args = $"{Fmt(extVal)}, {(int)st}, {Fmt(lo)}, {Fmt(hi)}, {Fmt(step)}";
            string argsN = $"{Fmt(normVal)}, {(int)st}, {Fmt(lo)}, {Fmt(hi)}, {Fmt(step)}";
            PyExec(
                $"d_norm = abs(_norm({args}) - {Fmt(normCs)})\n" +
                $"d_unnorm = abs(_unnorm({argsN}) - {Fmt(unnormCs)})\n" +
                $"d_round = abs(_round({argsN}) - {Fmt(roundCs)})\n");
            PyFloat("d_norm").Should().Be(0.0, $"normalize trial={trial} log={log} step={step}");
            PyFloat("d_unnorm").Should().Be(0.0, $"unnormalize trial={trial}");
            PyFloat("d_round").Should().Be(0.0, $"round trial={trial}");
        }
    }

    private static string Fmt(double d) => d.ToString("R", CultureInfo.InvariantCulture);
}
