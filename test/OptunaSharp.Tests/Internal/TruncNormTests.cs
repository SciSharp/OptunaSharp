using AwesomeAssertions;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Internal;

/// <summary>
///     Checks for the <see cref="TruncNorm" /> log-Gaussian-mass machinery (the subset of Optuna's
///     <c>_tpe/_truncnorm.py</c> that PED-ANOVA rides on), asserted against values from live Optuna
///     5.0.0.dev / SciPy. The wide-interval cases stress the small-argument <c>log1p</c> path where
///     a naive Kahan formulation drifts 1 ULP (the correction-term form is bit-exact there).
///
///     <para>Matched only to a few ULP, not bit-for-bit: these values route through
///     <c>_ndtr</c> → <c>erf</c>, and Optuna's <c>math.erf</c> is the platform libm (ucrt), which no
///     managed <c>erf</c> reproduces exactly (see <see cref="TruncNorm" />).</para>
/// </summary>
[TestClass]
public class TruncNormTests
{
    // A few ULP of headroom for the erf(c) system-libm divergence; still ~1e5x tighter than any
    // real defect would produce.
    private const double ErfTolerance = 1e-11;

    [TestMethod]
    public void LogGaussMass_MatchesOptuna()
    {
        var a = new[] { -2.0, -0.5, 0.3, 1.0, -1.5 };
        var b = new[] { -1.0, 0.5, 1.3, 3.0, 2.0 };
        var expected = new[]
        {
            -1.9957982691807548, -0.9599163336956226, -1.2542557558591771,
            -1.849566420547608, -0.09382435073398993,
        };

        var result = TruncNorm.LogGaussMass(a, b);

        for (int i = 0; i < expected.Length; i++)
            result[i].Should().BeApproximately(expected[i], ErfTolerance);
    }

    [TestMethod]
    public void LogGaussMass_WideIntervals_SmallLog1pArgument()
    {
        // The central case with a wide interval drives -(ndtr(a) + ndtr(-b)) toward 0, i.e. the
        // small-argument log1p region (results here are ~1e-9 and ~1e-8).
        var a = new[] { -6.0, -5.6, -0.001 };
        var b = new[] { 6.0, 5.6, 0.001 };
        var expected = new[] { -1.9731752917733668e-09, -2.14351807491803e-08, -7.133546798293504 };

        var result = TruncNorm.LogGaussMass(a, b);

        for (int i = 0; i < expected.Length; i++)
            result[i].Should().BeApproximately(expected[i], ErfTolerance);
    }
}
