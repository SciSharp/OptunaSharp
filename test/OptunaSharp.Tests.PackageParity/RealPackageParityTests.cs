using AwesomeAssertions;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.PackageParity;

/// <summary>
///     Proves the isolated, package-consuming assembly is really wired up before any oracle exists —
///     this runs with no Python and no generated oracle, so it is the "the shipped .nupkg restores and
///     runs" gate. Binding to the PACKAGED assembly (not a source build) is asserted via its strong-name
///     token.
/// </summary>
[TestClass]
public class PackageSmokeTests
{
    [TestMethod]
    public void PackagedOptunaSharp_Loads_Optimizes_AndIsTheSignedPackage()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(0));
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -10, 10);
            return x * x;
        }, nTrials: 5);
        study.Trials.Count.Should().Be(5, "the packaged OptunaSharp + NumSharp/OpenBLAS deps must run");

        // We must be bound to the shipped, strong-named assembly — not a source ProjectReference.
        var token = typeof(Optuna).Assembly.GetName().GetPublicKeyToken();
        token.Should().NotBeNull();
        Convert.ToHexString(token!).ToLowerInvariant()
            .Should().Be("cc7b13ffcd2ddd51", "the packaged assembly carries the shared SciSharp key");
    }
}

/// <summary>
///     Byte-for-byte / proximity parity of the PACKAGED OptunaSharp against real Optuna driving REAL
///     ML packages (xgboost, lightgbm, scikit-learn). gen_oracle.py runs the real study and records its
///     trajectory; here OptunaSharp replays it (Ask/Tell) and must reproduce the same suggestions.
///
///     <para>RandomSampler is bit-identical (value-independent), so the full xgboost / lightgbm spaces —
///     including their integer parameters — are asserted exactly. TPE is asserted to tolerance over a
///     continuous-only sklearn-SVM space (its integer-parameter path is excluded elsewhere for the
///     documented NumPy simd-sort tie-order divergence).</para>
///
///     <para>Self-skips (Inconclusive) when no oracle has been generated; CI sets PACKAGE_PARITY_REQUIRE
///     so a missing oracle goes red there instead.</para>
/// </summary>
[TestClass]
public class RealPackageParityTests
{
    [TestMethod]
    [TestCategory("RequiresOracle")]
    [DataRow("xgboost-random-seed0")]
    [DataRow("xgboost-random-seed42")]
    public void Xgboost_RandomSampler_BitIdentical(string oracle)
        => Replay.AssertParity(OracleStore.LoadOrInconclusive(oracle));

    [TestMethod]
    [TestCategory("RequiresOracle")]
    [DataRow("lightgbm-random-seed0")]
    public void LightGbm_RandomSampler_BitIdentical(string oracle)
        => Replay.AssertParity(OracleStore.LoadOrInconclusive(oracle));

    [TestMethod]
    [TestCategory("RequiresOracle")]
    [DataRow("svm-tpe-seed0")]
    [DataRow("svm-tpe-seed42")]
    public void SvmSklearn_Tpe_TrajectoryMatches(string oracle)
        => Replay.AssertParity(OracleStore.LoadOrInconclusive(oracle));
}
