using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Integration.LightGbm;

namespace OptunaSharp.Tests.Integration.LightGbm;

/// <summary>
///     Open defects of the LightGBM integration found while writing the real-world examples
///     (<c>examples/0_LoanDefaultLightGbmTunerCv.cs</c>, 2026-11-08). Each test fails today and passes once
///     the defect is fixed, at which point its <c>[TestCategory("OpenBugs")]</c> should be removed.
/// </summary>
[TestClass]
public class LightGbmRealWorldBugTests
{
    /// <summary>
    ///     A LightGBM fit whose parameters carry no <c>seed</c> must still be reproducible, as native
    ///     LightGBM is: with <c>seed</c> unset it falls back to fixed default seeds (<c>bagging_seed=3</c>,
    ///     <c>feature_fraction_seed=2</c>, <c>data_random_seed=1</c>, …), so the same data and parameters give
    ///     the same model on every run.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         <see cref="MlNetLightGbmTrainer" /> forwards only an explicit <c>seed</c> to ML.NET's
    ///         <c>Options.Seed</c>; without one it leaves <c>Seed = null</c>, and ML.NET then draws the
    ///         LightGBM seed from its unseeded <c>MLContext</c>'s random source. Row bagging and feature
    ///         subsampling therefore change from call to call, and so do the CV curves.
    ///     </para>
    ///     <para>
    ///         Real-world impact: the parameter dictionary of Optuna's own example
    ///         (<c>optuna-examples/lightgbm/lightgbm_tuner_cv.py</c>: objective, metric, verbosity,
    ///         boosting_type — no seed) makes <c>LightGBMTunerCV</c> return a different "best" configuration
    ///         on every run despite a fixed <c>optuna_seed</c>; three runs of the loan-default example picked
    ///         <c>min_child_samples</c> 10, 50 and 5 and different <c>lambda_l1</c> values. In Python the
    ///         same tuning reproduces exactly.
    ///     </para>
    /// </remarks>
    [TestMethod]
    [TestCategory("OpenBugs")]
    public void Cv_WithoutSeedParam_IsReproducible_LikeNativeLightGbmDefaultSeeds()
    {
        // A small binary task with signal in two of five columns.
        var rng = new Random(7);
        int rows = 400;
        var x = new double[rows, 5];
        var y = new double[rows];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < 5; j++)
                x[i, j] = rng.NextDouble();
            y[i] = x[i, 0] + 0.5 * x[i, 1] + 0.3 * rng.NextDouble() > 0.95 ? 1 : 0;
        }

        // Bagging and feature subsampling are exactly the randomness LightGBM's default seeds pin down.
        var @params = new Dictionary<string, object?>
        {
            ["objective"] = "binary",
            ["metric"] = "binary_logloss",
            ["verbosity"] = -1L,
            ["bagging_fraction"] = 0.6,
            ["bagging_freq"] = 1L,
            ["feature_fraction"] = 0.6,
        };
        var trainer = new MlNetLightGbmTrainer();
        CvRequest Request() => new(@params, new Dataset(x, y), numBoostRound: 25, nFold: 3,
            stratified: true, shuffle: true, seed: 0, returnCvBooster: false);

        var first = trainer.Cv(Request()).Results["binary_logloss-mean"].ToArray();
        var second = trainer.Cv(Request()).Results["binary_logloss-mean"].ToArray();

        second.Should().Equal(first, "the same data and parameters must give the same CV curve, as native LightGBM's fixed default seeds do");
    }
}
