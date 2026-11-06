using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Open parity bugs in the distribution classes, found by the 2026-11-02 parity QA run
///     (<c>docs/qa/PARITY_QA_2026-11-02.md</c>, findings QA-1 to QA-10). Every expected value is what
///     Optuna 5.0.0 returns on CPython 3.12 for the same call; each test fails today and passes once the
///     divergence is fixed, at which point its <c>[OpenBugs]</c> attribute should be removed.
/// </summary>
/// <remarks>
///     Most of these come from one root cause: <see cref="FloatDistribution" /> ports
///     <c>_adjust_discrete_uniform_high</c> with <see cref="decimal" />, but Python's <c>decimal</c> module
///     differs from .NET's in three ways that matter here. <c>float(Decimal)</c> is correctly rounded while
///     .NET's <c>(double)decimal</c> divides a rounded mantissa by a rounded power of ten; Python's exact zero
///     sum is <c>+0</c> while .NET can return a negative-zero decimal; and Python's context has unbounded
///     exponents with 28-digit precision while .NET's decimal saturates below <c>1e-28</c> and above
///     <c>7.9e28</c>.
/// </remarks>
[TestClass]
public class DistributionParityBugTests
{
    /// <summary>
    ///     QA-1 (fixed 2026-11-06): a stepped float range whose adjusted upper end is zero must store <c>+0.0</c>,
    ///     as Python does; the port stored <c>-0.0</c>, which then flowed into every sampled parameter value.
    ///     <c>AdjustDiscreteUniformHigh</c> now normalizes a zero decimal sum to positive zero.
    /// </summary>
    /// <remarks>
    ///     Python: <c>FloatDistribution(-0.5, 0.2, step=0.25).high</c> is <c>0.0</c>: <c>Decimal('0.50') +
    ///     Decimal('-0.5')</c> is a positive zero, and <c>float()</c> keeps it. .NET's decimal sum of the same
    ///     operands is a negative zero, so <c>(double)</c> gives <c>-0.0</c>. Sign-of-zero only appears when the
    ///     operand scales differ (step <c>0.5</c> gives <c>+0.0</c>), which is why it slipped past the existing
    ///     step tests. Python prints <c>{'x': 0.0}</c> in the trial log where the port prints <c>-0.0</c>.
    /// </remarks>
    [TestMethod]
    public void SteppedHigh_AdjustedToZero_IsPositiveZero()
    {
        var dist = new FloatDistribution(-0.5, 0.2, step: 0.25);

        BitConverter.DoubleToInt64Bits(dist.High).Should().Be(0L, "Python stores +0.0, not -0.0");
        Distribution.ToJson(dist).Should().Be(
            "{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": 0.25, \"low\": -0.5, \"high\": 0.0, \"log\": false}}");
    }

    /// <summary>
    ///     QA-1 (sampling side, fixed 2026-11-06): every sampler's top grid point for <c>suggest_float("x", -0.5,
    ///     0.2, step=0.25)</c> is <c>+0.0</c> in Python and was <c>-0.0</c> in the port, so the objective, the stored
    ///     parameter and <c>best_params</c> all carried the wrong zero sign.
    /// </summary>
    /// <remarks>
    ///     Measured with seed 1 over 40 trials: Random, TPE (independent and multivariate), QMC, CMA-ES and
    ///     NSGA-II return zero 12–23 times each; all of Python's zeros are positive and all of the port's are
    ///     negative. Seed 1, Random: Python's <c>study.best_params</c> is <c>{'x': 0.0, ...}</c>.
    /// </remarks>
    [TestMethod]
    public void SteppedHigh_AdjustedToZero_SamplersReturnPositiveZero()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 1));
        var zeros = new List<double>();
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -0.5, 0.2, step: 0.25);
            double y = t.SuggestFloat("y", -1.0, 1.0);
            if (x == 0.0)
                zeros.Add(x);
            return (x - 0.1) * (x - 0.1) + y * y;
        }, nTrials: 40);

        zeros.Should().HaveCount(12, "seed 1 reaches the top grid point 12 times in Python");
        zeros.Should().OnlyContain(z => !double.IsNegative(z), "Python returns +0.0 for every one of them");
        double.IsNegative((double)study.BestParams["x"]!).Should().BeFalse();
    }

    /// <summary>
    ///     QA-2 (fixed 2026-11-06): the adjusted upper end of a stepped float range must be Python's correctly
    ///     rounded <c>float(Decimal)</c>; the port's <c>(double)decimal</c> was one ULP off for about 8.6% of
    ///     realistic ranges whose bounds are not short decimals. It now converts through
    ///     <c>PyNumbers.DecimalToDouble</c>.
    /// </summary>
    /// <remarks>
    ///     A 20,000-case fuzz (random lows and highs, common steps) differs in 1,726 cases by one ULP. The
    ///     cases below are taken from it, with Python's results. The off-by-one-ULP high then changes where
    ///     the storages' JSON round trip lands (Optuna re-adjusts on every read), so a persistent study can
    ///     lose a whole step of its range in C# where it does not in Python, or the other way round.
    /// </remarks>
    [TestMethod]
    public void SteppedHigh_Adjustment_IsCorrectlyRounded()
    {
        var cases = new (double Low, double High, double Step, double PythonHigh)[]
        {
            (-1.5096162171497198, 15.027426276291042, 0.01, 15.02038378285028),
            (5.781883429807097, 22.148950277153446, 0.2, 21.981883429807098),
            (0.6918192460020727, 12.89, 0.05, 12.841819246002073),
            (-0.15, 9.44, 0.3333333333333333, 9.183333333333332),
            (0.0, 1e20, 7e-05, 1e20),
        };
        foreach (var (low, high, step, pythonHigh) in cases)
        {
            var dist = new FloatDistribution(low, high, step: step);
            PythonRepr.FloatRepr(dist.High).Should().Be(PythonRepr.FloatRepr(pythonHigh),
                $"Python's FloatDistribution({PythonRepr.FloatRepr(low)}, {PythonRepr.FloatRepr(high)}, step={PythonRepr.FloatRepr(step)}).high");
        }
    }

    /// <summary>
    ///     QA-2 (cross-language read, fixed 2026-11-06): C# must read a distribution that Python wrote to storage
    ///     exactly as Python reads it back; it differed in about 7% of stepped ranges with non-short bounds.
    /// </summary>
    /// <remarks>
    ///     Optuna's <c>json_to_distribution</c> re-runs the step adjustment on every read, and so does
    ///     <see cref="Distribution.FromJson" />. For this JSON (written by Python 5.0.0) Python reads
    ///     <c>high=21.981883429807098</c> and the port reads <c>21.981883429807095</c>. In a mixed C#/Python
    ///     fleet the two languages then hold different distributions for the same parameter, and
    ///     <c>IntersectionSearchSpace</c> (which compares distributions with <c>==</c>) drops it.
    /// </remarks>
    [TestMethod]
    public void FromJson_PythonWrittenSteppedDistribution_ReadsLikePython()
    {
        const string pythonWritten =
            "{\"name\": \"FloatDistribution\", \"attributes\": {\"step\": 0.2, \"low\": 5.781883429807097, \"high\": 21.981883429807098, \"log\": false}}";

        var dist = (FloatDistribution)Distribution.FromJson(pythonWritten);

        PythonRepr.FloatRepr(dist.High).Should().Be("21.981883429807098");
        Distribution.ToJson(dist).Should().Be(pythonWritten, "Python's json_to_distribution round trip keeps this one unchanged");
    }

    /// <summary>
    ///     QA-3: magnitudes outside .NET decimal's range must follow Python's unbounded-exponent decimal; the
    ///     port silently treats values below <c>1e-28</c> as zero, which can even leave <c>high &lt; low</c>.
    /// </summary>
    /// <remarks>
    ///     Python: <c>FloatDistribution(0.0, 5e-324, step=0.1).high</c> is <c>0.0</c> (with the "not divisible"
    ///     warning) and <c>FloatDistribution(5e-324, 0.30000000000000004, step=1.0).high</c> is <c>5e-324</c>.
    ///     The port keeps <c>5e-324</c> in the first case and stores <c>high = 0.0 &lt; low</c> in the second,
    ///     an invalid distribution that the constructor's own <c>low &lt;= high</c> check forbids.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void SteppedHigh_TinyMagnitudes_FollowPythonDecimal()
    {
        new FloatDistribution(0.0, 5e-324, step: 0.1).High.Should().Be(0.0);
        var inverted = new FloatDistribution(5e-324, 0.30000000000000004, step: 1.0);
        inverted.High.Should().BeGreaterThanOrEqualTo(inverted.Low, "a distribution must never end below its start");
        inverted.High.Should().Be(5e-324);
    }

    /// <summary>
    ///     QA-3 (Python raises): a stepped range whose quotient needs more than 28 digits makes Python's
    ///     decimal raise <c>InvalidOperation</c> (<c>DivisionImpossible</c>); the port builds the distribution.
    /// </summary>
    /// <remarks>
    ///     Python: <c>FloatDistribution(0.0, 1e20, step=1e-09)</c> raises
    ///     <c>decimal.InvalidOperation: [&lt;class 'decimal.DivisionImpossible'&gt;]</c>; NaN or infinite steps
    ///     and bounds give <c>high=nan</c> or <c>InvalidOperation</c> where the port throws
    ///     <see cref="FormatException" />. Whether the port should copy this Python limitation is an open
    ///     question for the owner (see the QA journal); the test pins Optuna's observable behavior.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void SteppedHigh_QuotientBeyondDecimalPrecision_RaisesLikePython()
    {
        var act = () => new FloatDistribution(0.0, 1e20, step: 1e-09);
        act.Should().Throw<Exception>("Python's decimal context (prec=28) raises DivisionImpossible here");
    }

    /// <summary>
    ///     QA-4: the stepped int adjustment must compute <c>high - low</c> without overflow, as Python's
    ///     unbounded ints do; the port's 64-bit subtraction wraps and collapses the range to one value.
    /// </summary>
    /// <remarks>
    ///     Python: <c>IntDistribution(-2**63, 2**63 - 1, step=2).high</c> is <c>9223372036854775806</c>. The
    ///     port computes <c>r = -1</c>, adjusts to <c>high = -9223372036854775808</c> and warns "It will be
    ///     replaced with [-9223372036854775808, -9223372036854775808]", so <c>Single()</c> becomes true. With
    ///     <c>(-2**62 - 1, 2**62 + 1, step=3)</c> Python adjusts to <c>4611686018427387904</c> and the port
    ///     does not adjust at all. Related to the known "span of 2^63 or more" sampling issue (catalog item
    ///     19c), but this one corrupts the distribution itself.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void SteppedIntHigh_FullInt64Range_DoesNotOverflow()
    {
        new IntDistribution(long.MinValue, long.MaxValue, step: 2).High.Should().Be(9223372036854775806L);
        new IntDistribution(-(1L << 62) - 1, (1L << 62) + 1, step: 3).High.Should().Be(4611686018427387904L);
    }

    /// <summary>
    ///     QA-27: mapping a transformed coordinate back to an int value must not overflow 64 bits for ranges
    ///     whose span is 2^63 or more, as Python's unbounded ints do not. Latent behind the known catalog item
    ///     19c (<c>Single()</c> overflows, so such a parameter never reaches a sampler): fixing 19c alone would
    ///     expose this one.
    /// </summary>
    /// <remarks>
    ///     <c>_untransform_numerical_param</c> computes <c>step_index = floor((t - low) / step + 0.5)</c> and
    ///     <c>step_index * step + low</c> in Python ints. The port converts the floor to <c>long</c>, which
    ///     saturates at <c>long.MaxValue</c> on net10.0 (and gives <c>long.MinValue</c> on net8.0), then adds
    ///     <c>low</c> with wrap-around. Over <c>[-2**63, 2**63 - 1]</c> every coordinate <c>t &gt;= 0</c> becomes
    ///     <c>-1</c>: Optuna maps <c>9.004502142883758e17</c> (RandomSampler seed 0's first draw) to
    ///     <c>900450214288375808</c>. At the top bound of <c>[-2**62, 2**62]</c> the port returns
    ///     <c>4611686018427387903</c>, and of <c>[-10**18, 2**63 - 1]</c> it returns <c>8223372036854775807</c>.
    ///     The 2026-11-02 transform fuzz found 240 of 376 such spaces affected and none of the 3,546 others.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void IntUntransform_SpanOf2Pow63OrMore_DoesNotOverflow()
    {
        static object? Untransform(long low, long high, double coordinate)
        {
            var space = new Collections.OrderedDict<string, BaseDistribution> { ["x"] = new IntDistribution(low, high) };
            return new SearchSpaceTransform(space).Untransform(np.array(new[] { coordinate }))["x"];
        }

        Untransform(long.MinValue, long.MaxValue, 9.004502142883758e17).Should().Be(900450214288375808L);
        Untransform(-(1L << 62), 1L << 62, 4.611686018427388e18).Should().Be(4611686018427387904L);
        Untransform(-1000000000000000000L, long.MaxValue, 9.223372036854776e18).Should().Be(long.MaxValue);
        Untransform(long.MinValue, 1000000000000000000L, 5e17).Should().Be(500000000000000000L);
    }

    /// <summary>
    ///     QA-30: clipping a sampled value to a bound must keep the value's own sign of zero, as
    ///     <c>np.clip</c> does; <c>MathUtil.Clip</c> is <c>Math.Min(Math.Max(x, low), high)</c>, whose IEEE
    ///     minimum and maximum order <c>-0.0</c> below <c>+0.0</c>.
    /// </summary>
    /// <remarks>
    ///     <c>np.clip(x, lo, hi)</c> returns <c>x</c> unless it lies strictly outside, so
    ///     <c>np.clip(0.0, -1.0, -0.0)</c> is <c>+0.0</c> and <c>np.clip(-0.0, 0.0, 1.0)</c> is <c>-0.0</c>; the port
    ///     returns <c>-0.0</c> and <c>+0.0</c>. Public path: a stepped float whose upper bound is <c>-0.0</c> (for
    ///     example <c>-min_shift</c> with <c>min_shift = 0.0</c>). <c>_untransform_numerical_param</c> computes the top
    ///     grid point as <c>4 * 0.25 + (-1.0) = +0.0</c> and clips it to <c>high = -0.0</c>: Optuna keeps
    ///     <c>0.0</c> (RandomSampler seed 1: trials 13, 20, 21, 24, 25 and 29), the port stores <c>-0.0</c>. Random,
    ///     QMC, CMA-ES and GP all go through this clip; TPE's discrete sampling and NSGA-II's mutation use the
    ///     same helper. Unlike QA-1 the user has to pass a negative-zero bound, so it is rarer.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void SteppedFloat_NegativeZeroHigh_TopGridPointIsPositiveZero()
    {
        BitConverter.DoubleToInt64Bits(MathUtil.Clip(0.0, -1.0, -0.0)).Should().Be(0L, "np.clip keeps +0.0");
        BitConverter.DoubleToInt64Bits(MathUtil.Clip(-0.0, 0.0, 1.0)).Should().Be(long.MinValue, "np.clip keeps -0.0");

        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 1));
        study.Optimize(t => t.SuggestFloat("x", -1.0, -0.0, step: 0.25), nTrials: 30);

        var zeros = study.Trials.Where(t => (double)t.Params["x"]! == 0.0).ToList();
        zeros.Select(t => t.Number).Should().Equal(13, 20, 21, 24, 25, 29);
        zeros.Should().OnlyContain(t => !double.IsNegative((double)t.Params["x"]!));
    }

    /// <summary>
    ///     QA-5: converting a non-finite internal value of an int distribution must fail as Python's
    ///     <c>int(float)</c> does; the port saturates to <c>long.MaxValue</c> or returns <c>0</c> for NaN.
    /// </summary>
    /// <remarks>
    ///     Python: <c>IntDistribution(0, 10).to_external_repr(float('inf'))</c> raises <c>OverflowError: cannot
    ///     convert float infinity to integer</c>, NaN raises <c>ValueError: cannot convert float NaN to
    ///     integer</c>. It is reachable from an enqueued or fixed <c>inf</c> for an int parameter, where
    ///     Python's <c>suggest_int</c> raises instead of returning a huge integer.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void IntToExternalRepr_NonFinite_RaisesLikePython()
    {
        var dist = new IntDistribution(0, 10);
        ((Action)(() => dist.ToExternalRepr(double.PositiveInfinity))).Should().Throw<OverflowException>();
        ((Action)(() => dist.ToExternalRepr(double.NaN))).Should().Throw<ValueError>()
            .WithMessage("cannot convert float NaN to integer");
    }

    /// <summary>
    ///     QA-5 (public path): an enqueued infinite value for an int parameter must make <c>suggest_int</c>
    ///     raise, as Python's <c>int(float('inf'))</c> does; the port returns <c>long.MaxValue</c> and stores it.
    /// </summary>
    /// <remarks>
    ///     Python 5.0.0: <c>study.enqueue_trial({"i": float("inf")})</c> then <c>trial.suggest_int("i", 0,
    ///     10)</c> raises <c>OverflowError: cannot convert float infinity to integer</c> (a warning first reports
    ///     the value as out of range). The port warns the same way, then returns and stores
    ///     <c>9223372036854775807</c>; <c>-inf</c> gives <c>long.MinValue</c>, and <c>1e30</c> (an exact Python
    ///     int) is clamped to <c>long.MaxValue</c>.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void SuggestInt_EnqueuedInfinity_RaisesLikePython()
    {
        var study = Optuna.CreateStudy();
        study.EnqueueTrial(new Collections.OrderedDict<string, object?> { ["i"] = double.PositiveInfinity });
        var trial = study.Ask();

        ((Action)(() => trial.SuggestInt("i", 0, 10))).Should().Throw<OverflowException>();
    }

    /// <summary>
    ///     QA-6: the numeric-string conversion behind <c>to_internal_repr</c> must accept what Python's
    ///     <c>float()</c> accepts, including digit-group underscores.
    /// </summary>
    /// <remarks>
    ///     Python: <c>FloatDistribution(0, 10).to_internal_repr('1_000')</c> returns <c>1000.0</c> (PEP 515
    ///     underscores are valid in <c>float()</c>); the port raises "'1_000' is not a valid type.
    ///     float-castable value is expected.". Same for <see cref="IntDistribution" />.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void ToInternalRepr_UnderscoreDigitString_ParsesLikePythonFloat()
    {
        new FloatDistribution(0, 10).ToInternalRepr("1_000").Should().Be(1000.0);
        new IntDistribution(0, 10).ToInternalRepr("1_000").Should().Be(1000.0);
    }

    /// <summary>
    ///     QA-7: categorical lookups must compare ints and floats exactly, as Python does; the port compares
    ///     through <see cref="double" />, so integers above 2^53 match the wrong choice.
    /// </summary>
    /// <remarks>
    ///     Python: with choices <c>(2**53, float(2**53))</c>, <c>to_internal_repr(2**53 + 1)</c> raises
    ///     <c>ValueError: '9007199254740993' not in (9007199254740992, 9007199254740992.0).</c>; with choices
    ///     <c>(2**53 + 1, float(2**53))</c>, <c>to_internal_repr(float(2**53))</c> is <c>1</c>. The port returns
    ///     <c>1</c> and <c>0</c>, because <c>(double)(2^53 + 1) == 2^53</c>.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void CategoricalLookup_BigIntVersusFloat_ComparesExactly()
    {
        var a = new CategoricalDistribution((CategoricalChoice)9007199254740992L, (CategoricalChoice)9007199254740992.0);
        ((Action)(() => a.ToInternalRepr(9007199254740993L))).Should().Throw<ValueError>()
            .WithMessage("'9007199254740993' not in (9007199254740992, 9007199254740992.0).");

        var b = new CategoricalDistribution((CategoricalChoice)9007199254740993L, (CategoricalChoice)9007199254740992.0);
        b.ToInternalRepr(9007199254740992.0).Should().Be(1.0);
    }

    /// <summary>
    ///     QA-8: <see cref="Distribution.FromJson" /> must validate the raw JSON values the way
    ///     <c>json_to_distribution</c> passes them to the constructors.
    /// </summary>
    /// <remarks>
    ///     Python checks <c>low &lt;= high</c> before it truncates int bounds, so
    ///     <c>{"low": 2.5, "high": 2.2}</c> raises <c>ValueError: `low &lt;= high` must hold, but got (low=2.5,
    ///     high=2.2).</c> where the port truncates both to 2 and accepts. Python treats <c>"log": 1</c> as true
    ///     (the port throws <see cref="InvalidCastException" />) and rejects unknown attributes with
    ///     <c>TypeError: FloatDistribution.__init__() got an unexpected keyword argument 'foo'</c> (the port
    ///     ignores them). Reachable from hand-written <c>--search-space</c> JSON in the CLI.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void FromJson_RawAttributeValues_ValidateLikePython()
    {
        ((Action)(() => Distribution.FromJson(
                "{\"name\": \"IntDistribution\", \"attributes\": {\"log\": false, \"step\": 1, \"low\": 2.5, \"high\": 2.2}}")))
            .Should().Throw<ValueError>().WithMessage("`low <= high` must hold, but got (low=2.5, high=2.2).");

        var logOne = (IntDistribution)Distribution.FromJson(
            "{\"name\": \"IntDistribution\", \"attributes\": {\"log\": 1, \"step\": 1, \"low\": 1, \"high\": 5}}");
        logOne.Log.Should().BeTrue();

        ((Action)(() => Distribution.FromJson(
                "{\"name\": \"FloatDistribution\", \"attributes\": {\"log\": false, \"step\": null, \"low\": 1, \"high\": 5, \"foo\": 1}}")))
            .Should().Throw<TypeError>();
    }

    /// <summary>
    ///     QA-8 (message text): the "range is not divisible by step" warning must print the bounds as given,
    ///     so JSON ints read by <see cref="Distribution.FromJson" /> print as ints, as in Python.
    /// </summary>
    /// <remarks>
    ///     Python: <c>json_to_distribution('{"type": "float", "low": 0, "high": 1, "step": 0.3}')</c> warns
    ///     "The distribution is specified by [0, 1] and step=0.3, ..." (the JSON ints reach the f-string); the
    ///     port warns "[0.0, 1.0]". NIT-level, but it is the text users see from the CLI's search-space JSON.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void FromJson_IntBoundsInFloatWarning_PrintLikePython()
    {
        var warnings = CaptureWarnings(() => Distribution.FromJson("{\"type\": \"float\", \"low\": 0, \"high\": 1, \"step\": 0.3}"));

        warnings.Should().ContainSingle().Which.Should().Be(
            "The distribution is specified by [0, 1] and step=0.3, but the range is not divisible by `step`. It will be replaced with [0, 0.9].");
    }

    /// <summary>
    ///     QA-9: a deprecated distribution class must emit its <c>FutureWarning</c> before any warning raised
    ///     while it is constructed, because Optuna's <c>@deprecated_class</c> warns before calling
    ///     <c>__init__</c>.
    /// </summary>
    /// <remarks>
    ///     Python: <c>DiscreteUniformDistribution(0.0, 1.0, 0.3)</c> warns "DiscreteUniformDistribution has been
    ///     deprecated in v3.0.0. ..." and then "The distribution is specified by [0.0, 1.0] and step=0.3, ...".
    ///     The port warns in the opposite order (its warning runs in the derived constructor body, after the
    ///     base constructor). Same for <c>IntUniformDistribution(0, 10, 3)</c>.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void DeprecatedDistribution_FutureWarning_ComesFirst()
    {
#pragma warning disable CS0618 // the deprecated aliases are what is under test
        var warnings = CaptureWarnings(() => new DiscreteUniformDistribution(0.0, 1.0, 0.3));
#pragma warning restore CS0618

        warnings.Should().HaveCount(2);
        warnings[0].Should().StartWith("DiscreteUniformDistribution has been deprecated in v3.0.0.");
        warnings[1].Should().StartWith("The distribution is specified by [0.0, 1.0] and step=0.3");
    }

    /// <summary>
    ///     QA-10: looking up a value of an unsupported type in a categorical distribution must raise
    ///     Python's <c>ValueError</c> ("not in"), not a <see cref="TypeError" />.
    /// </summary>
    /// <remarks>
    ///     Python: <c>CategoricalDistribution([1, 2]).to_internal_repr([1])</c> raises <c>ValueError: '[1]' not
    ///     in (1, 2).</c>; the port's <c>CategoricalChoice.From</c> throws <c>TypeError: Choices for a
    ///     categorical distribution should be None, bool, int, float or str, but got System.Int32[] ...</c>
    ///     before the lookup. Reachable from an enqueued trial whose value is an array.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void CategoricalLookup_UnsupportedValueType_RaisesValueError()
    {
        var dist = new CategoricalDistribution(1, 2);
        ((Action)(() => dist.ToInternalRepr(new List<object?> { 1L }))).Should().Throw<ValueError>()
            .WithMessage("'[1]' not in (1, 2).");
    }

    /// <summary>Runs an action and returns the messages of every Optuna warning it raised, in order.</summary>
    /// <param name="action">The code whose warnings are captured; its own exceptions propagate.</param>
    /// <returns>The warning messages, without their category prefix, in emission order.</returns>
    /// <remarks>
    ///     Subscribes to the process-wide <see cref="OptunaWarnings.WarningRaised" /> event for the duration of
    ///     the call only; tests running in parallel can add their own warnings to the list, so callers assert
    ///     on messages they can attribute to the action.
    /// </remarks>
    private static List<string> CaptureWarnings(Action action)
    {
        var messages = new List<string>();
        void Handler(string message, WarningCategory category) => messages.Add(message);
        OptunaWarnings.WarningRaised += Handler;
        try
        {
            action();
        }
        finally
        {
            // Unsubscribe even when the action throws, or later tests would see this test's handler.
            OptunaWarnings.WarningRaised -= Handler;
        }
        return messages;
    }
}
