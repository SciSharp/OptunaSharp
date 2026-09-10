using System;
using System.Collections.Generic;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests;

/// <summary>
///     Tests for the <see cref="ComponentLosses" /> observed-components channel — the shared
///     <c>SetComponentLosses</c> writer / <c>TryGetComponentLosses</c> reader that formalize the
///     <c>component_losses</c> user-attr convention JANUS's component mode (and, later, the MOO metric)
///     consume. Covers the plan's validation matrix: set/get round-trip (in-memory storage + the JSON
///     wire form RDB/journal storages serialize through), list-vs-array tolerance, non-finite/empty
///     rejection on both sides, the <c>rae</c> alias with Python-<c>or</c> fall-through, and the
///     per-trial / mixed-window substrate JANUS's scalar fallback is built on.
/// </summary>
[TestClass]
public class ComponentLossesTests
{
    // ---- round-trip: writer → reader on a single FrozenTrial -------------------------------------

    [TestMethod]
    public void SetGet_RoundTrip_OnFrozenTrial()
    {
        var r = new[] { 0.25, 1.5, 4.0 };
        var trial = FrozenTrial.CreateTrial(value: 5.75);
        trial.SetComponentLosses(r);

        trial.UserAttrs.Should().ContainKey(ComponentLosses.Key);
        trial.TryGetComponentLosses(out var got).Should().BeTrue();
        got.Should().Equal(r);
        got.Should().NotBeSameAs(r, "the reader hands back a caller-owned copy");
    }

    // ---- round-trip: through a real in-memory study storage --------------------------------------

    [TestMethod]
    public void SetGet_RoundTrip_ThroughStudyStorage()
    {
        var study = Optuna.CreateStudy();
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", -5, 5);
            var r = new[] { x * x, (x - 1.0) * (x - 1.0) };
            t.SetComponentLosses(r);
            return r[0] + r[1];
        }, nTrials: 4);

        foreach (var ft in study.Trials)
        {
            ft.TryGetComponentLosses(out var got).Should().BeTrue("every trial exposed a component vector");
            got.Length.Should().Be(2);
        }
    }

    // ---- round-trip: through the JSON wire form RDB / journal storages serialize -----------------

    [TestMethod]
    public void SetGet_RoundTrip_ThroughJsonWireForm()
    {
        // A double[] written via SetUserAttr is serialized to JSON by RDB/journal storages and read back
        // as a List<object?> of long/double. Prove the reader tolerates that exact transformation.
        var r = new[] { 0.0, 0.5, 2.25, 10.0 };
        object? wire = PyJson.Loads(PyJson.Dumps(r));
        wire.Should().BeOfType<List<object?>>("storages deserialize a JSON array to a List<object?>");

        var attrs = new OrderedDict<string, object?> { [ComponentLosses.Key] = wire };
        ComponentLosses.TryRead(attrs, out var got).Should().BeTrue();
        got.Should().Equal(r);
    }

    // ---- list-vs-array tolerance -----------------------------------------------------------------

    [TestMethod]
    public void Reader_Tolerates_ListArrayAndIntegerForms()
    {
        var expected = new[] { 1.0, 2.0, 3.0 };
        var forms = new object[]
        {
            new double[] { 1.0, 2.0, 3.0 },
            new List<object?> { 1.0, 2.0, 3.0 },
            new List<object?> { 1L, 2L, 3L },       // JSON integers (whole-number residuals)
            new object[] { 1.0, 2L, 3.0 },           // mixed long/double, as a round-tripped array may be
            new List<double> { 1.0, 2.0, 3.0 },
            new int[] { 1, 2, 3 },
            new float[] { 1f, 2f, 3f },
        };

        foreach (var form in forms)
        {
            var attrs = new OrderedDict<string, object?> { [ComponentLosses.Key] = form };
            ComponentLosses.TryRead(attrs, out var got)
                .Should().BeTrue($"the {form.GetType().Name} form is a valid component vector");
            got.Should().Equal(expected);
        }
    }

    // ---- writer rejects non-finite / empty (fail fast, nothing stored) ---------------------------

    [TestMethod]
    public void Writer_Rejects_NonFiniteAndEmpty()
    {
        var trial = FrozenTrial.CreateTrial(value: 0.0);

        Action nan = () => trial.SetComponentLosses(new[] { 1.0, double.NaN });
        nan.Should().Throw<ValueError>().WithMessage("*non-finite*");

        Action inf = () => trial.SetComponentLosses(new[] { double.PositiveInfinity });
        inf.Should().Throw<ValueError>().WithMessage("*non-finite*");

        Action empty = () => trial.SetComponentLosses(Array.Empty<double>());
        empty.Should().Throw<ValueError>().WithMessage("*empty*");

        trial.UserAttrs.Should().NotContainKey(ComponentLosses.Key, "a rejected write stores nothing");
    }

    // ---- reader rejects non-finite / null / empty / non-numeric ----------------------------------

    [TestMethod]
    public void Reader_Rejects_NonFiniteNullEmptyAndNonNumeric()
    {
        void ReadsFalse(object? value, string because)
        {
            var attrs = new OrderedDict<string, object?> { [ComponentLosses.Key] = value };
            ComponentLosses.TryRead(attrs, out var got).Should().BeFalse(because);
            got.Should().BeEmpty();
        }

        ReadsFalse(new List<object?> { 1.0, double.NaN }, "a NaN element is rejected");
        ReadsFalse(new List<object?> { double.PositiveInfinity }, "an infinite element is rejected");
        ReadsFalse(new double[] { 1.0, double.NegativeInfinity }, "a -inf element in a double[] is rejected");
        ReadsFalse(new List<object?> { 1.0, null }, "a null element is rejected");
        ReadsFalse(Array.Empty<double>(), "an empty vector is rejected");
        ReadsFalse(new List<object?>(), "an empty list is rejected");
        ReadsFalse("not a vector", "a string is never a component vector");
        ReadsFalse(42.0, "a bare scalar is not a component vector");
        ReadsFalse(null, "an explicit null value reads as absent");

        // No key at all ⇒ absent.
        ComponentLosses.TryRead(new OrderedDict<string, object?>(), out var none).Should().BeFalse();
        none.Should().BeEmpty();
    }

    // ---- the `rae` read alias and Python-`or` fall-through ---------------------------------------

    [TestMethod]
    public void Reader_UsesRaeAlias_AndFallsThroughFromInvalidCanonical()
    {
        // `rae` alias alone is honored (reference portability).
        var raeOnly = new OrderedDict<string, object?> { [ComponentLosses.LegacyKey] = new double[] { 5.0, 6.0 } };
        ComponentLosses.TryRead(raeOnly, out var g1).Should().BeTrue();
        g1.Should().Equal(new[] { 5.0, 6.0 });

        // Canonical present but empty/invalid ⇒ fall through to `rae` (mirrors `... or ...`).
        var fallThrough = new OrderedDict<string, object?>
        {
            [ComponentLosses.Key] = Array.Empty<double>(),
            [ComponentLosses.LegacyKey] = new double[] { 7.0 },
        };
        ComponentLosses.TryRead(fallThrough, out var g2).Should().BeTrue();
        g2.Should().Equal(new[] { 7.0 });

        // Canonical wins when both are valid.
        var bothValid = new OrderedDict<string, object?>
        {
            [ComponentLosses.Key] = new double[] { 1.0 },
            [ComponentLosses.LegacyKey] = new double[] { 2.0 },
        };
        ComponentLosses.TryRead(bothValid, out var g3).Should().BeTrue();
        g3.Should().Equal(new[] { 1.0 });
    }

    // ---- per-trial reads across a mixed window (substrate for JANUS's scalar fallback) -----------

    [TestMethod]
    public void Reader_IsPerTrial_AcrossMixedWindow()
    {
        // The channel is per-trial; cross-trial length consistency is the consumer's concern (JANUS
        // uses component mode only when ALL window trials carry components, else scalar mode). Verify the
        // substrate: present trials read their own-length vector, absent ones read false.
        var m3 = FrozenTrial.CreateTrial(value: 1.0);
        m3.SetComponentLosses(new[] { 1.0, 2.0, 3.0 });
        var m2 = FrozenTrial.CreateTrial(value: 1.0);
        m2.SetComponentLosses(new[] { 4.0, 5.0 });
        var none = FrozenTrial.CreateTrial(value: 1.0);

        m3.TryGetComponentLosses(out var a).Should().BeTrue();
        a.Length.Should().Be(3);
        m2.TryGetComponentLosses(out var b).Should().BeTrue();
        b.Length.Should().Be(2);
        none.TryGetComponentLosses(out var c).Should().BeFalse();
        c.Should().BeEmpty();
    }

    // ---- negative entries are permitted (JANUS reads them at their zero clamp) --------------------

    [TestMethod]
    public void Writer_Permits_NegativeEntries()
    {
        var trial = FrozenTrial.CreateTrial(value: 0.0);
        Action neg = () => trial.SetComponentLosses(new[] { -1.0, 2.0 });
        neg.Should().NotThrow("a negative residual is permitted (read at its zero clamp downstream)");

        trial.TryGetComponentLosses(out var got).Should().BeTrue();
        got.Should().Equal(new[] { -1.0, 2.0 });
    }

    // ---- integration: the typed writer feeds JANUS identically to the raw user-attr --------------

    [TestMethod]
    public void Janus_ComponentMode_ViaTypedApi_Converges()
    {
        // Same 10-D shifted sphere the raw-user-attr component-mode test converges on, driven through the
        // typed SetComponentLosses writer — proves the channel feeds JANUS exactly as before.
        var study = Optuna.CreateStudy(sampler: new JanusSampler(seed: 0, nStartupTrials: 16));
        study.Optimize(t =>
        {
            var r = new double[10];
            double v = 0;
            for (int i = 0; i < 10; i++)
            {
                double x = t.SuggestFloat($"x{i}", -5, 5);
                r[i] = (x - 0.5) * (x - 0.5);
                v += r[i];
            }
            t.SetComponentLosses(r);
            return v;
        }, nTrials: 400);

        study.BestValue.Should().BeLessThan(1e-1,
            "the typed component channel drives JANUS just like the raw component_losses user-attr");
    }

    // ---- integration: mixed component presence falls back to scalar mode without error -----------

    [TestMethod]
    public void Janus_MixedComponentPresence_StillRuns()
    {
        var study = Optuna.CreateStudy(sampler: new JanusSampler(seed: 7, nStartupTrials: 16));
        study.Optimize(t =>
        {
            var x = new double[6];
            var r = new double[6];
            double v = 0;
            for (int i = 0; i < 6; i++)
            {
                x[i] = t.SuggestFloat($"x{i}", -5, 5);
                r[i] = (x[i] - 0.5) * (x[i] - 0.5);
                v += r[i];
            }
            if (t.Number % 2 == 0)
                t.SetComponentLosses(r);   // only half the trials carry components
            return v;
        }, nTrials: 120);

        double best = study.BestValue;
        double.IsFinite(best).Should().BeTrue();
        best.Should().BeLessThan(20.0,
            "a window that is not all-components degrades to scalar mode and still optimizes without error");
    }
}
