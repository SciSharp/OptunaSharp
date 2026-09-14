using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests;

/// <summary>
///     The <c>TEnum</c> categorical convenience layer: <c>SuggestCategorical&lt;TEnum&gt;</c>, the
///     <c>GetCategorical&lt;TEnum&gt;</c> read lens, the write-boundary enum→name normalization at
///     enqueue / fixed-trial, and the <see cref="EnumCategorical.Choices{TEnum}" /> grid builder.
///
///     The load-bearing property is <b>engine transparency</b>: an enum is optimized as its
///     member-name strings, so a study driven by the enum overload is bit-identical to one driven by
///     the hand-written string overload. The type is never persisted — every stored value is the
///     name string — which is why the read side must re-supply <c>TEnum</c> and can fail on mismatch.
/// </summary>
[TestClass]
public class EnumCategoricalTests
{
    // Declaration order == underlying-value order (0, 1), so name/value order coincide here.
    private enum Optimizer { MomentumSGD, Adam }

    // Underlying values deliberately out of declaration order to pin the "order is by value" rule.
    private enum OutOfOrder { B = 2, A = 1, C = 3 }

    [Flags]
    private enum Flagged { None = 0, A = 1, B = 2, C = 4 }

    private enum Empty { }

    private static Study NewStudy(int seed = 0) => Optuna.CreateStudy(sampler: new RandomSampler(seed));

    [TestMethod]
    public void EnumOverload_Is_ByteIdentical_To_HandWritten_String_Categorical()
    {
        // Same seed, same choices (the member names in by-value order) => the RandomSampler draws the
        // same index in both studies, so the enum path and the string path must agree on the value AND
        // record the identical CategoricalDistribution. This is the whole correctness claim.
        string[] names = Enum.GetNames<Optimizer>();

        var enumStudy = NewStudy();
        var enumTrial = enumStudy.Ask();
        Optimizer chosen = enumTrial.SuggestCategorical<Optimizer>("optimizer");
        enumStudy.Tell(enumTrial, 0.0);

        var stringStudy = NewStudy();
        var stringTrial = stringStudy.Ask();
        string chosenString = stringTrial.SuggestCategorical("optimizer", names);
        stringStudy.Tell(stringTrial, 0.0);

        Enum.GetName(chosen).Should().Be(chosenString, "the enum layer emits exactly the string categorical");

        var enumStored = enumStudy.Trials.Single();
        var stringStored = stringStudy.Trials.Single();
        enumStored.Params["optimizer"].Should().Be(chosenString);
        enumStored.Params["optimizer"].Should().Be(stringStored.Params["optimizer"]);
        // The persisted distribution carries only the name strings — no enum metadata on the wire.
        enumStored.Distributions["optimizer"].Should().Be(new CategoricalDistribution("MomentumSGD", "Adam"));
        enumStored.Distributions["optimizer"].Should().Be(stringStored.Distributions["optimizer"]);
    }

    [TestMethod]
    public void Suggest_RoundTrips_Through_Read_Lens_And_Stores_A_String()
    {
        var study = NewStudy();
        var trial = study.Ask();
        Optimizer chosen = trial.SuggestCategorical<Optimizer>("optimizer");

        // Params holds the NAME STRING, not a boxed enum — consistent with any other categorical.
        trial.Params["optimizer"].Should().BeOfType<string>().And.Be(Enum.GetName(chosen));
        // Both read-lens overloads resolve it back to the same member.
        trial.GetCategorical<Optimizer>("optimizer").Should().Be(chosen);
        trial.Params.GetCategorical<Optimizer>("optimizer").Should().Be(chosen);

        study.Tell(trial, 0.0);
        study.Trials.Single().Params.GetCategorical<Optimizer>("optimizer").Should().Be(chosen);
    }

    [TestMethod]
    public void Subset_Overload_Preserves_Given_Order_As_Choice_Indices()
    {
        var study = NewStudy();
        var trial = study.Ask();
        // Explicit subset in reversed order — the stored distribution must keep THIS order (it fixes
        // the categorical indices), not the enum's by-value order.
        Optimizer chosen = trial.SuggestCategorical("optimizer", new[] { Optimizer.Adam, Optimizer.MomentumSGD });
        chosen.Should().BeOneOf(Optimizer.Adam, Optimizer.MomentumSGD);
        study.Tell(trial, 0.0);

        study.Trials.Single().Distributions["optimizer"]
            .Should().Be(new CategoricalDistribution("Adam", "MomentumSGD"));
    }

    [TestMethod]
    public void Choices_Builder_Orders_By_Underlying_Value_Not_Declaration()
    {
        // OutOfOrder declares B, A, C but values 2, 1, 3 => by-value order is A, B, C.
        EnumCategorical.Choices<OutOfOrder>().Select(c => c.Value)
            .Should().Equal("A", "B", "C");
    }

    [TestMethod]
    public void EnqueueTrial_Accepts_A_Raw_Enum_And_Stores_Its_Name()
    {
        var study = NewStudy();
        // A boxed enum at the enqueue boundary is normalized to its name before it is stored/compared.
        study.EnqueueTrial(new OrderedDict<string, object?> { ["optimizer"] = Optimizer.Adam });

        Optimizer? seen = null;
        study.Optimize(t => { seen = t.SuggestCategorical<Optimizer>("optimizer"); return 0.0; }, nTrials: 1);

        seen.Should().Be(Optimizer.Adam, "the enqueued enum drives the replayed trial");
        study.Trials.Single().Params["optimizer"].Should().Be("Adam", "stored as the member-name string");
    }

    [TestMethod]
    public void EnqueueTrial_SkipIfExists_Matches_A_Previously_Enqueued_Enum()
    {
        var study = NewStudy();
        study.EnqueueTrial(new OrderedDict<string, object?> { ["optimizer"] = Optimizer.Adam });
        // The skip check compares against the stored name string; normalizing the new enum first is
        // what makes this second enqueue recognize the duplicate.
        study.EnqueueTrial(new OrderedDict<string, object?> { ["optimizer"] = Optimizer.Adam }, skipIfExists: true);

        study.Trials.Count(t => t.State == TrialState.Waiting).Should().Be(1);
    }

    [TestMethod]
    public void FixedTrial_Accepts_A_Raw_Enum_And_Exposes_Its_Name()
    {
        var fixedTrial = new FixedTrial(new OrderedDict<string, object?> { ["optimizer"] = Optimizer.MomentumSGD });

        fixedTrial.SuggestCategorical<Optimizer>("optimizer").Should().Be(Optimizer.MomentumSGD);
        fixedTrial.Params["optimizer"].Should().Be("MomentumSGD", "the replayed value is recorded as a string");
        fixedTrial.GetCategorical<Optimizer>("optimizer").Should().Be(Optimizer.MomentumSGD);
    }

    [TestMethod]
    public void GridSampler_Over_Enum_Choices_Visits_Every_Member()
    {
        var searchSpace = new Dictionary<string, IReadOnlyList<CategoricalChoice>>
        {
            ["optimizer"] = EnumCategorical.Choices<Optimizer>(),
        };
        var study = Optuna.CreateStudy(sampler: new GridSampler(searchSpace, seed: 0));

        var seen = new List<Optimizer>();
        // GridSampler stops itself once the 2-cell grid is exhausted, so the large nTrials is a ceiling.
        study.Optimize(t => { seen.Add(t.SuggestCategorical<Optimizer>("optimizer")); return 0.0; }, nTrials: 10);

        seen.Distinct().Should().BeEquivalentTo(new[] { Optimizer.MomentumSGD, Optimizer.Adam });
    }

    [TestMethod]
    public void Read_Lens_Throws_KeyError_For_A_Missing_Parameter()
    {
        var study = NewStudy();
        var trial = study.Ask();

        var act = () => trial.GetCategorical<Optimizer>("absent");
        act.Should().Throw<KeyError>();
    }

    [TestMethod]
    public void Read_Lens_Throws_ValueError_When_The_String_Is_Not_A_Member()
    {
        var study = NewStudy();
        var trial = study.Ask();
        trial.SuggestCategorical("optimizer", new[] { "zzz" });  // a valid string categorical, alien to Optimizer

        var act = () => trial.GetCategorical<Optimizer>("optimizer");
        act.Should().Throw<ValueError>().WithMessage("*is not a declared member of enum*");
    }

    [TestMethod]
    public void Read_Lens_Throws_TypeError_For_A_NonString_Parameter()
    {
        var study = NewStudy();
        var trial = study.Ask();
        trial.SuggestFloat("x", 0, 1);  // a float parameter, not a categorical

        var act = () => trial.GetCategorical<Optimizer>("x");
        act.Should().Throw<TypeError>();
    }

    [TestMethod]
    public void Subset_Overload_Rejects_A_Flags_Combination()
    {
        var study = NewStudy();
        var trial = study.Ask();

        // A [Flags] combination has no single member name and could not round-trip, so it is rejected.
        var act = () => trial.SuggestCategorical("f", new[] { Flagged.A | Flagged.B });
        act.Should().Throw<ArgumentException>().WithMessage("*is not a single declared member*");
    }

    [TestMethod]
    public void Subset_Overload_Rejects_An_Undefined_Value()
    {
        var study = NewStudy();
        var trial = study.Ask();

        var act = () => trial.SuggestCategorical("o", new[] { (Optimizer)999 });
        act.Should().Throw<ArgumentException>().WithMessage("*is not a single declared member*");
    }

    [TestMethod]
    public void Subset_Overload_Rejects_Null_Choices()
    {
        var study = NewStudy();
        var trial = study.Ask();

        var act = () => trial.SuggestCategorical<Optimizer>("o", null!);
        act.Should().Throw<ArgumentNullException>();
    }

    [TestMethod]
    public void AllMembers_Overload_On_An_Empty_Enum_Throws_ValueError()
    {
        var study = NewStudy();
        var trial = study.Ask();

        // No members => no choices => the CategoricalDistribution constructor rejects it, same as an
        // empty string[] would.
        var act = () => trial.SuggestCategorical<Empty>("e");
        act.Should().Throw<ValueError>().WithMessage("*one or more elements*");
    }
}
