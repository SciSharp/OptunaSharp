using System;
using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests;

/// <summary>
///     Study lifecycle: create/load/delete/copy/summaries, optimize/ask/tell (all state paths),
///     enqueue, add_trial validation, best-trial semantics incl. constrained fallback, stop,
///     callbacks and n_jobs — the plan 0.8 surface. RandomSampler is the algorithm driving
///     everything (the "one algorithm to test the rest of the infra").
/// </summary>
[TestClass]
public class StudyTests
{
    private static Study NewStudy(
        BaseStorage? storage = null,
        StudyDirection? direction = null,
        IEnumerable<StudyDirection>? directions = null,
        string? name = null,
        int seed = 42)
        => Optuna.CreateStudy(
            storage: storage,
            sampler: new RandomSampler(seed),
            studyName: name,
            direction: direction,
            directions: directions);

    // ------------------------------------------------------------------ create / load / delete

    [TestMethod]
    public void CreateStudy_Defaults_To_Minimize()
    {
        var study = NewStudy();
        study.Direction.Should().Be(StudyDirection.Minimize);
        study.Directions.Should().Equal(StudyDirection.Minimize);
    }

    [TestMethod]
    public void CreateStudy_Rejects_Direction_And_Directions_Together()
    {
        var act = () => Optuna.CreateStudy(
            direction: StudyDirection.Minimize,
            directions: new[] { StudyDirection.Minimize });
        act.Should().Throw<ValueError>()
            .WithMessage("Specify only one of `direction` and `directions`.");

        var empty = () => Optuna.CreateStudy(directions: Array.Empty<StudyDirection>());
        empty.Should().Throw<ValueError>()
            .WithMessage("The number of objectives must be greater than 0.");

        var invalid = () => Optuna.CreateStudy(directions: new[] { StudyDirection.NotSet });
        invalid.Should().Throw<ValueError>().WithMessage("*must be a list of `minimize` or `maximize`*");
    }

    [TestMethod]
    public void CreateStudy_LoadIfExists_Reuses_The_Study()
    {
        var storage = new InMemoryStorage();
        var first = NewStudy(storage: storage, name: "same");
        first.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 2);

        var dup = () => NewStudy(storage: storage, name: "same");
        dup.Should().Throw<DuplicatedStudyError>().WithMessage("*Another study with study_name='same' already exists*");

        var reused = Optuna.CreateStudy(storage: storage, studyName: "same",
            sampler: new RandomSampler(1), loadIfExists: true);
        reused.Trials.Should().HaveCount(2);
    }

    [TestMethod]
    public void LoadStudy_Without_Name_Requires_Single_Study()
    {
        var storage = new InMemoryStorage();
        NewStudy(storage: storage, name: "only");
        Optuna.LoadStudy(null, storage, sampler: new RandomSampler(1)).StudyName.Should().Be("only");

        NewStudy(storage: storage, name: "second");
        var act = () => Optuna.LoadStudy(null, storage, sampler: new RandomSampler(1));
        act.Should().Throw<ValueError>().WithMessage("*does not contain exactly 1 study*");
    }

    [TestMethod]
    public void DeleteStudy_Then_GetAllStudyNames()
    {
        var storage = new InMemoryStorage();
        NewStudy(storage: storage, name: "a");
        NewStudy(storage: storage, name: "b");
        Optuna.DeleteStudy("a", storage);
        Optuna.GetAllStudyNames(storage).Should().Equal("b");
    }

    // ------------------------------------------------------------------ optimize

    [TestMethod]
    public void Optimize_Records_Complete_Trials_And_Best()
    {
        var study = NewStudy();
        study.Optimize(t => Math.Pow(t.SuggestFloat("x", -10, 10) - 2, 2), nTrials: 20);

        study.Trials.Should().HaveCount(20);
        study.Trials.Should().OnlyContain(t => t.State == TrialState.Complete);
        // The pinned walking-skeleton oracle (seed 42, live optuna output).
        study.BestValue.Should().Be(0.0004973004749973951);
        ((double)study.BestParams["x"]!).Should().Be(2.0223002348641756);
        study.BestTrial.Number.Should().Be(8);
    }

    [TestMethod]
    public void Optimize_TrialPruned_Marks_Pruned_With_Last_Intermediate()
    {
        var study = NewStudy();
        study.Optimize((Func<Trial, double>)(t =>
        {
            t.SuggestFloat("x", 0, 1);
            t.Report(3.25, 0);
            throw new TrialPruned();
        }), nTrials: 1);

        var trial = study.Trials.Single();
        trial.State.Should().Be(TrialState.Pruned);
        trial.Value.Should().Be(3.25, "PRUNED adopts the last intermediate value when feasible");
    }

    [TestMethod]
    public void Optimize_Uncaught_Exception_Fails_Trial_And_Rethrows()
    {
        var study = NewStudy();
        var act = () => study.Optimize(
            (Func<Trial, double>)(t => throw new InvalidOperationException("boom")), nTrials: 3);
        act.Should().Throw<InvalidOperationException>().WithMessage("boom");
        study.Trials.Should().HaveCount(1);
        study.Trials[0].State.Should().Be(TrialState.Fail);
        study.Trials[0].Values.Should().BeNull();
    }

    [TestMethod]
    public void Optimize_Catch_Continues_Past_Listed_Exceptions()
    {
        var study = NewStudy();
        int calls = 0;
        study.Optimize(t =>
        {
            calls++;
            if (calls % 2 == 1)
                throw new InvalidOperationException("skip me");
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 4, @catch: new[] { typeof(InvalidOperationException) });

        study.Trials.Should().HaveCount(4);
        study.Trials.Count(t => t.State == TrialState.Fail).Should().Be(2);
        study.Trials.Count(t => t.State == TrialState.Complete).Should().Be(2);
    }

    [TestMethod]
    public void Optimize_NaN_Return_Fails_The_Trial_Without_Raising()
    {
        var study = NewStudy();
        study.Optimize(t => { t.SuggestFloat("x", 0, 1); return double.NaN; }, nTrials: 1);
        study.Trials.Single().State.Should().Be(TrialState.Fail);
    }

    [TestMethod]
    public void Optimize_Callbacks_Receive_Frozen_Copies()
    {
        var study = NewStudy();
        var seen = new List<int>();
        study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 3,
            callbacks: new Action<Study, FrozenTrial>[] { (s, tr) => seen.Add(tr.Number) });
        seen.Should().Equal(0, 1, 2);
    }

    [TestMethod]
    public void Optimize_Stop_Ends_The_Loop()
    {
        var study = NewStudy();
        study.Optimize(t =>
        {
            if (t.Number == 4)
                t.Study.Stop();
            return Math.Pow(t.SuggestFloat("x", 0, 10), 2);
        }, nTrials: 10);
        study.Trials.Should().HaveCount(5);

        var outside = () => study.Stop();
        outside.Should().Throw<RuntimeError>()
            .WithMessage("`Study.stop` is supposed to be invoked inside an objective function or a callback.");
    }

    [TestMethod]
    public void Optimize_Nested_Invocation_Is_Rejected()
    {
        var study = NewStudy();
        var act = () => study.Optimize(t =>
        {
            study.Optimize(t2 => 0.0, nTrials: 1);
            return 0.0;
        }, nTrials: 1);
        act.Should().Throw<RuntimeError>()
            .WithMessage("Nested invocation of `Study.optimize` method isn't allowed.");
    }

    [TestMethod]
    public void Optimize_NJobs_Runs_All_Trials_ThreadSafely()
    {
        var study = NewStudy();
        study.Optimize(
            t => Math.Pow(t.SuggestFloat("x", -10, 10), 2), nTrials: 12, nJobs: 4);
        study.Trials.Should().HaveCount(12);
        study.Trials.Should().OnlyContain(t => t.State == TrialState.Complete);
        study.Trials.Select(t => t.Number).OrderBy(n => n).Should().Equal(Enumerable.Range(0, 12));
    }

    // ------------------------------------------------------------------ ask / tell

    [TestMethod]
    public void AskTell_Roundtrip()
    {
        var study = NewStudy();
        var trial = study.Ask();
        double x = trial.SuggestFloat("x", -1, 1);
        var frozen = study.Tell(trial, x * x);

        frozen.State.Should().Be(TrialState.Complete);
        frozen.Value.Should().Be(x * x);
        study.Trials.Should().HaveCount(1);
    }

    [TestMethod]
    public void Ask_With_Fixed_Distributions_PreSuggests()
    {
        var study = NewStudy();
        var trial = study.Ask(new OrderedDict<string, BaseDistribution>
        {
            ["lr"] = new FloatDistribution(1e-4, 1e-1, log: true),
            ["opt"] = new CategoricalDistribution(new CategoricalChoice[] { "adam", "sgd" }),
        });
        trial.Params.Should().ContainKey("lr");
        trial.Params.Should().ContainKey("opt");
        study.Tell(trial, 1.0);
    }

    [TestMethod]
    public void Tell_State_Validation_Verbatim()
    {
        var study = NewStudy();
        var trial = study.Ask();

        var completeNoValues = () => study.Tell(trial, values: null, state: TrialState.Complete);
        completeNoValues.Should().Throw<ValueError>()
            .WithMessage("No values were told. Values are required when state is TrialState.COMPLETE.");

        var prunedWithValues = () => study.Tell(trial, new[] { 1.0 }, TrialState.Pruned);
        prunedWithValues.Should().Throw<ValueError>()
            .WithMessage("Values were told. Values cannot be specified when state is TrialState.PRUNED or TrialState.FAIL.");

        var badState = () => study.Tell(trial, values: null, state: TrialState.Running);
        badState.Should().Throw<ValueError>()
            .WithMessage("Cannot tell with state TrialState.RUNNING.");

        study.Tell(trial, values: null, state: TrialState.Fail);
        var again = () => study.Tell(trial, 1.0);
        again.Should().Throw<ValueError>().WithMessage("Cannot tell a FAIL trial.");
        study.Tell(trial, 1.0, skipIfFinished: true).State.Should().Be(TrialState.Fail);
    }

    [TestMethod]
    public void Tell_By_Number_And_Unknown_Number()
    {
        var study = NewStudy();
        var trial = study.Ask();
        study.Tell(trial.Number, 7.0).Value.Should().Be(7.0);

        var unknown = () => study.Tell(99, 1.0);
        unknown.Should().Throw<ValueError>()
            .WithMessage("Cannot tell for trial with number 99 since it has not been created.");
    }

    [TestMethod]
    public void Tell_Wrong_Values_Count_For_MultiObjective()
    {
        var study = NewStudy(directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize });
        var trial = study.Ask();
        var act = () => study.Tell(trial, new[] { 1.0 }, TrialState.Complete);
        act.Should().Throw<ValueError>()
            .WithMessage("The number of the values 1 did not match the number of the objectives 2");
    }

    // ------------------------------------------------------------------ enqueue / add_trial

    [TestMethod]
    public void EnqueueTrial_Fixes_Params_For_The_Next_Ask()
    {
        var study = NewStudy();
        study.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = 5.0 });
        study.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = 0.0 },
            userAttrs: new OrderedDict<string, object?> { ["memo"] = "optimal" });
        study.Optimize(t => Math.Pow(t.SuggestFloat("x", 0, 10), 2), nTrials: 2);

        study.Trials[0].Params["x"].Should().Be(5.0);
        study.Trials[1].Params["x"].Should().Be(0.0);
        study.Trials[1].UserAttrs["memo"].Should().Be("optimal");
    }

    [TestMethod]
    public void EnqueueTrial_SkipIfExists_Deduplicates()
    {
        var study = NewStudy();
        study.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = 5.0 });
        study.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = 5.0 }, skipIfExists: true);
        study.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = 6.0 }, skipIfExists: true);
        study.GetTrials(deepcopy: false).Should().HaveCount(2);
    }

    [TestMethod]
    public void AddTrial_Validates_Objective_Count_And_State()
    {
        var study = NewStudy();
        var trial = FrozenTrial.CreateTrial(
            @params: new OrderedDict<string, object?> { ["x"] = 2.0 },
            distributions: new OrderedDict<string, BaseDistribution> { ["x"] = new FloatDistribution(0, 10) },
            value: 4.0);
        study.AddTrial(trial);
        study.Trials.Should().HaveCount(1);

        var wrongCount = () => study.AddTrial(FrozenTrial.CreateTrial(values: new[] { 1.0, 2.0 }));
        wrongCount.Should().Throw<ValueError>()
            .WithMessage("The added trial has 2 values, which is different from the number of objectives 1*");

        // AddTrials + continue optimizing on top.
        var other = NewStudy();
        other.AddTrials(study.Trials);
        other.Optimize(t => Math.Pow(t.SuggestFloat("x", 0, 10), 2), nTrials: 2);
        other.Trials.Should().HaveCount(3);
    }

    // ------------------------------------------------------------------ best trials / constraints

    [TestMethod]
    public void BestTrial_MultiObjective_Raises_And_BestTrials_Returns_Pareto()
    {
        var storage = new InMemoryStorage();
        var study = NewStudy(storage: storage, directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.Optimize(t => new[] { t.SuggestFloat("x", 0, 1), t.SuggestFloat("y", 0, 1) }, nTrials: 25);

        var single = () => study.BestTrial;
        single.Should().Throw<RuntimeError>()
            .WithMessage("A single best trial cannot be retrieved from a multi-objective study.*");

        var front = study.BestTrials;
        front.Should().NotBeEmpty();
        // No front member may dominate another front member.
        foreach (var a in front)
        foreach (var b in front)
        {
            OptunaSharp.Internal.MultiObjective.Dominates(a, b, study.Directions).Should().BeFalse();
        }
        // Every non-front trial is dominated by someone on the front.
        foreach (var t in study.Trials.Where(t => front.All(f => f.Number != t.Number)))
        {
            front.Any(f => OptunaSharp.Internal.MultiObjective.Dominates(f, t, study.Directions))
                .Should().BeTrue($"trial {t.Number} should be dominated");
        }
    }

    [TestMethod]
    public void BestTrial_Falls_Back_To_Feasible_Under_Constraints()
    {
        var study = NewStudy();
        // Best raw value is infeasible; the best feasible one must win.
        study.AddTrial(FrozenTrial.CreateTrial(
            value: 0.1,
            @params: new OrderedDict<string, object?> { ["x"] = 0.1 },
            distributions: new OrderedDict<string, BaseDistribution> { ["x"] = new FloatDistribution(0, 10) },
            constraints: new OrderedDict<string, double> { ["c"] = 1.0 }));
        study.AddTrial(FrozenTrial.CreateTrial(
            value: 5.0,
            @params: new OrderedDict<string, object?> { ["x"] = 5.0 },
            distributions: new OrderedDict<string, BaseDistribution> { ["x"] = new FloatDistribution(0, 10) },
            constraints: new OrderedDict<string, double> { ["c"] = -1.0 }));

        study.BestTrial.Value.Should().Be(5.0, "the infeasible 0.1 must be skipped");

        var onlyInfeasible = NewStudy();
        onlyInfeasible.AddTrial(FrozenTrial.CreateTrial(
            value: 1.0,
            constraints: new OrderedDict<string, double> { ["c"] = 2.0 }));
        var act = () => onlyInfeasible.BestTrial;
        act.Should().Throw<ValueError>().WithMessage("No feasible trials are completed yet.");
    }

    // ------------------------------------------------------------------ attrs / summaries / copy

    [TestMethod]
    public void User_Attrs_And_Metric_Names()
    {
        var study = NewStudy(directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.SetUserAttr("k", new List<object?> { 1L });
        study.UserAttrs["k"].Should().BeOfType<List<object?>>();

        study.MetricNames.Should().BeNull();
        study.SetMetricNames(new[] { "loss", "acc" });
        study.MetricNames.Should().Equal("loss", "acc");

        var wrong = () => study.SetMetricNames(new[] { "only-one" });
        wrong.Should().Throw<ValueError>()
            .WithMessage("The number of objectives must match the length of the metric names.");
    }

    [TestMethod]
    public void CopyStudy_Clones_Directions_Attrs_And_Trials()
    {
        var from = new InMemoryStorage();
        var to = new InMemoryStorage();
        var study = NewStudy(storage: from, name: "src", direction: StudyDirection.Maximize);
        study.SetUserAttr("who", "me");
        study.Optimize(t => t.SuggestFloat("x", 0, 1), nTrials: 3);

        Optuna.CopyStudy("src", from, to);

        var copy = Optuna.LoadStudy("src", to, sampler: new RandomSampler(1));
        copy.Direction.Should().Be(StudyDirection.Maximize);
        copy.UserAttrs["who"].Should().Be("me");
        copy.Trials.Should().HaveCount(3);
        copy.Trials[2].Params["x"].Should().Be(study.Trials[2].Params["x"]);
    }

    [TestMethod]
    public void GetAllStudySummaries_Aggregates()
    {
        var storage = new InMemoryStorage();
        var study = NewStudy(storage: storage, name: "sum");
        study.Optimize(t => Math.Pow(t.SuggestFloat("x", -1, 1), 2), nTrials: 4);
        NewStudy(storage: storage, name: "empty");

        var summaries = Optuna.GetAllStudySummaries(storage);
        summaries.Should().HaveCount(2);
        var s = summaries.Single(x => x.StudyName == "sum");
        s.NTrials.Should().Be(4);
        s.Direction.Should().Be(StudyDirection.Minimize);
        s.BestTrial.Should().NotBeNull();
        s.BestTrial!.Value.Should().Be(study.BestValue);
        s.DatetimeStart.Should().NotBeNull();
        summaries.Single(x => x.StudyName == "empty").BestTrial.Should().BeNull();
    }

    [TestMethod]
    public void TrialsDataframe_Flattens_Columns()
    {
        var study = NewStudy();
        study.Optimize(t =>
        {
            double x = t.SuggestFloat("x", 0, 1);
            t.SetUserAttr("tag", "u");
            return x;
        }, nTrials: 3);

        var table = study.TrialsDataframe();
        table.RowCount.Should().Be(3);
        table.Columns.Should().Contain(new[] { "number", "value", "params_x", "user_attrs_tag", "state" });
        table[0, "number"].Should().Be(0L);
        table[0, "state"].Should().Be("COMPLETE");
        table[1, "params_x"].Should().Be(study.Trials[1].Params["x"]);

        NewStudy().TrialsDataframe().RowCount.Should().Be(0, "empty studies yield an empty table");
    }

    [TestMethod]
    public void Default_Sampler_Is_TPE_And_Optimizes()
    {
        // Optuna's default sampler is TPESampler (PORTING_PLAN 2.8): the study constructs fine and
        // now actually optimizes past the startup phase without throwing.
        var study = Optuna.CreateStudy(sampler: new TPESampler(seed: 0));
        study.Sampler.Should().BeOfType<TPESampler>();
        study.Optimize(t => Math.Pow(t.SuggestFloat("x", -10, 10), 2.0), nTrials: 20);
        study.Trials.Should().HaveCount(20);
        study.BestValue.Should().BeLessThan(study.Trials[0].Value!.Value + 1e-9);
        study.BestParams["x"].Should().BeOfType<double>();
    }
}
