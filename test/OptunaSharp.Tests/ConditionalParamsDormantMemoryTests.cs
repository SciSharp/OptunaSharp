using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.NsgaiiiCpt;
using OptunaSharp.SearchSpace;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests;

/// <summary>
///     The <b>dormant-value memory</b> divergence — <see cref="ConditionalParams.PreserveDormantValues" />.
///     Off by default (Optuna always re-samples); when on, a conditional parameter whose branch is inactive
///     is parked in a latent side channel (a trial system attribute — out of every equation: not suggested,
///     not scored, not crossed or mutated) and a reactivating branch resumes from that value instead of a
///     fresh random draw. The unit tests drive <c>ApplyDormantCarry</c> directly for a crisp, deterministic
///     proof of resume / park / no-op; the end-to-end tests run whole studies.
/// </summary>
[TestClass]
public class ConditionalParamsDormantMemoryTests
{
    private const double LrOpt = 0.3, Beta1Opt = 0.9, MomentumOpt = 0.7;

    private static ConditionalParams Rules(bool preserve) => new ConditionalParams { PreserveDormantValues = preserve }
        .When("optimizer", "adam").Activates("beta1")
        .When("optimizer", "sgd").Activates("momentum");

    private static OrderedDict<string, BaseDistribution> Space() => new()
    {
        ["beta1"] = new FloatDistribution(0.5, 0.999),
        ["lr"] = new FloatDistribution(0.0, 1.0),
        ["momentum"] = new FloatDistribution(0.0, 0.99),
        ["optimizer"] = new CategoricalDistribution("adam", "sgd"),
    };

    // ---- unit level: ApplyDormantCarry in isolation (deterministic) -----------------------------

    [TestMethod]
    public void ReactivatedBranch_ResumesLatentValue_InsteadOfRandom()
    {
        var (study, current) = NewRunningTrial();
        // One adam parent established beta1 = 0.87; the rest are sgd (beta1 dormant for them).
        var parents = new List<FrozenTrial>
        {
            Trial(0, ("optimizer", "adam", Cat()), ("lr", 0.10, Lr()), ("beta1", 0.87, Beta1())),
            Trial(1, ("optimizer", "sgd", Cat()), ("lr", 0.20, Lr()), ("momentum", 0.55, Mom())),
            Trial(2, ("optimizer", "sgd", Cat()), ("lr", 0.30, Lr()), ("momentum", 0.65, Mom())),
        };
        // An adam child that no *selected* parent fed beta1 to (it is simply missing from the child).
        var child = new OrderedDict<string, object?> { ["optimizer"] = "adam", ["lr"] = 0.3 };

        Rules(preserve: true).ApplyDormantCarry(study, current, child, parents, Space());

        child.Should().ContainKey("beta1");
        BitConverter.DoubleToInt64Bits((double)child["beta1"]!)
            .Should().Be(BitConverter.DoubleToInt64Bits(0.87), "the reactivating branch resumes its latent value");
    }

    [TestMethod]
    public void ReactivatedBranch_ResumesLatentValue_ParkedByAPriorTrial()
    {
        var (study, current) = NewRunningTrial();
        // No parent has beta1 *active*, but one parked it while dormant — the value still carries.
        var parents = new List<FrozenTrial>
        {
            Trial(0, Dormant(("beta1", 0.73)), ("optimizer", "sgd", Cat()), ("lr", 0.10, Lr()), ("momentum", 0.4, Mom())),
            Trial(1, ("optimizer", "sgd", Cat()), ("lr", 0.20, Lr()), ("momentum", 0.55, Mom())),
        };
        var child = new OrderedDict<string, object?> { ["optimizer"] = "adam", ["lr"] = 0.3 };

        Rules(preserve: true).ApplyDormantCarry(study, current, child, parents, Space());

        BitConverter.DoubleToInt64Bits((double)child["beta1"]!)
            .Should().Be(BitConverter.DoubleToInt64Bits(0.73), "a value parked while dormant is still resumable");
    }

    [TestMethod]
    public void InactiveBranch_IsParkedOutOfTheGenome_NotSuggested()
    {
        var (study, current) = NewRunningTrial();
        var parents = new List<FrozenTrial>
        {
            Trial(0, ("optimizer", "adam", Cat()), ("lr", 0.10, Lr()), ("beta1", 0.87, Beta1())),
            Trial(1, ("optimizer", "sgd", Cat()), ("lr", 0.20, Lr()), ("momentum", 0.55, Mom())),
        };
        // An sgd child: beta1 is inactive here.
        var child = new OrderedDict<string, object?> { ["optimizer"] = "sgd", ["lr"] = 0.2, ["momentum"] = 0.6 };

        Rules(preserve: true).ApplyDormantCarry(study, current, child, parents, Space());

        // beta1 never enters the child's genome — it is not suggested, scored, crossed or mutated ...
        child.Should().NotContainKey("beta1");
        child.Keys.Should().BeEquivalentTo(new[] { "optimizer", "lr", "momentum" });
        // ... it is parked in the trial's latent side channel, where a later reactivation can find it.
        var parked = study.Storage.GetTrialSystemAttrs(current.TrialId);
        parked.Should().ContainKey(ConditionalParams.DormantAttrKey);
        var dormant = (IReadOnlyDictionary<string, object?>)parked[ConditionalParams.DormantAttrKey]!;
        BitConverter.DoubleToInt64Bits((double)dormant["beta1"]!).Should().Be(BitConverter.DoubleToInt64Bits(0.87));
        dormant.Should().NotContainKey("momentum", "momentum is active for this child, so it is not parked");
    }

    [TestMethod]
    public void WithoutTheFlag_DormantCarry_IsACompleteNoOp()
    {
        var (study, current) = NewRunningTrial();
        var parents = new List<FrozenTrial>
        {
            Trial(0, ("optimizer", "adam", Cat()), ("lr", 0.10, Lr()), ("beta1", 0.87, Beta1())),
            Trial(1, ("optimizer", "sgd", Cat()), ("lr", 0.20, Lr()), ("momentum", 0.55, Mom())),
        };
        var child = new OrderedDict<string, object?> { ["optimizer"] = "adam", ["lr"] = 0.3 };

        Rules(preserve: false).ApplyDormantCarry(study, current, child, parents, Space());

        child.Should().NotContainKey("beta1", "with the flag off nothing is injected (Optuna would re-sample)");
        study.Storage.GetTrialSystemAttrs(current.TrialId).Should().NotContainKey(ConditionalParams.DormantAttrKey);
    }

    [TestMethod]
    public void UndecidedActivator_LeavesTheParameterToIndependentSampling()
    {
        var (study, current) = NewRunningTrial();
        var parents = new List<FrozenTrial>
        {
            Trial(0, ("optimizer", "adam", Cat()), ("lr", 0.10, Lr()), ("beta1", 0.87, Beta1())),
        };
        // The activator `optimizer` was dropped (e.g. selected for mutation): activity is undecidable, so
        // dormant carry must not inject beta1 nor park anything — it defers to independent sampling.
        var child = new OrderedDict<string, object?> { ["lr"] = 0.3 };

        Rules(preserve: true).ApplyDormantCarry(study, current, child, parents, Space());

        child.Should().NotContainKey("beta1");
        study.Storage.GetTrialSystemAttrs(current.TrialId).Should().NotContainKey(ConditionalParams.DormantAttrKey);
    }

    // ---- end to end: whole studies --------------------------------------------------------------

    [DataTestMethod]
    [DataRow("NSGAIIICptSampler")]
    public void EndToEnd_ParksInactiveBranch_AndStaysStable(string samplerName)
    {
        var study = RunTwoBranch(samplerName, seed: 0, preserve: true);

        // Every trial is one branch, so the other branch's parameter is parked — the latent channel shows
        // up on the bulk of trials once a value has been learned on each branch.
        int parked = study.Trials.Count(t => t.SystemAttrs.ContainsKey(ConditionalParams.DormantAttrKey));
        parked.Should().BeGreaterThan(study.Trials.Count / 2, "the inactive branch's value is parked each trial");

        // Parked values are always legal values of their (inactive) parameter — never garbage.
        var beta1 = new FloatDistribution(0.5, 0.999);
        var momentum = new FloatDistribution(0.0, 0.99);
        foreach (var t in study.Trials)
        {
            if (!t.SystemAttrs.TryGetValue(ConditionalParams.DormantAttrKey, out var raw) || raw is null)
                continue;
            var dormant = (IReadOnlyDictionary<string, object?>)raw;
            foreach (var (name, value) in dormant)
            {
                var dist = name == "beta1" ? beta1 : momentum;
                dist.Contains(dist.ToInternalRepr(value)).Should().BeTrue($"parked `{name}`={value} is in range");
                // A parked parameter is never simultaneously an active parameter of the same trial.
                t.Params.Should().NotContainKey(name, "a parked parameter is out of the active genome");
            }
        }

        // Both branches still learn — parking does not harm the active-gene optimization. Thresholds sit
        // below the random-draw medians (beta1 0.150, momentum 0.245) with margin for a pop-16 single seed.
        var late = study.Trials.Where(t => Convert.ToInt32(t.SystemAttrs[$"{samplerName}:generation"]) >= 12).ToList();
        Median(late, "beta1", Beta1Opt).Should().BeLessThan(0.14);
        Median(late, "momentum", MomentumOpt).Should().BeLessThan(0.15);
    }

    [DataTestMethod]
    [DataRow("NSGAIIICptSampler")]
    public void TheFlag_IsTheOnlyThingThatChangesBehavior(string samplerName)
    {
        // Same rules, same seed: flag off is a bit-exact no-op (identical to rules alone); flag on diverges,
        // because reactivations resume a latent value instead of drawing independently, and that write never
        // happens with the flag off.
        string off = Canonical(RunTwoBranch(samplerName, seed: 1, preserve: false), samplerName);
        string offAgain = Canonical(RunTwoBranch(samplerName, seed: 1, preserve: false), samplerName);
        string on = Canonical(RunTwoBranch(samplerName, seed: 1, preserve: true), samplerName);

        off.Should().Be(offAgain, "deterministic");
        on.Should().NotBe(off, "the dormant-memory flag is a real, observable divergence");
    }

    // ---- the headline measurement, as an assertion ----------------------------------------------

    [DataTestMethod]
    [DataRow("NSGAIIICptSampler")]
    public void DormantMemory_ReducesRandomReactivations(string samplerName)
    {
        // "Fresh" = a branch parameter's ACTIVE value is one never seen before on that parameter — the
        // signature of a random SampleIndependent draw when a reactivating branch had no carrier parent.
        // With dormant memory on, a reactivation resumes its latent value instead, so fresh counts drop.
        // Measured (seed 2, 256 trials): NSGA-II off (50,50) -> on (20,34); NSGA-III off (51,58) -> on (31,27).
        var off = FreshCounts(RunTwoBranch(samplerName, seed: 2, preserve: false), samplerName);
        var on = FreshCounts(RunTwoBranch(samplerName, seed: 2, preserve: true), samplerName);

        on.beta1.Should().BeLessThan(off.beta1, "reactivating adam resumes `beta1` instead of a random draw");
        on.momentum.Should().BeLessThan(off.momentum, "reactivating sgd resumes `momentum` instead of a random draw");
        (on.beta1 + on.momentum).Should().BeLessThan((int)((off.beta1 + off.momentum) * 0.85),
            "dormant memory removes a large share of random reactivations");
        on.parked.Should().BeGreaterThan(off.parked, "only the memory run parks the inactive branch");
        off.parked.Should().Be(0, "with the flag off nothing is ever parked");
    }

    // ---- helpers -------------------------------------------------------------------------------

    private static (Study, FrozenTrial) NewRunningTrial()
    {
        var study = Optuna.CreateStudy(storage: new InMemoryStorage(), sampler: new RandomSampler(seed: 0),
            direction: StudyDirection.Minimize);
        var trial = study.Ask().GetLatestTrial();   // a real, storage-backed running trial id
        return (study, trial);
    }

    private static FloatDistribution Lr() => new(0.0, 1.0);
    private static FloatDistribution Beta1() => new(0.5, 0.999);
    private static FloatDistribution Mom() => new(0.0, 0.99);
    private static CategoricalDistribution Cat() => new("adam", "sgd");

    private static (string Key, object? Value)[] Dormant(params (string, double)[] values)
        => new[] { ("__dormant__", (object?)values.ToDictionary(v => v.Item1, v => (object?)v.Item2, StringComparer.Ordinal)) };

    // Build a FrozenTrial; a leading Dormant(...) tuple (key "__dormant__") seeds its parked side channel.
    private static FrozenTrial Trial(int number, params (string Name, object? Value, BaseDistribution Distribution)[] entries)
        => Trial(number, Array.Empty<(string, object?)>(), entries);

    private static FrozenTrial Trial(
        int number, (string Key, object? Value)[] dormantSeed,
        params (string Name, object? Value, BaseDistribution Distribution)[] entries)
    {
        var @params = new OrderedDict<string, object?>();
        var distributions = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, value, distribution) in entries)
        {
            @params[name] = value;
            distributions[name] = distribution;
        }
        var systemAttrs = new OrderedDict<string, object?>();
        foreach (var (key, value) in dormantSeed)
            if (key == "__dormant__")
                systemAttrs[ConditionalParams.DormantAttrKey] = value;
        return new FrozenTrial(
            number, TrialState.Complete, value: 0.0, DateTime.UtcNow, DateTime.UtcNow, @params, distributions,
            new OrderedDict<string, object?>(), systemAttrs, new OrderedDict<long, double>(), trialId: number);
    }

    private static Study RunTwoBranch(string samplerName, int seed, bool preserve)
    {
        // The conditional feature (and its dormant-value memory) lives only in NSGA-III-CPT now; `samplerName`
        // is retained only so the callers can read the "NSGAIIICptSampler:generation" system-attribute key.
        _ = samplerName;
        var rules = Rules(preserve);
        BaseSampler sampler = new NSGAIIICptSampler(populationSize: 16, seed: seed, conditionalParams: rules);
        var study = Optuna.CreateStudy(
            storage: new InMemoryStorage(), sampler: sampler,
            directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize });
        study.Optimize(TwoBranchObjective, nTrials: 16 * 16);
        return study;
    }

    private static IReadOnlyList<double> TwoBranchObjective(Trial t)
    {
        string optimizer = t.SuggestCategorical("optimizer", new[] { "adam", "sgd" });
        double lr = t.SuggestFloat("lr", 0.0, 1.0);
        double shared = Math.Pow(lr - LrOpt, 2.0);
        if (optimizer == "adam")
        {
            double beta1 = t.SuggestFloat("beta1", 0.5, 0.999);
            return new[] { shared + Math.Pow(beta1 - Beta1Opt, 2.0), 1.0 + shared };
        }
        double momentum = t.SuggestFloat("momentum", 0.0, 0.99);
        return new[] { 1.0 + shared, shared + Math.Pow(momentum - MomentumOpt, 2.0) };
    }

    private static (int beta1, int momentum, int parked) FreshCounts(Study study, string samplerName)
    {
        var seenBeta1 = new HashSet<long>();
        var seenMomentum = new HashSet<long>();
        int freshBeta1 = 0, freshMomentum = 0, parked = 0;
        foreach (var t in study.Trials)
        {
            if (t.SystemAttrs.ContainsKey(ConditionalParams.DormantAttrKey)) parked++;
            int gen = Convert.ToInt32(t.SystemAttrs[$"{samplerName}:generation"]);
            if (t.Params.TryGetValue("beta1", out var b))
            {
                long bits = BitConverter.DoubleToInt64Bits((double)b!);
                if (gen > 0 && seenBeta1.Add(bits)) freshBeta1++;
                else seenBeta1.Add(bits);
            }
            if (t.Params.TryGetValue("momentum", out var m))
            {
                long bits = BitConverter.DoubleToInt64Bits((double)m!);
                if (gen > 0 && seenMomentum.Add(bits)) freshMomentum++;
                else seenMomentum.Add(bits);
            }
        }
        return (freshBeta1, freshMomentum, parked);
    }

    private static string Canonical(Study study, string samplerName)
    {
        var sb = new StringBuilder();
        foreach (var t in study.Trials)
        {
            sb.Append(t.Number).Append(':').Append(t.SystemAttrs[$"{samplerName}:generation"]).Append('|');
            foreach (var key in t.Params.Keys.OrderBy(k => k, StringComparer.Ordinal))
            {
                object? v = t.Params[key];
                sb.Append(key).Append('=').Append(v is double d ? BitConverter.DoubleToInt64Bits(d).ToString() : v?.ToString() ?? "null").Append(',');
            }
            sb.Append(';');
        }
        return sb.ToString();
    }

    private static double Median(IEnumerable<FrozenTrial> trials, string name, double optimum)
    {
        var sorted = trials.Where(t => t.Params.ContainsKey(name))
            .Select(t => Math.Abs((double)t.Params[name]! - optimum)).OrderBy(d => d).ToList();
        sorted.Should().NotBeEmpty();
        return sorted[sorted.Count / 2];
    }
}
