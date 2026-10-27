using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.Samplers.Nsgaii;
using OptunaSharp.Samplers.NsgaiiiCpt;
using OptunaSharp.Storages;

namespace OptunaSharp.Tests.Samplers;

/// <summary>
///     Regression tests for pre-release review item H3: a sampler instance reused for a second study (a DI
///     registration, a benchmark loop, a notebook) must give that study its own state. Every fresh in-memory
///     storage starts at study id 0, so the per-study caches the samplers kept by study id alone handed study A's
///     generation counts, populations, optimizers and latent points to study B: the GA samplers crashed with
///     <c>ArgumentOutOfRangeException: The view holds 1 trials</c>, and SMS-EMOA, MOEA/D, JANUS, JANUS-DE and
///     REMBO silently bred study B from study A. Optuna keeps no such state, so a reused Python sampler starts
///     the second study at generation 0.
///     <para>The Python-parity versions for NSGA-II and NSGA-III, the two of these samplers with an Optuna
///     counterpart, live in the interop suite (<c>NsgaiiSamplerParityTests</c>, <c>NsgaiiiSamplerParityTests</c>).
///     A recreated study under a reused id on one storage object is covered end to end over a gRPC proxy to
///     SQLite in the storage suite.</para>
/// </summary>
[TestClass]
public class SamplerReuseAcrossStudiesTests
{
    private static readonly StudyDirection[] TwoMinimize = { StudyDirection.Minimize, StudyDirection.Minimize };

    /// <summary>
    ///     The review's repro, for every sampler built on <see cref="BaseGASampler" />: after 60 trials of a first
    ///     study, a second in-memory study (study id 0 again) runs three full generations. Every trial completes,
    ///     the generations restart at 0 and advance every population-size trials, and each generation's parents
    ///     are trials of the second study. Before the fix the second study's first parent lookup threw
    ///     <see cref="ArgumentOutOfRangeException" />.
    /// </summary>
    /// <param name="kind">Which genetic sampler to reuse (see <see cref="MakeGa" />).</param>
    [TestMethod]
    [DataRow("nsga2")]
    [DataRow("nsga3")]
    [DataRow("nsga3cpt")]
    [DataRow("agemoea2")]
    [DataRow("agemoea2cpt")]
    [DataRow("rvea")]
    [DataRow("janus-moo-cmix")]
    [DataRow("janus-agemoea2-cmix")]
    [DataRow("janus-agemoea2-cmix-cpt")]
    public void GaSampler_ReusedForASecondStudy_StartsItAtGenerationZero(string kind)
    {
        BaseGASampler sampler = MakeGa(kind);
        Optuna.CreateStudy(sampler: sampler, directions: TwoMinimize).Optimize(FirstObjective, nTrials: 60);
        // RVEA pins its population size to the reference-vector count on first use.
        int populationSize = sampler.PopulationSize!.Value;

        var second = Optuna.CreateStudy(sampler: sampler, directions: TwoMinimize);
        second.Optimize(SecondObjective, nTrials: 3 * populationSize);

        second.Trials.Should().OnlyContain(t => t.State == TrialState.Complete);
        int[] generations = second.Trials
            .Select(t => Convert.ToInt32(t.SystemAttrs[sampler.GenerationKey], CultureInfo.InvariantCulture))
            .ToArray();
        generations.Should().Equal(Enumerable.Range(0, 3 * populationSize).Select(i => i / populationSize),
            "the second study starts at generation 0, as Optuna's stateless scan does");

        var studyAttrs = second.Storage.GetStudySystemAttrs(second.StudyId);
        for (int generation = 1; generation <= 2; generation++)
        {
            var parentIds = ((IEnumerable<object?>)studyAttrs[sampler.ParentCacheKeyPrefix + generation])
                .Select(id => Convert.ToInt32(id, CultureInfo.InvariantCulture))
                .ToList();
            var eligible = second.Trials
                .Where(t => Convert.ToInt32(t.SystemAttrs[sampler.GenerationKey], CultureInfo.InvariantCulture) < generation)
                .Select(t => t.TrialId);
            parentIds.Should().NotBeEmpty().And.BeSubsetOf(eligible,
                $"generation {generation}'s parents are earlier trials of the second study");
        }
    }

    /// <summary>
    ///     The general invariant behind H3, for every sampler that keeps per-study state: what a reused sampler does
    ///     in a second study must not depend on what it saw in the first. Two instances with the same seed each run
    ///     a first study of the same length on <i>different</i> objectives, then the same second study. The
    ///     operators are chosen so that the first study consumes the same random numbers whatever its objective
    ///     values (uniform crossover, no rejection loop), so both instances enter the second study with identical
    ///     random states and must produce bit-identical second studies. Before the fix they did not: each carried
    ///     its own first study's survivors, population, optimizer or latent points into the second (or crashed).
    /// </summary>
    /// <remarks>
    ///     RVEA is checked separately (<see cref="ReusedRvea_SelectsTheSecondStudysParentsLikeAFreshSampler" />):
    ///     its survivor count follows the objective values, and with it how many random numbers its parent
    ///     selection draws, so two first studies on different objectives leave its random state in different places.
    /// </remarks>
    /// <param name="kind">Which sampler to reuse (see <see cref="MakeStateful" />).</param>
    [TestMethod]
    [DataRow("nsga2")]
    [DataRow("agemoea2")]
    [DataRow("smsemoa")]
    [DataRow("moead")]
    [DataRow("janus")]
    [DataRow("janusde")]
    [DataRow("rembo")]
    [DataRow("rembo-tpe")]
    public void ReusedSampler_SecondStudy_DoesNotDependOnTheFirstStudysValues(string kind)
    {
        string afterFirst = SecondStudyFingerprint(kind, firstVariant: 0);
        string afterOther = SecondStudyFingerprint(kind, firstVariant: 1);

        afterOther.Should().Be(afterFirst,
            "the second study must be sampled from its own history, not from the first study's");
    }

    /// <summary>
    ///     RVEA's survival operator keeps per-study state: a running ideal point and reference vectors re-scaled
    ///     every few generations. A reused <see cref="RveaSampler" /> must select a second study's parents exactly
    ///     as a new sampler replaying that study's history does, generation by generation. The first study runs
    ///     long enough to move the ideal point and adapt the vectors (every two generations here), which a shared
    ///     survival operator would carry into the second study's selections.
    /// </summary>
    [TestMethod]
    public void ReusedRvea_SelectsTheSecondStudysParentsLikeAFreshSampler()
    {
        var reused = new RveaSampler(populationSize: 6, seed: 3, adaptFreq: 0.5, nMaxGenerations: 4);
        Optuna.CreateStudy(sampler: reused, directions: TwoMinimize).Optimize(FirstObjective, nTrials: 30);
        var second = Optuna.CreateStudy(sampler: reused, directions: TwoMinimize);
        second.Optimize(SecondObjective, nTrials: 30);
        int populationSize = reused.PopulationSize!.Value;
        int lastGeneration = (second.Trials.Count - 1) / populationSize;

        // A different seed on purpose: parent selection draws no random numbers, so only the study history and the
        // survival operator's own state can decide it.
        var replay = new RveaSampler(populationSize: 6, seed: 99, adaptFreq: 0.5, nMaxGenerations: 4);
        var studyAttrs = second.Storage.GetStudySystemAttrs(second.StudyId);
        lastGeneration.Should().BeGreaterThanOrEqualTo(3, "the comparison must span an adaptation");
        for (int generation = 1; generation <= lastGeneration; generation++)
        {
            // In generation order, as the reused sampler selected them, so the replay's survival state evolves the same way.
            var expected = replay.SelectParent(second, generation).Select(t => t.TrialId).OrderBy(id => id);
            var actual = ((IEnumerable<object?>)studyAttrs[reused.ParentCacheKeyPrefix + generation])
                .Select(id => Convert.ToInt32(id, CultureInfo.InvariantCulture))
                .OrderBy(id => id);
            actual.Should().Equal(expected, $"generation {generation}'s parents come from the second study's own survival state");
        }
    }

    /// <summary>
    ///     <see cref="ConditionalParams" /> memoizes the latent values of the last parent population. Its key was
    ///     the parents' trial numbers alone, and trial numbers repeat in every study, so a rule set reused (with its
    ///     sampler) for a second study resumed that study's reactivated branches from the first study's values.
    ///     The key now includes the study and each parent's identity.
    /// </summary>
    [TestMethod]
    public void ConditionalParams_DormantCarry_ReadsTheCurrentStudysParents()
    {
        var rules = new ConditionalParams { PreserveDormantValues = true }.When("opt", "a").Activates("p");
        var searchSpace = new Dictionary<string, BaseDistribution>
        {
            ["opt"] = new CategoricalDistribution(new CategoricalChoice[] { "a", "b" }),
            ["p"] = new FloatDistribution(0, 1),
        };

        double CarriedValue(double parentValue)
        {
            // A fresh in-memory study each time: the parent is trial 0 (trial id 0) in both studies.
            var study = Optuna.CreateStudy(sampler: new RandomSampler(0));
            study.AddTrial(FrozenTrial.CreateTrial(
                value: 1.0,
                @params: new OrderedDict<string, object?> { ["opt"] = "a", ["p"] = parentValue },
                distributions: new OrderedDict<string, BaseDistribution>
                {
                    ["opt"] = searchSpace["opt"], ["p"] = searchSpace["p"],
                }));
            var parents = study.Trials;
            var running = study.Ask();
            var child = new OrderedDict<string, object?> { ["opt"] = "a" };

            rules.ApplyDormantCarry(study, study.Storage.GetTrial(running.TrialId), child, parents, searchSpace);
            return Convert.ToDouble(child["p"], CultureInfo.InvariantCulture);
        }

        CarriedValue(0.25).Should().Be(0.25);
        CarriedValue(0.75).Should().Be(0.75, "the second study's parent, not the first study's, supplies the latent value");
    }

    /// <summary>
    ///     <see cref="PerStudyState{TState}" /> keys a state by storage object and study id: two in-memory studies
    ///     that both have study id 0 get separate states, two <see cref="Study" /> objects opened on the same
    ///     in-memory study share one, and two studies of one storage get one each.
    /// </summary>
    [TestMethod]
    public void PerStudyState_KeysByStorageObjectAndStudyId()
    {
        var states = new PerStudyState<List<int>>();
        var first = Optuna.CreateStudy(studyName: "a");
        var second = Optuna.CreateStudy(studyName: "a");
        var sharedStorage = new InMemoryStorage();
        var x = Optuna.CreateStudy(storage: sharedStorage, studyName: "x");
        var xAgain = Optuna.LoadStudy(studyName: "x", storage: sharedStorage);
        var y = Optuna.CreateStudy(storage: sharedStorage, studyName: "y");

        first.StudyId.Should().Be(second.StudyId, "the collision being guarded against");
        states.Resolve(first, () => new List<int>())
            .Should().NotBeSameAs(states.Resolve(second, () => new List<int>()));
        states.Resolve(x, () => new List<int>())
            .Should().BeSameAs(states.Resolve(xAgain, () => new List<int>()));
        states.Resolve(x, () => new List<int>())
            .Should().NotBeSameAs(states.Resolve(y, () => new List<int>()));
    }

    /// <summary>
    ///     A state is replaced when its caller reports it stale (<c>isCurrent</c> answers <c>false</c>) and kept
    ///     otherwise; the replacement is what later calls get.
    /// </summary>
    [TestMethod]
    public void PerStudyState_ReplacesAStateReportedStale()
    {
        var states = new PerStudyState<List<int>>();
        var study = Optuna.CreateStudy();

        List<int> original = states.Resolve(study, () => new List<int> { 1 });
        List<int> kept = states.Resolve(study, () => new List<int> { 2 }, _ => true);
        List<int> replaced = states.Resolve(study, () => new List<int> { 3 }, _ => false);

        kept.Should().BeSameAs(original);
        replaced.Should().Equal(3);
        states.Resolve(study, () => new List<int> { 4 }).Should().BeSameAs(replaced);
    }

    /// <summary>
    ///     <see cref="TrialAnchor" /> recognizes the anchored trial by number, storage id and start time: a live
    ///     study keeps matching (also once a waiting trial gets its start time), while a recreated study's trial
    ///     with the same number does not, whether its storage id differs or only its start time does (SQLite reuses
    ///     both study and trial ids). A list without the anchored number does not contain it.
    /// </summary>
    [TestMethod]
    public void TrialAnchor_RecognizesOnlyTheAnchoredTrial()
    {
        var start = new DateTime(2026, 10, 27, 12, 0, 0, DateTimeKind.Utc);
        FrozenTrial Trial(int number, int trialId, DateTime? datetimeStart) => new(
            number, datetimeStart is null ? TrialState.Waiting : TrialState.Complete, datetimeStart is null ? null : 1.0,
            datetimeStart, datetimeStart, new OrderedDict<string, object?>(), new OrderedDict<string, BaseDistribution>(),
            new OrderedDict<string, object?>(), new OrderedDict<string, object?>(), new OrderedDict<int, double>(), trialId);

        var anchor = TrialAnchor.Of(Trial(0, 7, start));
        anchor.IsIn(new[] { Trial(0, 7, start), Trial(1, 8, start) }).Should().BeTrue("same trial, full list");
        anchor.IsIn(new[] { Trial(3, 10, start), Trial(0, 7, start) }).Should().BeTrue("same trial, unordered subset");
        anchor.IsIn(new[] { Trial(0, 9, start) }).Should().BeFalse("another storage id under the same number");
        anchor.IsIn(new[] { Trial(0, 7, start.AddSeconds(5)) }).Should().BeFalse("a reused id, started later");
        anchor.IsIn(new[] { Trial(1, 8, start) }).Should().BeFalse("the anchored number is missing");
        anchor.IsIn(Array.Empty<FrozenTrial>()).Should().BeFalse("an empty study has no trial 0");

        var waiting = TrialAnchor.Of(Trial(0, 7, null));
        waiting.IsIn(new[] { Trial(0, 7, start) }).Should().BeTrue("a waiting trial that started since is the same trial");
    }

    // ---- objectives -------------------------------------------------------------------------------

    /// <summary>The first study's two-objective problem over <c>x, y ∈ [0, 1]</c>.</summary>
    /// <param name="trial">The trial suggesting the parameters.</param>
    /// <returns>Both objective values.</returns>
    private static IReadOnlyList<double> FirstObjective(Trial trial)
    {
        double x = trial.SuggestFloat("x", 0, 1);
        double y = trial.SuggestFloat("y", 0, 1);
        return new[] { x, 1 - y };
    }

    /// <summary>A different two-objective problem over the same parameters, with different values everywhere.</summary>
    /// <param name="trial">The trial suggesting the parameters.</param>
    /// <returns>Both objective values.</returns>
    private static IReadOnlyList<double> OtherFirstObjective(Trial trial)
    {
        double x = trial.SuggestFloat("x", 0, 1);
        double y = trial.SuggestFloat("y", 0, 1);
        return new[] { (x - 0.3) * (x - 0.3), (y - 0.6) * (y - 0.6) + 0.1 };
    }

    /// <summary>The second study's two-objective problem over the same parameters.</summary>
    /// <param name="trial">The trial suggesting the parameters.</param>
    /// <returns>Both objective values.</returns>
    private static IReadOnlyList<double> SecondObjective(Trial trial)
    {
        double x = trial.SuggestFloat("x", 0, 1);
        double y = trial.SuggestFloat("y", 0, 1);
        return new[] { x + y, (1 - x) * (1 + y) };
    }

    /// <summary>
    ///     A single-objective problem over <c>a, b, c ∈ [-3, 3]</c>; <paramref name="variant" /> selects one of three
    ///     unrelated surfaces (0 and 1 for the first study, 2 for the second).
    /// </summary>
    /// <param name="trial">The trial suggesting the parameters.</param>
    /// <param name="variant">Which surface to evaluate.</param>
    /// <returns>The objective value.</returns>
    private static double SingleObjective(Trial trial, int variant)
    {
        double a = trial.SuggestFloat("a", -3, 3);
        double b = trial.SuggestFloat("b", -3, 3);
        double c = trial.SuggestFloat("c", -3, 3);
        return variant switch
        {
            0 => a * a + b * b + c * c,
            1 => Math.Abs(a - 1) + Math.Abs(b + 2) + 3 * Math.Abs(c),
            _ => (a - 0.5) * (a - 0.5) + (b + 0.5) * (b + 0.5) + (c - 1) * (c - 1),
        };
    }

    // ---- runs -------------------------------------------------------------------------------------

    /// <summary>
    ///     Builds a fresh, seeded <paramref name="kind" /> sampler, runs a first study on objective
    ///     <paramref name="firstVariant" />, then a second study on the shared second objective, and returns the
    ///     second study's parameters as raw IEEE bits, one line per trial.
    /// </summary>
    /// <param name="kind">Which sampler to build (see <see cref="MakeStateful" />).</param>
    /// <param name="firstVariant">0 or 1: which first-study objective to run.</param>
    /// <returns>A bit-exact fingerprint of the second study's sampled parameters.</returns>
    private static string SecondStudyFingerprint(string kind, int firstVariant)
    {
        (BaseSampler sampler, bool multiObjective, int firstTrials, int secondTrials) = MakeStateful(kind);
        Study second;
        if (multiObjective)
        {
            Optuna.CreateStudy(sampler: sampler, directions: TwoMinimize)
                .Optimize(firstVariant == 0 ? FirstObjective : OtherFirstObjective, nTrials: firstTrials);
            second = Optuna.CreateStudy(sampler: sampler, directions: TwoMinimize);
            second.Optimize(SecondObjective, nTrials: secondTrials);
        }
        else
        {
            Optuna.CreateStudy(sampler: sampler).Optimize(t => SingleObjective(t, firstVariant), nTrials: firstTrials);
            second = Optuna.CreateStudy(sampler: sampler);
            second.Optimize(t => SingleObjective(t, 2), nTrials: secondTrials);
        }

        second.Trials.Should().OnlyContain(t => t.State == TrialState.Complete);
        return string.Join("\n", second.Trials.Select(t => string.Join(",", t.Params
            .OrderBy(p => p.Key, StringComparer.Ordinal)
            .Select(p => $"{p.Key}={BitConverter.DoubleToInt64Bits(Convert.ToDouble(p.Value, CultureInfo.InvariantCulture))}"))));
    }

    /// <summary>Builds a seeded genetic sampler of the given kind with a population of 6.</summary>
    /// <param name="kind">The DataRow name of the sampler.</param>
    /// <returns>A fresh sampler instance.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The kind is not one of the covered samplers.</exception>
    private static BaseGASampler MakeGa(string kind) => kind switch
    {
        "nsga2" => new NSGAIISampler(populationSize: 6, seed: 1),
        "nsga3" => new NSGAIIISampler(populationSize: 6, seed: 1),
        "nsga3cpt" => new NSGAIIICptSampler(populationSize: 6, seed: 1),
        "agemoea2" => new AgeMoea2Sampler(populationSize: 6, seed: 1),
        "agemoea2cpt" => new AgeMoea2CptSampler(populationSize: 6, seed: 1),
        "rvea" => new RveaSampler(populationSize: 6, seed: 1),
        "janus-moo-cmix" => new JanusMooCmixSampler(populationSize: 6, seed: 1, cmixStartTrial: 12),
        "janus-agemoea2-cmix" => new JanusAgeMoea2CmixSampler(populationSize: 6, seed: 1, cmixStartTrial: 12),
        "janus-agemoea2-cmix-cpt" => new JanusAgeMoea2CmixCptSampler(populationSize: 6, seed: 1, cmixStartTrial: 12),
        _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Unknown genetic sampler."),
    };

    /// <summary>
    ///     Builds a seeded sampler whose first-study random-number consumption does not depend on objective values,
    ///     with the study lengths used by <see cref="ReusedSampler_SecondStudy_DoesNotDependOnTheFirstStudysValues" />.
    ///     Crossovers are uniform (children copy parent values, so no out-of-bounds retry loop draws extra numbers).
    /// </summary>
    /// <param name="kind">The DataRow name of the sampler.</param>
    /// <returns>The sampler, whether it solves the two-objective problems, and the two study lengths.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The kind is not one of the covered samplers.</exception>
    private static (BaseSampler Sampler, bool MultiObjective, int FirstTrials, int SecondTrials) MakeStateful(string kind)
        => kind switch
        {
            "nsga2" => (new NSGAIISampler(populationSize: 6, seed: 3), true, 30, 24),
            "agemoea2" => (new AgeMoea2Sampler(populationSize: 6, seed: 3, crossover: new UniformCrossover()), true, 30, 24),
            "smsemoa" => (new SmsEmoaSampler(populationSize: 4, seed: 3, crossover: new UniformCrossover()), true, 24, 20),
            "moead" => (new MoeaDSampler(populationSize: 6, nNeighbors: 3, seed: 3, crossover: new UniformCrossover()), true, 30, 24),
            "janus" => (new JanusSampler(seed: 3), false, 24, 20),
            "janusde" => (new JanusDeSampler(seed: 3), false, 24, 20),
            "rembo" => (new RandomEmbeddingSampler(nEmbeddingDims: 2, nStartupTrials: 3, seed: 3), false, 8, 8),
            "rembo-tpe" => (new RandomEmbeddingSampler(nEmbeddingDims: 2, innerSampler: new TPESampler(seed: 5), seed: 3), false, 16, 16),
            _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, "Unknown stateful sampler."),
        };
}
