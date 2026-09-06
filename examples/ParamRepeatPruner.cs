#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// ParamRepeatPruner — an OptunaSharp port of aisu-wata0's community gist (6★):
//   https://gist.github.com/aisu-wata0/24045ab5e8ad007b4a09f708adfe359f
//   "Class that prunes trials with the same parameters as past trials."
//
// A userland helper (not part of Core): inside the objective it prunes a trial whose parameters
// exactly repeat an already-seen trial's, so a sampler that re-proposes the same point doesn't waste
// an evaluation. `repeatsMax > 0` allows a few repeats (useful for noisy objectives) and
// `ValueOfRepeats` averages them instead of pruning.
//
// The gist drives it with TPESampler(seed=42); this example uses RandomSampler(seed) so the run is
// reproducible AND byte-for-byte checked against the Python gist in
// test/OptunaSharp.Tests.Interop/ParamRepeatPrunerParityTests.cs (TPE over an int parameter with
// repeats isn't byte-comparable — NumPy simd-sort tie order — so parity is asserted on the RNG-exact
// path). Over dummy_param-0 ∈ [1,20] with 40 trials, repeats are inevitable and get pruned.
//
//   Run:  dotnet run examples/ParamRepeatPruner.cs

using System.Linq;
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

var study = Optuna.CreateStudy(
    storage: new InMemoryStorage(),
    sampler: new RandomSampler(seed: 0),
    direction: StudyDirection.Minimize,
    studyName: "param-repeat");

var pruner = new ParamRepeatPruner(study);

// dummy_param-0 has only 20 possible values, so across 40 trials the sampler is forced to repeat some —
// each repeat is pruned, so only the ~unique values ever "complete".
study.Optimize(t =>
{
    long p = t.SuggestInt("dummy_param-0", 1, 20);   // the one parameter — a small integer space
    pruner.CheckParams(t);                           // prune this trial if its params repeat an earlier completed one
    return p;                                        // trivial objective — the point is the repeat-pruning, not the value
}, nTrials: 40);

int completed = study.Trials.Count(t => t.State == TrialState.Complete);
int pruned = study.Trials.Count(t => t.State == TrialState.Pruned);
Console.WriteLine($"Trials: {study.Trials.Count}  |  completed (unique params): {completed}  |  pruned (repeats): {pruned}");
Console.WriteLine($"Best value: {study.BestValue}  params: {{ dummy_param-0 = {study.BestParams["dummy_param-0"]} }}");

/// <summary>
///     Prunes trials whose parameters repeat an already-seen trial's — a faithful port of the gist's
///     class. Mirrored in the interop parity test (the two are intentionally standalone copies; this
///     is a userland pattern, not Core).
/// </summary>
public sealed class ParamRepeatPruner
{
    private readonly Study _study;
    private readonly int _repeatsMax;
    private readonly HashSet<TrialState> _compareStates;
    private readonly Dictionary<int, List<int>> _repeats = new();

    public ParamRepeatPruner(Study study, int repeatsMax = 0, IEnumerable<TrialState>? shouldCompareStates = null)
    {
        _study = study;
        _repeatsMax = repeatsMax;
        _compareStates = new HashSet<TrialState>(shouldCompareStates ?? new[] { TrialState.Complete });
    }

    /// <summary>Returns the number of the trial this one repeats (-1 if none); prunes past repeatsMax.</summary>
    public int CheckParams(Trial current)
    {
        int repeatedNumber = -1;
        foreach (var past in _study.GetTrials(deepcopy: false))
        {
            if (past.Number == current.Number)
                continue;                                   // ignore the running trial itself
            if (!_compareStates.Contains(past.State))
                continue;                                   // only compare against the chosen states (default: COMPLETE)
            if (ParamsEqual(current.Params, past.Params))
            {
                repeatedNumber = past.Number;
                break;
            }
        }

        // a match was found — record this repeat under the original trial, and prune past the allowance
        if (repeatedNumber > -1)
        {
            if (!_repeats.TryGetValue(repeatedNumber, out var seen))
                _repeats[repeatedNumber] = seen = new List<int>();   // first repeat seen of that original
            seen.Add(current.Number);
            if (seen.Count > _repeatsMax)                            // repeatsMax = 0 → prune every repeat
                throw new TrialPruned();
        }
        return repeatedNumber;
    }

    /// <summary>Aggregate (default: mean) an original trial's value with its repeats' — the gist's "allow repeats" path.</summary>
    public double ValueOfRepeats(int repeatedNumber, Func<IReadOnlyList<double>, double>? aggregate = null)
    {
        aggregate ??= values => values.Average();
        var byNumber = _study.GetTrials(deepcopy: false).ToDictionary(t => t.Number);
        var collected = new List<double>();
        if (byNumber[repeatedNumber].Value is { } original)
            collected.Add(original);
        if (_repeats.TryGetValue(repeatedNumber, out var seen))
            foreach (int n in seen)
                if (byNumber[n].Value is { } v)
                    collected.Add(v);
        return aggregate(collected);
    }

    private static bool ParamsEqual(IReadOnlyDictionary<string, object?> a, IReadOnlyDictionary<string, object?> b)
    {
        if (a.Count != b.Count)
            return false;
        foreach (var (key, av) in a)
            if (!b.TryGetValue(key, out var bv) || !Equals(av, bv))
                return false;
        return true;
    }
}
