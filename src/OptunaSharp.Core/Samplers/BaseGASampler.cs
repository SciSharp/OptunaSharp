using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace OptunaSharp.Samplers;

/// <summary>
///     Base class for Genetic Algorithm (GA) samplers. Port of
///     <c>optuna/samplers/_ga/_base.py: BaseGASampler</c>.
///
///     GA samplers generate new trials by mimicking natural selection, using generations and
///     populations to iteratively improve solutions. This base class defines the interface and
///     provides utility methods for managing generations and populations. Parent selection is
///     handled by <see cref="SelectParent" />, which subclasses implement.
/// </summary>
public abstract class BaseGASampler : BaseSampler
{
    private int? _populationSize;

    protected BaseGASampler(int? populationSize)
    {
        _populationSize = populationSize;
    }

    // Python __init_subclass__ sets these per subclass to f"{cls.__name__}:...". GetType().Name at
    // runtime is the concrete subclass name, so "NSGAIISampler:generation" etc. — byte-identical.
    internal string GenerationKey => $"{GetType().Name}:generation";
    internal string ParentCacheKeyPrefix => $"{GetType().Name}:parent:";

    /// <summary>Number of individuals (trials) in a generation.</summary>
    public int? PopulationSize
    {
        get => _populationSize;
        set => _populationSize = value;
    }

    /// <summary>
    ///     Select parent trials from the population for the given generation. Called once per
    ///     generation; the result is cached in the study system attributes.
    /// </summary>
    public abstract List<FrozenTrial> SelectParent(Study study, int generation);

    /// <summary>
    ///     Get the generation number of the given trial. If not already set in the trial's system
    ///     attributes, it is calculated (from the max generation of all completed trials) and stored.
    /// </summary>
    public int GetTrialGeneration(Study study, FrozenTrial trial)
    {
        var stored = trial.SystemAttrsDict.GetValueOrDefault(GenerationKey);
        if (stored is not null)
            return Convert.ToInt32(stored, CultureInfo.InvariantCulture);

        var trials = study.GetTrialsInternal(
            deepcopy: false, states: new[] { TrialState.Complete }, useCache: true);

        int maxGeneration = 0, maxGenerationCount = 0;
        for (int i = trials.Count - 1; i >= 0; i--)
        {
            var raw = trials[i].SystemAttrsDict.GetValueOrDefault(GenerationKey);
            int generation = raw is null ? -1 : Convert.ToInt32(raw, CultureInfo.InvariantCulture);

            if (generation < maxGeneration)
                continue;
            if (generation > maxGeneration)
            {
                maxGeneration = generation;
                maxGenerationCount = 1;
            }
            else
            {
                maxGenerationCount += 1;
            }
        }

        System.Diagnostics.Debug.Assert(_populationSize is not null, "Population size must be set.");
        int result = maxGenerationCount < _populationSize ? maxGeneration : maxGeneration + 1;
        study.Storage.SetTrialSystemAttr(trial.TrialId, GenerationKey, result);
        return result;
    }

    /// <summary>Get the population (completed trials) of the given generation.</summary>
    public List<FrozenTrial> GetPopulation(Study study, int generation)
    {
        var result = new List<FrozenTrial>();
        foreach (var trial in study.GetTrialsInternal(
                     deepcopy: false, states: new[] { TrialState.Complete }, useCache: true))
        {
            var raw = trial.SystemAttrsDict.GetValueOrDefault(GenerationKey);
            if (raw is not null && Convert.ToInt32(raw, CultureInfo.InvariantCulture) == generation)
                result.Add(trial);
        }
        return result;
    }

    /// <summary>
    ///     Get the parent population of the given generation (cached in the study system attributes).
    ///     Returns an empty list when <paramref name="generation" /> is 0.
    /// </summary>
    public List<FrozenTrial> GetParentPopulation(Study study, int generation)
    {
        if (generation == 0)
            return new List<FrozenTrial>();

        var studySystemAttrs = study.Storage.GetStudySystemAttrs(study.StudyId);
        var cached = studySystemAttrs.GetValueOrDefault(
            ParentCacheKeyPrefix + generation.ToString(CultureInfo.InvariantCulture));

        if (cached is not null)
        {
            var ids = new HashSet<int>(
                ((IEnumerable<object?>)cached).Select(x => Convert.ToInt32(x, CultureInfo.InvariantCulture)));
            var trials = study.GetTrialsInternal(deepcopy: false);
            return trials.Where(t => ids.Contains(t.TrialId)).ToList();
        }

        var parentPopulation = SelectParent(study, generation);
        study.Storage.SetStudySystemAttr(
            study.StudyId,
            ParentCacheKeyPrefix + generation.ToString(CultureInfo.InvariantCulture),
            parentPopulation.Select(t => (object?)t.TrialId).ToList());
        return parentPopulation;
    }
}
