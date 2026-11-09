using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;

namespace OptunaSharp.SearchSpace;

/// <summary>Port of <c>optuna/search_space/group_decomposed.py: _SearchSpaceGroup</c>.</summary>
internal sealed class SearchSpaceGroup
{
    private List<OrderedDict<string, BaseDistribution>> _searchSpaces = new();

    /// <summary>The disjoint parameter-distribution groups inferred from observed trial co-occurrence.</summary>
    public List<OrderedDict<string, BaseDistribution>> SearchSpaces => _searchSpaces;

    /// <summary>Splits existing groups so every resulting group has the same presence pattern in the incoming trial.</summary>
    /// <param name="distributions">
    ///     The parameter distributions being incorporated into the decomposed search space; only read, so a compact
    ///     finished trial's read-only view (plan 13, item S12) serves as well as a dictionary.
    /// </param>
    public void AddDistributions(IReadOnlyDictionary<string, BaseDistribution> distributions)
    {
        var distKeys = new HashSet<string>(distributions.Keys);
        var nextSearchSpaces = new List<OrderedDict<string, BaseDistribution>>();

        foreach (var searchSpace in _searchSpaces)
        {
            var keys = new HashSet<string>(searchSpace.Keys);

            var intersection = new OrderedDict<string, BaseDistribution>();
            var difference = new OrderedDict<string, BaseDistribution>();
            foreach (var (name, dist) in searchSpace)
            {
                if (distKeys.Contains(name))
                    intersection[name] = dist;
                else
                    difference[name] = dist;
            }
            nextSearchSpaces.Add(intersection);
            nextSearchSpaces.Add(difference);

            distKeys.ExceptWith(keys);
        }

        var remainder = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, dist) in distributions)
        {
            if (distKeys.Contains(name))
                remainder[name] = dist;
        }
        nextSearchSpaces.Add(remainder);

        _searchSpaces = nextSearchSpaces.Where(searchSpace => searchSpace.Count > 0).ToList();
    }

    /// <summary>Returns a new group containing independently copied distribution dictionaries.</summary>
    /// <returns>A new group containing independently copied distribution dictionaries.</returns>
    internal SearchSpaceGroup DeepCopy()
    {
        var copy = new SearchSpaceGroup();
        foreach (var searchSpace in _searchSpaces)
            copy._searchSpaces.Add(searchSpace.ShallowClone());
        return copy;
    }
}

/// <summary>Port of <c>optuna/search_space/group_decomposed.py: _GroupDecomposedSearchSpace</c>.</summary>
internal sealed class GroupDecomposedSearchSpace
{
    private readonly SearchSpaceGroup _searchSpace = new();
    private int? _studyId;
    private readonly bool _includePruned;

    /// <summary>Creates an incremental co-occurrence decomposition for one study's parameter distributions.</summary>
    /// <param name="includePruned">Whether pruned trials contribute their recorded parameter distributions.</param>
    public GroupDecomposedSearchSpace(bool includePruned = false)
    {
        _includePruned = includePruned;
    }

    /// <summary>Returns the cached decomposition of the study's observed parameter distributions.</summary>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="useCache">Whether the study may reuse its cached trial snapshots for this calculation.</param>
    /// <returns>The cached decomposition of the study's observed parameter distributions.</returns>
    /// <exception cref="NumSharp.ValueError">This cache is reused with a different study identifier.</exception>
    public SearchSpaceGroup Calculate(Study study, bool useCache = false)
    {
        if (_studyId is null)
        {
            _studyId = study.StudyId;
        }
        else
        {
            // Note that the check below is meaningless when InMemoryStorage is used because
            // InMemoryStorage.CreateNewStudy always returns the same study ID.
            if (_studyId != study.StudyId)
                throw new ValueError("`_GroupDecomposedSearchSpace` cannot handle multiple studies.");
        }

        var statesOfInterest = _includePruned
            ? new[] { TrialState.Complete, TrialState.Pruned }
            : new[] { TrialState.Complete };

        // The same trials in the same (number) order as study._get_trials(deepcopy=False, states, use_cache), filtered
        // here instead of in a copied list: on an in-memory storage the read is the zero-copy view (plan 13, item S3).
        foreach (var trial in study.GetTrialsForSampler(useCache))
        {
            if (System.Array.IndexOf(statesOfInterest, trial.State) >= 0)
                _searchSpace.AddDistributions(trial.Distributions);
        }

        return _searchSpace.DeepCopy();
    }
}
