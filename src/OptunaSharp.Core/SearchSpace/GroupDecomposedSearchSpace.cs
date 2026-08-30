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

    public List<OrderedDict<string, BaseDistribution>> SearchSpaces => _searchSpaces;

    public void AddDistributions(OrderedDict<string, BaseDistribution> distributions)
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

    public GroupDecomposedSearchSpace(bool includePruned = false)
    {
        _includePruned = includePruned;
    }

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

        foreach (var trial in study.GetTrialsInternal(deepcopy: false, states: statesOfInterest, useCache: useCache))
            _searchSpace.AddDistributions(trial.DistributionsDict);

        return _searchSpace.DeepCopy();
    }
}
