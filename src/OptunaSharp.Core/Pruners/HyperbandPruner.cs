using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NumSharp;
using OptunaSharp.Logging;

namespace OptunaSharp.Pruners;

/// <summary>
///     Pruner using Hyperband. Port of <c>optuna/pruners/_hyperband.py: HyperbandPruner</c>.
///
///     As SuccessiveHalving (SHA) requires the number of configurations <c>n</c> as its
///     hyperparameter, for a given finite budget <c>B</c> all the configurations have the
///     resources of <c>B/n</c> on average. Hyperband attacks this trade-off by trying different
///     <c>n</c> values for a fixed budget. Hyperband runs multiple
///     <see cref="SuccessiveHalvingPruner" />s ("brackets"); the number of brackets is
///     <c>floor(log_eta(maxResource/minResource)) + 1</c>.
///
///     <c>HyperbandPruner</c> computes the bracket ID for each trial from
///     <c>study_name</c> and the trial number (CRC-32), so specify <c>studyName</c> to make the
///     pruning algorithm reproducible.
/// </summary>
public class HyperbandPruner : BasePruner
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.pruners._hyperband");

    private readonly int _minResource;
    private int? _maxResource;
    private readonly bool _maxResourceIsAuto;
    private readonly int _reductionFactor;
    private readonly List<SuccessiveHalvingPruner> _pruners = new();
    private readonly int _bootstrapCount;
    private int _totalTrialAllocationBudget;
    private readonly List<int> _trialAllocationBudgets = new();
    private int? _nBrackets;

    /// <param name="minResource">
    ///     The minimum resource allocated to a trial (<c>r</c> in the paper). See
    ///     <see cref="SuccessiveHalvingPruner" />.
    /// </param>
    /// <param name="maxResource">
    ///     The maximum resource allocated to a trial, or <c>null</c> for Optuna's <c>"auto"</c>:
    ///     estimated from the first completed trial's largest reported step (+1); no trials are
    ///     pruned until determined. This value should match the maximum iteration steps.
    /// </param>
    /// <param name="reductionFactor">Reduction factor of promotable trials (<c>eta</c> in the paper).</param>
    /// <param name="bootstrapCount">
    ///     Number of trials required in a rung before any trial can be promoted. Incompatible
    ///     with automatic <c>maxResource</c>.
    /// </param>
    public HyperbandPruner(
        int minResource = 1,
        int? maxResource = null,
        int reductionFactor = 3,
        int bootstrapCount = 0)
    {
        _minResource = minResource;
        _maxResource = maxResource;
        _maxResourceIsAuto = maxResource is null;
        _reductionFactor = reductionFactor;
        _bootstrapCount = bootstrapCount;
        _totalTrialAllocationBudget = 0;
        _nBrackets = null;

        if (_bootstrapCount > 0 && _maxResourceIsAuto)
            throw new ValueError(
                "bootstrap_count > 0 and max_resource == 'auto' " +
                $"are mutually incompatible, bootstrap_count is {_bootstrapCount}");
    }

    public override bool Prune(Study study, FrozenTrial trial)
    {
        if (_pruners.Count == 0)
        {
            TryInitialization(study);
            if (_pruners.Count == 0)
                return false;
        }

        int bracketId = GetBracketId(study, trial);
        _logger.Debug($"{bracketId}th bracket is selected");
        var bracketStudy = CreateBracketStudy(study, bracketId);
        return _pruners[bracketId].Prune(bracketStudy, trial);
    }

    internal override Study FilterStudy(Study study, FrozenTrial trial)
        => CreateBracketStudy(study, GetBracketId(study, trial));

    private void TryInitialization(Study study)
    {
        if (_maxResource is null)
        {
            var trials = study.GetTrials(deepcopy: false, states: new[] { TrialState.Complete });
            var nSteps = trials.Where(t => t.LastStep is not null).Select(t => t.LastStep!.Value).ToList();

            if (nSteps.Count == 0)
                return;

            _maxResource = nSteps.Max() + 1;
        }

        if (_nBrackets is null)
        {
            // n_brackets = floor(log_{reduction_factor}(max_resource / min_resource)) + 1
            // (combining the Hyperband and ASHA formulations; see the Python source).
            _nBrackets = (int)Math.Floor(
                Math.Log((double)_maxResource.Value / _minResource) / Math.Log(_reductionFactor)) + 1;
        }

        _logger.Debug($"Hyperband has {_nBrackets} brackets");

        for (int bracketId = 0; bracketId < _nBrackets.Value; bracketId++)
        {
            int trialAllocationBudget = CalculateTrialAllocationBudget(bracketId);
            _totalTrialAllocationBudget += trialAllocationBudget;
            _trialAllocationBudgets.Add(trialAllocationBudget);

            var pruner = new SuccessiveHalvingPruner(
                minResource: _minResource,
                reductionFactor: _reductionFactor,
                minEarlyStoppingRate: bracketId,
                bootstrapCount: _bootstrapCount);
            _pruners.Add(pruner);
        }
    }

    /// <summary>
    ///     Compute the trial allocated budget for a bracket (Python
    ///     <c>_calculate_trial_allocation_budget</c>): the ratio of the number of trials since
    ///     the total is unknown in Optuna's learning scheme.
    /// </summary>
    private int CalculateTrialAllocationBudget(int bracketId)
    {
        System.Diagnostics.Debug.Assert(_nBrackets is not null);
        int s = _nBrackets!.Value - 1 - bracketId;
        return (int)Math.Ceiling(_nBrackets.Value * Math.Pow(_reductionFactor, s) / (s + 1));
    }

    /// <summary>
    ///     Compute the index of bracket for a trial (Python <c>_get_bracket_id</c>) — CRC-32 of
    ///     <c>"{study_name}_{trial_number}"</c> modulo the total allocation budget.
    /// </summary>
    internal int GetBracketId(Study study, FrozenTrial trial)
    {
        if (_pruners.Count == 0)
            return 0;

        System.Diagnostics.Debug.Assert(_nBrackets is not null);
        long n = Crc32.Compute(Encoding.UTF8.GetBytes($"{study.StudyName}_{trial.Number}"))
            % _totalTrialAllocationBudget;
        for (int bracketId = 0; bracketId < _nBrackets!.Value; bracketId++)
        {
            n -= _trialAllocationBudgets[bracketId];
            if (n < 0)
                return bracketId;
        }

        throw new InvalidOperationException("This line should be unreachable.");
    }

    /// <summary>Python <c>_create_bracket_study</c>.</summary>
    internal Study CreateBracketStudy(Study study, int bracketId)
        => new BracketStudy(study, this, bracketId);

    /// <summary>
    ///     A study view passed to <see cref="SuccessiveHalvingPruner.Prune" /> that filters
    ///     trials to those in the same bracket. (Python's <c>_BracketStudy</c>; the
    ///     attribute-guard <c>__getattribute__</c> has no C# analog and is intentionally not
    ///     ported — the compiler enforces the used surface.)
    /// </summary>
    private sealed class BracketStudy : Study
    {
        private readonly Study _study;
        private readonly int _bracketId;

        internal BracketStudy(Study study, HyperbandPruner pruner, int bracketId)
            : base(studyName: study.StudyName, storage: study.Storage, sampler: study.Sampler, pruner: pruner)
        {
            _study = study;
            _bracketId = bracketId;
        }

        public override List<FrozenTrial> GetTrials(
            bool deepcopy = true, IReadOnlyCollection<TrialState>? states = null)
        {
            var trials = GetTrialsInternal(deepcopy: deepcopy, states: states);
            var pruner = (HyperbandPruner)Pruner;
            return trials.Where(t => pruner.GetBracketId(this, t) == _bracketId).ToList();
        }

        public override void Stop()
        {
            // `stop` should stop the original study's optimization loop instead of the bracket's.
            _study.Stop();
        }
    }
}

/// <summary>CRC-32 (zlib/binascii polynomial), matching Python's <c>binascii.crc32</c>.</summary>
internal static class Crc32
{
    private static readonly uint[] _table = BuildTable();

    private static uint[] BuildTable()
    {
        var table = new uint[256];
        for (uint i = 0; i < 256; i++)
        {
            uint c = i;
            for (int k = 0; k < 8; k++)
                c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        return table;
    }

    internal static uint Compute(byte[] data)
    {
        uint crc = 0xFFFFFFFFu;
        foreach (byte b in data)
            crc = _table[(crc ^ b) & 0xFF] ^ (crc >> 8);
        return crc ^ 0xFFFFFFFFu;
    }
}
