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
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html">Official Optuna HyperbandPruner reference.</seealso>
public class HyperbandPruner : BasePruner
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.pruners._hyperband");

    private readonly long _minResource;
    private long? _maxResource;
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
    /// <summary>Configures resource brackets whose successive-halving competitions have separate trial histories.</summary>
    /// <exception cref="NumSharp.ValueError">Positive bootstrap is combined with automatic maximum resource.</exception>
    public HyperbandPruner(
        long minResource = 1,
        long? maxResource = null,
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

    /// <inheritdoc />
    /// <exception cref="NumSharp.RuntimeError">The study has multiple objectives and the pruning rule requests its scalar direction.</exception>
    /// <exception cref="NumSharp.ValueError">Deferred bracket initialization rejects invalid resource, reduction, or bootstrap settings.</exception>
    /// <exception cref="System.InvalidOperationException">Initialized bracket allocation budgets are inconsistent.</exception>
    public override bool Prune(Study study, FrozenTrial trial)
    {
        if (_pruners.Count == 0)
        {
            // Automatic maximum resource depends on completed reports; until available, there is
            // no stable bracket schedule under which this trial could fairly be pruned.
            TryInitialization(study);
            if (_pruners.Count == 0)
                return false;
        }

        int bracketId = GetBracketId(study, trial);
        _logger.Debug($"{bracketId}th bracket is selected");
        // Compete only against this bracket's trials, whose minimum-resource schedules match.
        var bracketStudy = CreateBracketStudy(study, bracketId);
        return _pruners[bracketId].Prune(bracketStudy, trial);
    }

    /// <inheritdoc />
    /// <exception cref="System.InvalidOperationException">Initialized bracket allocation budgets are inconsistent.</exception>
    internal override Study FilterStudy(Study study, FrozenTrial trial)
        => CreateBracketStudy(study, GetBracketId(study, trial));

    /// <summary>Initializes Hyperband brackets once the maximum resource can be determined.</summary>
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <exception cref="NumSharp.ValueError">A successive-halving bracket rejects the configured resource, reduction, or bootstrap settings.</exception>
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
    /// <param name="bracketId">Zero-based Hyperband bracket index.</param>
    /// <returns>The integer allocation weight for the bracket.</returns>
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
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="trial">Trial snapshot being examined; copy its contents before modifying them.</param>
    /// <returns>The zero-based bracket assigned to the trial, or zero before initialization.</returns>
    /// <exception cref="System.InvalidOperationException">Initialized bracket budgets fail to cover the computed allocation index.</exception>
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
    /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
    /// <param name="bracketId">Zero-based Hyperband bracket index.</param>
    /// <returns>A view sharing study storage and exposing only trials assigned to the requested bracket.</returns>
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

        /// <summary>Creates a view sharing the original study storage but restricting public trial history to one bracket.</summary>
        /// <param name="study">Study supplying trial history, objective directions, and storage.</param>
        /// <param name="pruner">Hyperband pruner assigning trials to brackets.</param>
        /// <param name="bracketId">Zero-based Hyperband bracket index.</param>
        internal BracketStudy(Study study, HyperbandPruner pruner, int bracketId)
            : base(studyName: study.StudyName, storage: study.Storage, sampler: study.Sampler, pruner: pruner)
        {
            _study = study;
            _bracketId = bracketId;
        }

        /// <inheritdoc />
        public override List<FrozenTrial> GetTrials(
            bool deepcopy = true, IReadOnlyCollection<TrialState>? states = null)
        {
            var trials = GetTrialsInternal(deepcopy: deepcopy, states: states);
            var pruner = (HyperbandPruner)Pruner;
            return trials.Where(t => pruner.GetBracketId(this, t) == _bracketId).ToList();
        }

        /// <inheritdoc />
    /// <exception cref="NumSharp.RuntimeError">The original study is not currently executing an optimization loop.</exception>
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

    /// <summary>Builds the byte lookup table for the reflected IEEE CRC-32 polynomial.</summary>
    /// <returns>The 256-entry reflected CRC lookup table.</returns>
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

    /// <summary>Computes the zlib-compatible CRC-32 checksum used for deterministic bracket assignment.</summary>
    /// <param name="data">Bytes whose CRC checksum is computed.</param>
    /// <returns>The unsigned CRC-32 checksum.</returns>
    internal static uint Compute(byte[] data)
    {
        uint crc = 0xFFFFFFFFu;
        foreach (byte b in data)
            crc = _table[(crc ^ b) & 0xFF] ^ (crc >> 8);
        return crc ^ 0xFFFFFFFFu;
    }
}
