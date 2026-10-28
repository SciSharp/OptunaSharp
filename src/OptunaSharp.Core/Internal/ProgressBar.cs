using System;
using System.Globalization;
using OptunaSharp.Internal;

namespace OptunaSharp.Internal;

/// <summary>
///     Port of <c>optuna/progress_bar.py: _ProgressBar</c> — a minimal, dependency-free stand-in
///     for tqdm that writes progress lines to stderr when enabled.
/// </summary>
internal sealed class ProgressBar
{
    private readonly bool _enabled;
    private readonly int? _nTrials;
    private readonly double? _timeout;
    private int _count;
    private string _postfix = string.Empty;

    // Optimize(nJobs) calls Update from every worker. tqdm takes its own lock on each update; without one the
    // unlocked `_count++` lost increments (the bar could end short of n_trials) and a line built from an older
    // count could be written after a newer one. Lock order: this lock, then the storage's (the best-trial read).
    private readonly object _updateLock = new();

    /// <summary>Configures count- or timeout-based console progress reporting for an optimization run.</summary>
    /// <param name="enabled">Whether progress output should be emitted.</param>
    /// <param name="nTrials">The trial-count limit; null leaves the count unlimited.</param>
    /// <param name="timeout">The optimization time limit in seconds; null leaves time unlimited.</param>
    internal ProgressBar(bool enabled, int? nTrials, double? timeout)
    {
        // Match optuna/progress_bar.py: _ProgressBar.__init__ — warn (on the raw `enabled`, before
        // the `_enabled` reduction below) when asked to show a bar with neither a trial count nor a
        // timeout to size it against.
        if (enabled && nTrials is null && timeout is null)
            OptunaWarnings.Warn("Progress bar won't be displayed because n_trials and timeout are None.");

        _enabled = enabled && (nTrials is not null || timeout is not null);
        _nTrials = nTrials;
        _timeout = timeout;
    }

    /// <summary>Refreshes elapsed progress and the best completed objective value when display is enabled.</summary>
    /// <param name="elapsedSeconds">The time elapsed since optimization started, in seconds.</param>
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <remarks>Safe to call from several workers at once: each call counts once and writes its line whole, in order.</remarks>
    internal void Update(double elapsedSeconds, Study study)
    {
        if (!_enabled)
            return;

        lock (_updateLock)
        {
            _count++;
            try
            {
                var bestTrial = study.GetBestTrialInternal(deepcopy: false);
                _postfix = $"Best trial: {bestTrial.Number}. Best value: {PythonRepr.FloatRepr(bestTrial.Value ?? double.NaN)}";
            }
            catch (NumSharp.ValueError)
            {
                // No feasible trials are completed yet.
            }
            catch (NumSharp.RuntimeError)
            {
                // Multi-objective study: no single best trial.
            }

            string progress;
            if (_nTrials is not null)
            {
                int percent = (int)(100.0 * _count / _nTrials.Value);
                progress = $"{percent,3}%| {_count}/{_nTrials.Value}";
            }
            else
            {
                progress = $"{elapsedSeconds:0.00}s/{_timeout!.Value.ToString("0.00", CultureInfo.InvariantCulture)}s";
            }

            Console.Error.Write($"\r{_postfix} [{progress}]");
        }
    }

    /// <summary>Ends the current console progress line when any progress was displayed.</summary>
    internal void Close()
    {
        if (_enabled && _count > 0)
            Console.Error.WriteLine();
    }
}
