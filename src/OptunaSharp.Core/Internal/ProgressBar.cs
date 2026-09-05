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

    internal void Update(double elapsedSeconds, Study study)
    {
        if (!_enabled)
            return;

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

    internal void Close()
    {
        if (_enabled && _count > 0)
            Console.Error.WriteLine();
    }
}
