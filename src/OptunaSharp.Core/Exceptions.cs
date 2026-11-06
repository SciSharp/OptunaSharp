using System;

namespace OptunaSharp;

// Port of optuna/exceptions.py. Python exceptions that map onto existing NumSharp/BCL types
// (ValueError, TypeError, KeyError, RuntimeError, NotImplementedError) are NOT redefined here —
// OptunaSharp throws NumSharp.ValueError / NumSharp.TypeError / NumSharp.KeyError /
// NumSharp.RuntimeError so that typed errors stay consistent across the SciSharp STACK.

/// <summary>Base class for Optuna specific errors.</summary>
public class OptunaError : Exception
{
    /// <summary>Creates a OptunaError carrying the supplied failure details, if any.</summary>
    public OptunaError() { }

    /// <summary>Creates a OptunaError carrying the supplied failure details, if any.</summary>
    /// <param name="message">Error message explaining the failure.</param>
    public OptunaError(string message) : base(message) { }

    /// <summary>Creates a OptunaError carrying the supplied failure details, if any.</summary>
    /// <param name="message">Error message explaining the failure.</param>
    /// <param name="innerException">Original failure retained as the cause of this exception.</param>
    public OptunaError(string message, Exception innerException) : base(message, innerException) { }
}

/// <summary>
///     Exception for pruned trials.
///
///     This error tells a trainer that the current <see cref="Trial" /> was pruned. It is
///     supposed to be raised after <see cref="Trial.ShouldPrune" /> as shown in the following
///     example.
/// </summary>
/// <example>
///     <code>
///     for (int step = 0; step &lt; nTrainIter; step++)
///     {
///         // ... partial fit ...
///         trial.Report(intermediateValue, step);
///         if (trial.ShouldPrune())
///             throw new TrialPruned();
///     }
///     </code>
/// </example>
public class TrialPruned : OptunaError
{
    /// <summary>
    ///     Creates a TrialPruned with an empty <see cref="Exception.Message" />, as Python's
    ///     <c>str(TrialPruned())</c> is <c>''</c>.
    /// </summary>
    /// <remarks>
    ///     The message is not cosmetic: <c>Optimize</c> logs every pruned trial as
    ///     <c>"Trial {number} pruned. {message}"</c>, the port of Optuna's <c>f"... pruned. {str(func_err)}"</c>.
    ///     Left to the BCL default, the bare <c>throw new TrialPruned();</c> every prune loop ends with would log
    ///     <c>Exception of type 'OptunaSharp.TrialPruned' was thrown.</c> on each pruned trial where Optuna logs
    ///     nothing after the period.
    /// </remarks>
    public TrialPruned() : base(string.Empty) { }

    /// <summary>Creates a TrialPruned carrying the supplied failure details, if any.</summary>
    /// <param name="message">Error message explaining the failure.</param>
    public TrialPruned(string message) : base(message) { }
}

/// <summary>
///     Exception for CLI.
///
///     CLI raises this exception when it receives invalid configuration.
/// </summary>
public class CLIUsageError : OptunaError
{
    /// <summary>Creates a CLIUsageError carrying the supplied failure details, if any.</summary>
    public CLIUsageError() { }

    /// <summary>Creates a CLIUsageError carrying the supplied failure details, if any.</summary>
    /// <param name="message">Error message explaining the failure.</param>
    public CLIUsageError(string message) : base(message) { }
}

/// <summary>
///     Exception for storage operation.
///
///     This error is raised when an operation failed in backend DB of storage.
/// </summary>
public class StorageInternalError : OptunaError
{
    /// <summary>Creates a StorageInternalError carrying the supplied failure details, if any.</summary>
    public StorageInternalError() { }

    /// <summary>Creates a StorageInternalError carrying the supplied failure details, if any.</summary>
    /// <param name="message">Error message explaining the failure.</param>
    public StorageInternalError(string message) : base(message) { }
}

/// <summary>
///     Exception for a duplicated study name.
///
///     This error is raised when a specified study name already exists in the storage.
/// </summary>
public class DuplicatedStudyError : OptunaError
{
    /// <summary>Creates a DuplicatedStudyError carrying the supplied failure details, if any.</summary>
    public DuplicatedStudyError() { }

    /// <summary>Creates a DuplicatedStudyError carrying the supplied failure details, if any.</summary>
    /// <param name="message">Error message explaining the failure.</param>
    public DuplicatedStudyError(string message) : base(message) { }
}

/// <summary>
///     Exception for updating a finished trial.
///
///     This error is raised when attempting to update a finished trial.
///     (Python: inherits both <c>OptunaError</c> and <c>RuntimeError</c>; C# keeps the
///     <see cref="OptunaError" /> base.)
/// </summary>
public class UpdateFinishedTrialError : OptunaError
{
    /// <summary>Creates a UpdateFinishedTrialError carrying the supplied failure details, if any.</summary>
    public UpdateFinishedTrialError() { }

    /// <summary>Creates a UpdateFinishedTrialError carrying the supplied failure details, if any.</summary>
    /// <param name="message">Error message explaining the failure.</param>
    public UpdateFinishedTrialError(string message) : base(message) { }
}
