using System;

namespace OptunaSharp;

// Port of optuna/exceptions.py. Python exceptions that map onto existing NumSharp/BCL types
// (ValueError, TypeError, KeyError, RuntimeError, NotImplementedError) are NOT redefined here —
// OptunaSharp throws NumSharp.ValueError / NumSharp.TypeError / NumSharp.KeyError /
// NumSharp.RuntimeError so that typed errors stay consistent across the SciSharp STACK.

/// <summary>Base class for Optuna specific errors.</summary>
public class OptunaError : Exception
{
    public OptunaError() { }

    public OptunaError(string message) : base(message) { }

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
    public TrialPruned() { }

    public TrialPruned(string message) : base(message) { }
}

/// <summary>
///     Exception for CLI.
///
///     CLI raises this exception when it receives invalid configuration.
/// </summary>
public class CLIUsageError : OptunaError
{
    public CLIUsageError() { }

    public CLIUsageError(string message) : base(message) { }
}

/// <summary>
///     Exception for storage operation.
///
///     This error is raised when an operation failed in backend DB of storage.
/// </summary>
public class StorageInternalError : OptunaError
{
    public StorageInternalError() { }

    public StorageInternalError(string message) : base(message) { }
}

/// <summary>
///     Exception for a duplicated study name.
///
///     This error is raised when a specified study name already exists in the storage.
/// </summary>
public class DuplicatedStudyError : OptunaError
{
    public DuplicatedStudyError() { }

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
    public UpdateFinishedTrialError() { }

    public UpdateFinishedTrialError(string message) : base(message) { }
}
