# Logging

OptunaSharp emits progress and diagnostic messages — new best values, independent-sampling warnings,
deprecations — through its own logging module, a port of `optuna.logging` in **`OptunaSharp.Logging`**.
You control how much it says and where the output goes.

## Setting verbosity

```csharp
using OptunaSharp;
using OptunaSharp.Logging;

// Quiet everything below warnings (a common choice for scripted runs).
OptunaLogging.SetVerbosity(LogLevel.Warning);

var study = Optuna.CreateStudy();
study.Optimize(Objective, nTrials: 100);   // no per-trial INFO chatter
```

`LogLevel` mirrors Python's numeric levels: `Debug = 10`, `Info = 20`, `Warning = 30`, `Error = 40`,
`Critical = 50` (with `Warn`/`Fatal` aliases and `NotSet = 0`). Read the current level with
`OptunaLogging.GetVerbosity()`.

## Handlers

By default, records go to a built-in handler that writes to standard error. Turn it off, back on, or
add your own sinks:

```csharp
OptunaLogging.DisableDefaultHandler();          // stop the built-in stderr output
OptunaLogging.AddHandler(new StreamLogHandler(myWriter));   // route records elsewhere
// ...
OptunaLogging.EnableDefaultHandler();
```

Implement `ILogHandler` to bridge into your application's logging framework (Serilog,
`Microsoft.Extensions.Logging`, a file, a test sink). `GetLogger(string name)` returns the `Logger`
used internally, should you need to emit records under the same configuration.

**Threads.** Under `nJobs` several trials log at once. As in Python, where `Handler.handle` holds the handler's
lock around `emit`, OptunaSharp gives each handler one record at a time under a lock of its own, so a handler
needs no lock for what only its `Emit` touches; different handlers still run at the same time. `Emit` runs on the
thread that logged and other threads that log wait for it, so keep it quick, and do not throw from it, call
OptunaSharp from it, or take the Python GIL in it. `StreamLogHandler` writes each record, its exception included,
in one write and flushes it, as Python's `StreamHandler` does, so records from parallel trials never interleave
even over a plain `StreamWriter`. Adding a handler that is already registered does nothing, as in Python.

**Divergence.** Optuna's `enable_propagation` / `disable_propagation` — which toggle whether records
bubble up to Python's *root* logger — have no analog here: OptunaSharp is not built on the Python
`logging` package, so there is no root logger to propagate to. Route records into another framework by
adding your own `ILogHandler` (or removing the default handler) instead.

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `optuna.logging.set_verbosity(optuna.logging.WARNING)` | `OptunaLogging.SetVerbosity(LogLevel.Warning)` |
| `optuna.logging.get_verbosity()` | `OptunaLogging.GetVerbosity()` |
| `optuna.logging.disable_default_handler()` | `OptunaLogging.DisableDefaultHandler()` |
| `optuna.logging.enable_default_handler()` | `OptunaLogging.EnableDefaultHandler()` |

## See also

[Getting Started](getting-started.md) · [Study and Trials](study-and-trials.md) ·
the [API reference](../api/index.md).
