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
