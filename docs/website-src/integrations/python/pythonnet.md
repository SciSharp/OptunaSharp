# Python.NET runtime

`OptunaSharp.Interop.pythonnet` supplies the embedded CPython runtime used by OptunaHub, BoTorch,
pycma and pymoo. Its namespace is `OptunaSharp.Interop.PythonNet`. The runtime handles interpreter
discovery, initialization, GIL acquisition and cached imports; each integration handles its own
library calls.

## Install and select an interpreter

```shell
dotnet add package OptunaSharp.Interop.pythonnet
```

The project targets .NET 8 and .NET 10 and references NuGet `pythonnet` with the range
`[3.2.1, 4.0.0)`, which supports Python 3.11–3.15 (and the free-threaded 3.14 build). Python 3.7–3.10
hosts are not supported: pythonnet 3.1.0 removed 3.7–3.9 and 3.2.0 raised its minimum to 3.11. This
range does not install CPython or any Python library. The individual bridge packages have their own
Python requirements.
[Package definition](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Interop.pythonnet/OptunaSharp.Interop.pythonnet.csproj).

Install dependencies into the interpreter that will host the process. For example, for OptunaHub:

```shell
python -m pip install optuna optunahub
```

Before the first Python call, choose one discovery path:

| Configuration | Runtime behavior |
|---|---|
| `PYTHON` names a Python executable | Probe that executable first for its shared library and site-packages |
| No `PYTHONNET_PYDLL` | Try `PYTHON`, then `python3`, then `python`; add the discovered site-packages to embedded `sys.path` |
| `PYTHONNET_PYDLL` names the shared library | Honor it directly; skip executable probing and automatic site-packages injection |
| Host already initialized `PythonEngine` | Reuse the existing interpreter as configured by the host |

For a virtual environment on Windows, an executable selection can be set in PowerShell:

```powershell
$env:PYTHON = 'C:\path\to\venv\Scripts\python.exe'
dotnet run
```

Alternatively, `PYTHONNET_PYDLL` must point to the actual shared library, such as `python312.dll`
or `libpython3.12.so`, rather than to `python.exe`. When pinning the library directly, the host is
responsible for making the desired environment's packages importable. Inspect `sys.path` when an
import fails even though `pip install` succeeded.

## Verify the embedded environment

This example imports only Python's standard library and returns ordinary .NET strings:

```csharp
using System;
using OptunaSharp.Interop.PythonNet;

var details = PythonRuntime.Run(() =>
{
    var sys = PythonRuntime.Import("sys");
    using var version = sys.GetAttr("version");
    using var path = sys.GetAttr("path");
    return (Version: version.ToString(), Path: path.ToString());
});

Console.WriteLine(details.Version);
Console.WriteLine(details.Path);
```

`Run(Action)` and `Run<T>(Func<T>)` initialize the runtime if necessary and run the body under the GIL
on the runtime's Python thread; the caller waits for it. `Initialize()` is also public, idempotent and
synchronized. `IsInitialized` reports whether Python.NET's engine is running.

## The Python thread

Every `Run` body, and so every call of the BoTorch, pycma, pymoo and OptunaHub bridges, runs on one
long-lived background thread named `OptunaSharp Python`. That thread keeps one CPython thread state for
the life of the process and has a 16 MB stack.

On any other .NET thread, Python.NET's `Py.GIL()` creates a CPython thread state when the outermost scope
opens and frees it when the scope ends. Extensions built with pybind11, such as scipy and torch, keep using
the thread state of the thread on which they first loaded. Measured on 2026-11-07:

- `PymooSampler` driven off the main thread crashed the process with an access violation in scipy's
  `cdist` in 7 of 8 runs;
- torch, imported on one .NET thread and used from another, hung.

On the Python thread nothing is freed, so neither can happen, and samplers work at any `nJobs`. The
16 MB stack is the size CPython gives its own threads on macOS, where numba's compiler needs it.

- A body that calls `Run` again runs inline.
- Code that already holds the GIL inside its own `Py.GIL()` may call `Run`: it releases the GIL while it
  waits.
- The caller's culture and `AsyncLocal` values flow into the body, and the body's exception reaches the
  caller with its stack trace.
- A body must not wait for another thread that may itself be waiting for the Python thread. For example,
  do not call `Trial` or `Study` methods inside `Run` while other threads optimize the same study with a
  Python sampler. For Python code of your own inside an objective, take the GIL with `Py.GIL()`.
- Calls run one at a time, as Python does under the GIL. Python code from concurrent studies that releases
  the GIL (torch kernels, large NumPy operations) therefore no longer overlaps. Each call costs about
  1.5 µs for the hand-off.

## GIL and object ownership

Keep every operation on a `PyObject`, including its disposal, inside `PythonRuntime.Run` or another
correct GIL scope. Convert results to managed values before returning. Python.NET requires the GIL
for calls into its Python API. [Python.NET embedding documentation](https://pythonnet.github.io/pythonnet/dotnet.html).

`PythonRuntime.Import(name)` caches a module for the process. Its return value is borrowed: do not
dispose it. Fresh `GetAttr`, `Invoke` and similar results are owned handles; use `using` inside the
GIL scope. `Builtins` follows the same cached-module rule. `Import` itself expects the caller to hold
the GIL.

The runtime has no public shutdown method. It starts the engine on the Python thread, calls
`BeginAllowThreads` after initialization, and keeps the engine and cached imports for the process lifetime.
With an engine the host started itself, the Python thread takes a thread state of its own and keeps it. Do not finalize or replace the interpreter
between studies; the bridge is designed around one engine for the process. This runtime dependency also
means the Python bridges are outside the pure-managed/AOT deployment path.

The sampler packages hide their transient Python handles. OptunaHub's `OptunaHubModule`,
`OptunaHubSampler`, `OptunaHubPruner` and `OptunaHubProblem` instead own persistent handles and implement
`IDisposable`; keep them alive until optimization completes and dispose them afterward.

## Conversion and errors

`PyConvert` converts Optuna scalar parameter values and float, integer and categorical distributions.
Returned scalar values become `null`, `bool`, `long`, `double` or `string`, including supported NumPy
scalar spellings. It is not a general serializer for arbitrary Python object graphs. Its methods also
require the GIL; newly returned Python objects belong to the caller.

Initialization and generic import failures are wrapped in `PythonInteropException`. OptunaHub loading
adds `OptunaHubInteropException` with package-specific context. Backend calls can also propagate Python
or OptunaSharp exceptions. Diagnose a missing module using the embedded interpreter's import path,
then install the missing dependency into that environment; changing an unrelated system Python will
not fix the running process.

See the [runtime implementation](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Interop.pythonnet/PythonRuntime.cs)
and [conversion implementation](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Interop.pythonnet/PyConvert.cs).
