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
`[3.0.5,4.0.0)`. The repository's package configuration documents Python 3.7–3.13 for that 3.0.5
floor and a direct `pythonnet` 3.1.0 reference for a Python 3.14 host. This range does not install
CPython or any Python library. The individual bridge packages have their own Python requirements.
[Package definition](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Interop.pythonnet/OptunaSharp.Interop.pythonnet.csproj).

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

`Run(Action)` and `Run<T>(Func<T>)` initialize the runtime if necessary and acquire the GIL for the
body. `Initialize()` is also public, idempotent and synchronized. `IsInitialized` reports whether
Python.NET's engine is running.

## GIL and object ownership

Keep every operation on a `PyObject`, including its disposal, inside `PythonRuntime.Run` or another
correct GIL scope. Convert results to managed values before returning. Python.NET requires the GIL
for calls into its Python API. [Python.NET embedding documentation](https://pythonnet.github.io/pythonnet/dotnet.html).

`PythonRuntime.Import(name)` caches a module for the process. Its return value is borrowed: do not
dispose it. Fresh `GetAttr`, `Invoke` and similar results are owned handles; use `using` inside the
GIL scope. `Builtins` follows the same cached-module rule. `Import` itself expects the caller to hold
the GIL.

The runtime has no public shutdown method. It calls `BeginAllowThreads` after initialization and keeps
the engine and cached imports for the process lifetime. Do not finalize or replace the interpreter
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

See the [runtime implementation](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Interop.pythonnet/PythonRuntime.cs)
and [conversion implementation](https://github.com/SciSharp/OptunaSharp/blob/master/src/OptunaSharp.Interop.pythonnet/PyConvert.cs).
