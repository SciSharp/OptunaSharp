---
name: pythonic
description: Write or use OptunaSharp's "pythonic" pythonnet interop layer — the `Python.*` namespace pattern that makes C# driving embedded CPython read like Python, plus the GIL/PyObject-lifetime discipline and NumSharp's ZERO-COPY NDArray⇄numpy⇄torch crossing (so you never hand-roll element loops). Use when building or editing a pythonnet bridge (BoTorch, OptunaHub, …), adding a `Python.*` facade member or a PyObject tensor-extension, crossing data between C# and numpy/torch, deciding typed-facade vs `evalStmts` snippet, or debugging a GIL / leak / lifetime / "numpy scalar isn't a Python int" bug in the interop.
keywords: pythonic layer, Python.* facade, pythonnet bridge, make C# read like Python, zero-copy tensor crossing, NDArray to torch, ToTorch, ToTorchNDArray, GIL discipline, PyObject lifetime, evalStmts snippet runner, PythonRuntime.Run
keywords-sparse: pythonic interop, Python.* namespace, write a pythonnet bridge, zero-copy numpy/torch crossing, GIL and PyObject lifetime, typed facade vs evalStmts, tensor extension member
---

# pythonic — the `Python.*` interop layer (read like Python, cross data zero-copy)

How OptunaSharp's pythonnet bridges (`OptunaSharp.Interop.BoTorch`, `.OptunaHub`, …) drive **embedded
CPython** so the C# reads like the Python it ports — and how to move arrays/tensors across the boundary
**without copying, element by element, ever**. Built on the generic backend
`OptunaSharp.Interop.pythonnet` (`PythonRuntime`) and NumSharp's `NumSharp.Interop.pythonnet` crossing.

**Reference implementations to copy:**
- `refs/NumSharp/src/NumSharp.Interop.pythonnet/Pythonic.cs` — the **canonical typed facade** (`Python.np`,
  `ctypes`, `weakref`, `builtins`): one C# member per Python call, **zero Python source strings**.
- `refs/NumSharp/test/NumSharp.Tests.Interop/Pythonic.cs` — the **hybrid**: facade rows PLUS the
  `eval`/`with`/`evalStmts` snippet-runner escape hatch.
- `src/OptunaSharp.Interop.BoTorch/Pythonic.cs` — our `botorch` snippet-runner + `manual_seed` scope +
  the `TorchTensorExtensions` (`t.dim()`/`t.size(-1)`/`t.squeeze(0)`).
- `src/OptunaSharp.Interop.pythonnet/PythonRuntime.cs` — the generic, library-agnostic backend.

---

## 1 — The two flavors (pick the right one)

A pythonic layer lives in **`namespace Python`** (beside `Python.Runtime`) so `using Python;` lets call
sites read `torch.tensor(...)` / `botorch.evalStmts(...)`. It comes in two shapes; use both, for
different jobs:

**(A) Typed facade — one C# member per Python call, NO source strings.** Dispatches straight to a cached
module/callable. The C# *reads* like Python but *is* C#. This is `Python.np` in NumSharp.

```csharp
internal static class torch                         // namespace Python
{
    private static PyObject Module => PythonRuntime.Import("torch");   // cached, process-scoped
    internal static PyObject float64 => Module.GetAttr("float64");     // caller disposes
    internal static PyObject tensor(PyObject data)
    {
        using PyObject fn = Module.GetAttr("tensor");
        using PyObject dtype = float64;
        using var kwargs = new PyDict(); kwargs["dtype"] = dtype;
        return fn.Invoke(new[] { data }, kwargs);                      // torch.tensor(data, dtype=float64)
    }
}
```

Plus **extension members on `PyObject`** for the instance-side, spelled as Python (`mv.tobytes("C")`,
`arr.setflags(write:false)`, `t.size(-1)`):

```csharp
internal static class TorchTensorExtensions           // namespace Python, brought in by `using Python;`
{
    internal static long size(this PyObject t, long dim)
    { using PyObject d = new PyInt(dim); using PyObject s = t.InvokeMethod("size", d); return s.As<long>(); }
    internal static PyObject squeeze(this PyObject t, long dim)
    { using PyObject d = new PyInt(dim); return t.InvokeMethod("squeeze", d); }   // caller disposes
}
```

**(B) Snippet runner — execute multi-statement Python SOURCE.** For a body that is a little *program*
(control flow, kwarg-heavy constructor calls) and must **diff 1:1 against the reference Python**, don't
translate it to `GetAttr`/`Invoke` chains — run it as text. This is NumSharp's test `Python.np`
`eval`/`with`/`evalStmts`, and our `botorch`:

```csharp
internal static PyObject evalStmts(string setup, string resultVar, params (string name, PyObject val)[] vars)
{
    using PyModule scope = Py.CreateScope();
    scope.Exec(Imports);                              // `import torch`, `from botorch… import …`
    foreach ((string name, PyObject val) in vars) scope.Set(name, val);
    scope.Exec(setup);                                // the 7-line GP-fit + acqf + optimize body
    return scope.Eval(resultVar);
}
```
The body lives as a `const string` that reads like `botorch.py`:
```csharp
private const string LogEiBody =
    "train_x = normalize(train_x, bounds=bounds)\n" +
    "model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.size(-1)))\n" +
    "fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))\n" + …;
```

**The decision, in one line:** a single clean call → a **facade row** (A); a multi-statement program you
want to diff against the Python source → the **snippet runner** (B). Don't push single calls into strings,
and don't unroll multi-statement Python into `PyObject` chains. NumSharp keeps *both* on purpose.

---

## 2 — GIL discipline (non-negotiable)

Everything that touches a `PyObject` — including its `Dispose` — runs **under the GIL**. Go through
`PythonRuntime.Run`, which acquires it (and starts the engine if needed):

```csharp
double best = PythonRuntime.Run(() =>                 // holds Py.GIL() for the whole body
{
    using PyObject t = BoTorchInterop.ToTensor(matrix);
    return t.size(0);
});
```
- Compute the **C# side first (no GIL)**, then open the GIL only for the Python part + the comparison.
- `Py.GIL()` is **re-entrant** — NumSharp's crossing re-acquires it internally, so calling `ToTorch()` /
  `ToTorchNDArray()` *inside* a `Run` body is fine.
- Python→.NET callbacks do **not** inherit the GIL (pythonnet releases it around the managed body). If a
  facade member can run from such a callback, keep GIL management on.

---

## 3 — Lifetime & ownership (where the leaks and use-after-frees hide)

- **Cached modules are process-scoped — never dispose them.** `PythonRuntime.Import("torch")` returns a
  session-owned handle; a `using` on it decrefs the cache. Long-lived singletons (a dtype object) may be
  cached the same way; on a race, **loser-dispose** (`ConcurrentDictionary.TryAdd`, dispose the loser).
- **A returned `PyObject` is owned by the caller — dispose it** (`using`). Bound operands passed into a
  scope stay owned by the caller (the throwaway scope only increfs).
- **Fluent chains LEAK — use `using` locals.** `t.detach().cpu().tolist()` drops the `detach()` and
  `cpu()` intermediates on the floor (the interop leak gate goes red). Spell each step:
  ```csharp
  using PyObject detached = t.detach();
  using PyObject cpu = detached.cpu();
  using PyObject list = cpu.tolist();
  ```
  (This is why the facade members return owned `PyObject`s and the docs say "dispose it".)
- **A context manager becomes a disposable scope.** `with manual_seed(seed):` →
  `using (botorch.manual_seed(seed)) …` (enter on ctor, `__exit__` on `Dispose`, both under the GIL).

---

## 4 — Data crossing: NEVER hand-roll element loops — use NumSharp's interop

Add the package (version single-sourced with Core's NumSharp):
```xml
<PackageReference Include="NumSharp.Interop.pythonnet" Version="$(NumSharpVersion)" />
```
It **requires an already-running engine** (`EnsureEngine` throws otherwise) and wires onto the SAME
embedded engine `PythonRuntime` starts — it does NOT start its own, and its GIL is re-entrant. Use the
explicit extension methods; you do **not** need `RegisterCodec()` (avoid the global side effect).

**The four verbs** (`NumSharp.Interop.PythonNet.NDArrayPythonInterop`):
| Verb | Direction | Copy? |
|---|---|---|
| `nd.ToNumpy()` | NDArray → numpy | **zero-copy view** (rooted by a Python `weakref.finalize`) |
| `nd.ToNumpyCopy()` | NDArray → numpy | independent copy |
| `pyobj.ToNDArray()` | numpy/adapter → NDArray | copy into a fresh C-contiguous NDArray |
| `pyobj.ToNDArrayView()` | numpy → NDArray | zero-copy view (leases the buffer) |

**torch** (`TorchInterop`): `nd.ToTorch(copy:false)` = `torch.from_numpy(nd.ToNumpy())` (zero-copy view;
NumSharp roots its buffer for the tensor's life); `pyTensor.ToTorchNDArray(force:true)` = the documented
`detach().cpu().resolve_conj().resolve_neg().numpy()` → `ToNDArray` (one bulk copy, detached);
`pyTensor.AsTorchNDArray()` = zero-copy view of a CPU tensor.

**Copyless from a MANAGED array — `copy: false` PINS it.** `np.array(data, copy: false)` does
`GCHandle.Alloc(data, GCHandleType.Pinned)` and views the pinned address (see
`UnmanagedMemoryBlock.FromArray`), so combined with `ToTorch` the whole chain is a view of one buffer:

```csharp
// managed double[,]  →(pin)→ NDArray view →(np.frombuffer)→ numpy view →(from_numpy)→ torch tensor
internal static PyObject ToTensor(double[,] data) => np.array(data, copy: false).ToTorch();
// torch result → NDArray (one bulk copy, safe to dispose the tensor after)
internal static NDArray  ToNDArray(PyObject tensor) => tensor.ToTorchNDArray(force: true);
```
NumSharp's `ExportKeeper` + `weakref.finalize` hold the pin until the last Python view dies, then
`UnmanagedMemoryBlock`'s `Disposer` calls `GCHandle.Free()`. So the managed array must be **short-lived
and not mutated** by the Python side (pin lifetime = tensor lifetime, released on Python GC) — true for
per-call training matrices. The old PyList-append / `tolist()`-loop marshaling is the anti-pattern this
replaces.

---

## 5 — The generic backend vs library specifics

`OptunaSharp.Interop.pythonnet/PythonRuntime.cs` is **library-agnostic**: `Initialize()` (idempotent,
process-scoped; honors `PYTHONNET_PYDLL`, else probes `python`/`python3` for libpython + injects
site-packages so `pip install`ed packages import), `Run(Action)`/`Run<T>(Func<T>)` (GIL), `Import(name)`
(cached), `Builtins`. It does **no** startup imports. Library-specific accessors (versions, `EnsureQuiet`,
module handles) live in their own runtime (e.g. `OptunaHubRuntime`). Exception base:
`PythonInteropException` (e.g. `OptunaHubInteropException : PythonInteropException`).

Discovery pin for this machine: `PYTHONNET_PYDLL=C:/Users/ELI/.claude/python/python312.dll`.

---

## 6 — Gotchas (each cost real time)

- **`using static` does NOT import extension methods.** To read `t.size(-1)` / `torch.tensor(...)` you need
  the namespace form `using Python;`, not `using static Python.torch;` (which would also drop the
  `torch.`/`botorch.` prefix that makes it read like the API).
- **A numpy scalar is not a Python `int`.** pythonnet's primitive conversions never consult codecs, so
  `PyLong("arr[i]")` on an int64 array throws — use `int(arr[i])` / `.item()`. numpy floats ARE Python
  floats; a 0-d ndarray is neither (`float(z0)`).
- **Don't dispose cached module handles** (§3). And don't dispose `PyObject.None` casually — mirror the
  reference (NumSharp `using`s it, but it's a shared singleton; follow the existing code).
- **`GetEnumerator()` must not return `this`** for an object that owns unmanaged/GIL state — `foreach`
  disposes the enumerator (closes the iterator). Hand out a thin wrapper (see NumSharp's `nditer` notes).
- **Never build arrays/tensors element by element.** No PyList-append loops in, no `tolist()` loops out —
  §4's crossing does it as a view or one bulk copy.
- **Keep the reference Python close.** Snippet bodies (flavor B) must diff 1:1 against the source
  (`botorch.py`, the OptunaHub module) — port them near-verbatim, `import`s in the scope, don't
  paraphrase.

---

## Definition of done

The bridge reads like the Python it ports: single calls are **facade rows** or **PyObject extensions**
(`namespace Python`, `using Python;`), multi-statement bodies are **`evalStmts` snippets** that diff 1:1
against the source; every `PyObject` touch is under the GIL via `PythonRuntime.Run`; every returned
`PyObject` is `using`-disposed and no fluent chain leaks an intermediate; and **all** array/tensor data
crosses through NumSharp's zero-copy `NumSharp.Interop.pythonnet` verbs (`np.array(x, copy:false).ToTorch()`
in, `ToTorchNDArray` out) — never a hand-rolled element loop. Validate with the live interop gate
(byte-exact vs the real library) on both TFMs; a leak gate that returns to baseline proves the lifetimes.
