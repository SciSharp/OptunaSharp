using System;
using Python.Runtime;
using OptunaSharp.Interop.PythonNet;

namespace Python;

/// <summary>
///     <c>import botorch</c>, the OptunaSharp flavour. The BoTorch bridge's Python layer has two halves,
///     both living in the <c>Python</c> namespace (beside <c>Python.Runtime</c>) so the consuming C# reads
///     for what it is — <c>botorch.evalStmts(...)</c> and <c>torch.tensor(...)</c> run real BoTorch/torch.
///     Bring them into scope with <c>using Python;</c>.
///     <list type="bullet">
///       <item>
///         <b>This class, <c>botorch</c> — the Python-SNIPPET runner.</b> <see cref="with" /> and
///         <see cref="evalStmts" /> execute the multi-statement <c>candidates_func</c> bodies as real
///         Python <em>source</em> (the bodies live as string constants in
///         <see cref="OptunaSharp.Interop.BoTorch.BoTorchCandidates" /> and must diff 1:1 against
///         <c>botorch.py</c>), plus the module-level calls the bridge makes on botorch itself
///         (<see cref="check" />, <see cref="manual_seed" />). This is deliberately NOT a typed facade —
///         it is the escape hatch NumSharp's test-side <c>Python.np</c> also keeps (its
///         <c>eval</c>/<c>with</c>/<c>evalStmts</c>) for the code Python expresses far more clearly than a
///         chain of <see cref="PyObject" /> calls.
///       </item>
///       <item>
///         <b><see cref="torch" /> + the tensor extension members — the TYPED facade.</b> One C# member
///         per torch call, dispatching straight to the cached module (no Python source), modelled on
///         NumSharp's src <c>Python.np</c>. That is where the tensor marshaling reads like Python —
///         <c>torch.tensor(list)</c>, <c>t.detach().cpu()</c>, <c>t.size(-1)</c>.
///       </item>
///     </list>
///     <para>Every member calls the C-API and MUST run under the GIL (from a
///     <see cref="PythonRuntime.Run(System.Action)" /> body). A returned <see cref="PyObject" /> outlives
///     the throwaway scope and is owned by the caller (dispose it); bound operands stay owned by the
///     caller (the throwaway scope only increfs them).</para>
/// </summary>
internal static class botorch
{
    /// <summary>
    ///     The names each snippet resolves against — <c>torch</c> plus the BoTorch/GPyTorch symbols the
    ///     ported <c>candidates_func</c>s use. Imports are cached in <c>sys.modules</c>, so re-running
    ///     this per call is a dictionary lookup, not a real import.
    /// </summary>
    private const string Imports =
        "import torch\n" +
        "from botorch.utils.transforms import normalize, unnormalize\n" +
        "from botorch.models import SingleTaskGP\n" +
        "from botorch.models.transforms.outcome import Standardize\n" +
        "from gpytorch.mlls import ExactMarginalLogLikelihood\n" +
        "from botorch.fit import fit_gpytorch_mll\n" +
        "from botorch.acquisition.analytic import LogExpectedImprovement\n" +
        "from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement\n" +
        "from botorch.acquisition.logei import qLogExpectedImprovement\n" +
        "from botorch.sampling import SobolQMCNormalSampler\n" +
        "from botorch.optim import optimize_acqf\n";

    /// <summary>Ensure torch + botorch import (botorch.py's <c>_imports.check()</c>). Under the GIL.</summary>
    internal static void check()
    {
        _ = PythonRuntime.Import("torch");
        _ = PythonRuntime.Import("botorch");
    }

    /// <summary>
    ///     Evaluate a single botorch expression with the given operands bound — the workhorse for the
    ///     one-liners. Under the GIL.
    /// </summary>
    internal static PyObject with(string expr, params (string name, PyObject val)[] vars)
    {
        using PyModule scope = Py.CreateScope();
        scope.Exec(Imports);
        foreach ((string name, PyObject val) in vars)
            scope.Set(name, val);
        return scope.Eval(expr);
    }

    /// <summary>
    ///     Run the setup <b>statements</b> with the operands bound, then return
    ///     <paramref name="resultVar" /> — the multi-step form a <c>candidates_func</c> needs (the GP is
    ///     fit in place, the acqf built, then optimized). Under the GIL.
    /// </summary>
    internal static PyObject evalStmts(string setup, string resultVar, params (string name, PyObject val)[] vars)
    {
        using PyModule scope = Py.CreateScope();
        scope.Exec(Imports);
        foreach ((string name, PyObject val) in vars)
            scope.Set(name, val);
        scope.Exec(setup);
        return scope.Eval(resultVar);
    }

    /// <summary>
    ///     <c>with botorch.utils.sampling.manual_seed(seed): …</c> as a disposable scope — enters the
    ///     context manager now and exits it on <see cref="IDisposable.Dispose" />; a <c>null</c> seed is a
    ///     no-op scope (matching Optuna). Wrap the acquisition optimization in
    ///     <c>using (botorch.manual_seed(seed)) …</c> to make it reproducible. Under the GIL (create AND
    ///     dispose).
    /// </summary>
    internal static ManualSeedScope manual_seed(int? seed) => new(seed);

    /// <summary>The live <c>manual_seed</c> context manager — entered on construction, exited on dispose.</summary>
    internal sealed class ManualSeedScope : IDisposable
    {
        private readonly PyObject _cm;

        internal ManualSeedScope(int? seed)
        {
            PyObject sampling = PythonRuntime.Import("botorch.utils.sampling");
            using PyObject fn = sampling.GetAttr("manual_seed");
            using PyObject seedArg = seed is { } s ? s.ToPython() : PyObject.None;
            _cm = fn.Invoke(seedArg);
            using PyObject enter = _cm.GetAttr("__enter__");
            enter.Invoke().Dispose();
        }

        public void Dispose()
        {
            using (_cm)
            {
                using PyObject exit = _cm.GetAttr("__exit__");
                exit.Invoke(PyObject.None, PyObject.None, PyObject.None).Dispose();
            }
        }
    }
}

/// <summary>
///     <c>import torch</c>, the OptunaSharp flavour — the torch calls the BoTorch bridge marshals through,
///     spelled as they read in Python. Modelled on NumSharp's src <c>Python.np</c>: one typed C# member
///     per torch call, dispatching to the cached module via <see cref="PythonRuntime.Import" /> — no
///     Python source strings. Under the GIL.
/// </summary>
internal static class torch
{
    /// <summary>The cached <c>torch</c> module (process-scoped; do NOT dispose it). Under the GIL.</summary>
    private static PyObject Module => PythonRuntime.Import("torch");

    /// <summary><c>torch.float64</c> — a fresh handle to the dtype singleton; the caller disposes it. Under the GIL.</summary>
    internal static PyObject float64 => Module.GetAttr("float64");

    /// <summary>
    ///     <c>torch.tensor(data, dtype=torch.float64)</c> — the bridge marshals every tensor as float64
    ///     (BoTorch's <c>SingleTaskGP</c> fits in float64; a float32 tensor would silently break parity).
    ///     The returned tensor is owned by the caller (dispose it). Under the GIL.
    /// </summary>
    internal static PyObject tensor(PyObject data)
    {
        using PyObject fn = Module.GetAttr("tensor");
        using PyObject dtype = float64;
        using var kwargs = new PyDict();
        kwargs["dtype"] = dtype;
        return fn.Invoke(new[] { data }, kwargs);
    }
}

/// <summary>
///     Python-shaped instance members on a torch tensor <see cref="PyObject" /> — the attribute reads and
///     method calls the bridge makes, spelled as they are in Python (<c>t.detach()</c>, <c>t.cpu()</c>,
///     <c>t.tolist()</c>, <c>t.dim()</c>, <c>t.size(-1)</c>, <c>t.squeeze(0)</c>). Brought into scope by
///     <c>using Python;</c>. Members returning a <see cref="PyObject" /> hand ownership to the caller
///     (dispose it); the rest return CLR values. Under the GIL.
/// </summary>
internal static class TorchTensorExtensions
{
    /// <summary><c>tensor.detach()</c> — a fresh detached view. Dispose it.</summary>
    internal static PyObject detach(this PyObject t) => t.InvokeMethod("detach");

    /// <summary><c>tensor.cpu()</c> — the tensor on the CPU device. Dispose it.</summary>
    internal static PyObject cpu(this PyObject t) => t.InvokeMethod("cpu");

    /// <summary><c>tensor.tolist()</c> — a (nested) Python list of the values. Dispose it.</summary>
    internal static PyObject tolist(this PyObject t) => t.InvokeMethod("tolist");

    /// <summary><c>tensor.dim()</c> — the number of dimensions.</summary>
    internal static long dim(this PyObject t)
    {
        using PyObject d = t.InvokeMethod("dim");
        return d.As<long>();
    }

    /// <summary><c>tensor.size(dim)</c> — the length of one dimension (negative indexes from the end).</summary>
    internal static long size(this PyObject t, long dim)
    {
        using PyObject d = new PyInt(dim);
        using PyObject s = t.InvokeMethod("size", d);
        return s.As<long>();
    }

    /// <summary><c>tensor.size()</c> — the whole <c>torch.Size</c> shape (e.g. for a diagnostic message). Dispose it.</summary>
    internal static PyObject size(this PyObject t) => t.InvokeMethod("size");

    /// <summary><c>tensor.squeeze(dim)</c> — drop a size-1 dimension. Dispose it.</summary>
    internal static PyObject squeeze(this PyObject t, long dim)
    {
        using PyObject d = new PyInt(dim);
        return t.InvokeMethod("squeeze", d);
    }
}
