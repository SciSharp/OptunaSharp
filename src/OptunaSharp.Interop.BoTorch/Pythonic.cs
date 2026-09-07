using Python.Runtime;
using OptunaSharp.Interop.PythonNet;

namespace Python;

/// <summary>
///     <c>import botorch</c>, the OptunaSharp flavour — a Pythonic access layer that lets the BoTorch
///     bridge express its <c>candidates_func</c> bodies as real Python (bind the tensors, run the
///     snippet, take the result) instead of long <see cref="PyObject" /> <c>GetAttr</c>/<c>Invoke</c>
///     chains. It sits in the <c>Python</c> namespace (beside <c>Python.Runtime</c>) so the C# reads for
///     what it is — <c>Python.botorch.evalStmts(...)</c> runs real BoTorch — modelled on NumSharp's
///     <c>Python.np</c> test helper.
///     <para>Every member calls the C-API and MUST run under the GIL (from a
///     <see cref="PythonRuntime.Run(System.Action)" /> body). The bound operands stay owned by the
///     caller (the throwaway scope only increfs them); the returned <see cref="PyObject" /> outlives the
///     scope and is owned by the caller.</para>
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
}
