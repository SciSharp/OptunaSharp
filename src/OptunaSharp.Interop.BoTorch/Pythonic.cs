using System;
using Python.Runtime;
using OptunaSharp.Interop.PythonNet;

namespace Python;

/// <summary>
///     <c>import botorch</c>, the OptunaSharp flavour — the BoTorch bridge's Python layer, living in the
///     <c>Python</c> namespace (beside <c>Python.Runtime</c>) so the consuming C# reads for what it is
///     — <c>botorch.evalStmts(...)</c> runs real BoTorch. Bring it into scope with <c>using Python;</c>.
///     <para>
///     This is the Python-SNIPPET runner: <see cref="with" /> and <see cref="evalStmts" /> execute the
///     multi-statement <c>candidates_func</c> bodies as real Python <em>source</em> (the bodies live as
///     string constants in <see cref="OptunaSharp.Interop.BoTorch.BoTorchCandidates" /> and must diff
///     1:1 against <c>botorch.py</c>), plus the module-level calls the bridge makes on botorch itself
///     (<see cref="check" />, <see cref="manual_seed" />). It is deliberately NOT a typed torch facade —
///     the tensor MARSHALING is done zero-copy through NumSharp's interop
///     (<c>NumSharp.Interop.pythonnet</c>: <c>NDArray.ToTorch</c> / <c>PyObject.ToTorchNDArray</c>, see
///     <see cref="OptunaSharp.Interop.BoTorch.BoTorchInterop" />), and this class keeps only the escape
///     hatch NumSharp's own test-side <c>Python.np</c> also keeps (its <c>eval</c>/<c>with</c>/
///     <c>evalStmts</c>) for code Python expresses far more clearly than a chain of
///     <see cref="PyObject" /> calls. The small torch-tensor introspection the bridge still needs for
///     botorch.py's candidate-shape checks is the extension members in <see cref="TorchTensorExtensions" />.
///     </para>
///     <para>Every member calls the C-API and MUST run under the GIL (from a
///     <see cref="PythonRuntime.Run(System.Action)" /> body). A returned <see cref="PyObject" /> outlives
///     the throwaway scope and is owned by the caller (dispose it); bound operands stay owned by the
///     caller (the throwaway scope only increfs them).</para>
/// </summary>
internal static class botorch
{
    /// <summary>
    ///     The names each snippet resolves against — <c>torch</c> plus the BoTorch/GPyTorch symbols the
    ///     ported <c>candidates_func</c>s use — followed by the botorch.py module-level helpers the bodies
    ///     call (<c>_get_constraint_funcs</c>, <c>_get_sobol_qmc_normal_sampler</c>,
    ///     <c>_validate_botorch_version_for_constrained_opt</c>) and the <c>try_import</c> success flags
    ///     (<c>_logei_ok</c> / <c>_qlogei_ok</c> / <c>_qloghvi_ok</c> / <c>_qhvkg_ok</c>) that mirror
    ///     botorch.py's <c>_imports_*.is_successful()</c>. The version-gated symbols are wrapped in
    ///     <c>try/except ImportError</c> exactly as botorch.py wraps them in <c>with try_import()</c>, so a
    ///     too-old botorch fails inside the ported func (with the verbatim message) rather than in the
    ///     preamble. Imports and defs are cached in <c>sys.modules</c> / re-bound per call (a dictionary
    ///     lookup), so re-running this per candidates_func call is cheap.
    /// </summary>
    private const string Imports =
        "import torch\n" +
        "from packaging import version\n" +
        "import botorch.version\n" +
        "from botorch.utils.transforms import normalize, unnormalize\n" +
        "from botorch.models import SingleTaskGP, ModelListGP\n" +
        "from botorch.models.transforms.outcome import Standardize\n" +
        "from gpytorch.mlls import ExactMarginalLogLikelihood\n" +
        "from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood\n" +
        "from botorch.fit import fit_gpytorch_mll\n" +
        "from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement\n" +
        "from botorch.acquisition.knowledge_gradient import qKnowledgeGradient\n" +
        "from botorch.acquisition.objective import ConstrainedMCObjective, GenericMCObjective\n" +
        "from botorch.acquisition.multi_objective import monte_carlo\n" +
        "from botorch.acquisition.multi_objective import objective as multi_objective_objective\n" +
        "from botorch.acquisition.multi_objective.analytic import ExpectedHypervolumeImprovement\n" +
        "from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective\n" +
        "from botorch.sampling import SobolQMCNormalSampler\n" +
        "from botorch.sampling.list_sampler import ListSampler\n" +
        "from botorch.optim import optimize_acqf\n" +
        "from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning\n" +
        "from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization\n" +
        "from botorch.utils.sampling import sample_simplex\n" +
        // botorch>=0.8.1 — LogExpectedImprovement / LogConstrainedExpectedImprovement (_imports_logei).
        "try:\n" +
        "    from botorch.acquisition.analytic import LogExpectedImprovement, LogConstrainedExpectedImprovement\n" +
        "    _logei_ok = True\n" +
        "except ImportError:\n" +
        "    _logei_ok = False\n" +
        // botorch>=0.9.0 — qLogExpectedImprovement (_imports_qlogei).
        "try:\n" +
        "    from botorch.acquisition.logei import qLogExpectedImprovement\n" +
        "    _qlogei_ok = True\n" +
        "except ImportError:\n" +
        "    _qlogei_ok = False\n" +
        // qLog(Noisy)ExpectedHypervolumeImprovement (_imports_qloghvi).
        "try:\n" +
        "    from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement, qLogNoisyExpectedHypervolumeImprovement\n" +
        "    _qloghvi_ok = True\n" +
        "except ImportError:\n" +
        "    _qloghvi_ok = False\n" +
        // botorch>=0.9.5 — hypervolume_knowledge_gradient (_imports_qhvkg).
        "try:\n" +
        "    from botorch.acquisition.multi_objective import hypervolume_knowledge_gradient\n" +
        "    _qhvkg_ok = True\n" +
        "except ImportError:\n" +
        "    _qhvkg_ok = False\n" +
        // botorch.py module-level helpers (ported verbatim), so the bodies read 1:1.
        "def _get_sobol_qmc_normal_sampler(num_samples):\n" +
        "    return SobolQMCNormalSampler(torch.Size((num_samples,)))\n" +
        "def _get_constraint_funcs(n_constraints):\n" +
        "    def _make_func(i):\n" +
        "        def _constraint(Z):\n" +
        "            return Z[..., -n_constraints + i]\n" +
        "        return _constraint\n" +
        "    return [_make_func(i) for i in range(n_constraints)]\n" +
        "def _validate_botorch_version_for_constrained_opt(func_name):\n" +
        "    if version.parse(botorch.version.version) < version.parse('0.9.0'):\n" +
        "        raise ImportError(\n" +
        "            f'{func_name} requires botorch>=0.9.0 for constrained problems, but got '\n" +
        "            f'botorch={botorch.version.version}.\\n'\n" +
        "            'Please run ``pip install botorch --upgrade``.')\n";

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
///     The small set of torch-tensor introspection members the bridge needs to reproduce botorch.py's
///     candidate-shape validation, spelled as they read in Python (<c>t.dim()</c>, <c>t.size(-1)</c>,
///     <c>t.size()</c>, <c>t.squeeze(0)</c>). Data marshaling does NOT go through here — that is NumSharp's
///     zero-copy crossing (see <see cref="OptunaSharp.Interop.BoTorch.BoTorchInterop" />); these are pure
///     metadata/shape calls. Brought into scope by <c>using Python;</c>. Members returning a
///     <see cref="PyObject" /> hand ownership to the caller (dispose it); the rest return CLR values.
///     Under the GIL.
/// </summary>
internal static class TorchTensorExtensions
{
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
