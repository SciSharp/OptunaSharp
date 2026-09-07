using System;
using AwesomeAssertions;
using Python.Runtime;
using OptunaSharp.Interop.BoTorch;
using OptunaSharp.Interop.PythonNet;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     <b>Function-level</b> parity of the BoTorch API the bridge implements — the complement to the
///     end-to-end <see cref="BoTorchSamplerParityTests" />. Each ported <c>candidates_func</c> is called
///     directly over pinned <c>torch.Tensor</c>s and asserted <b>byte-for-byte identical</b> to the real
///     <c>optuna_integration.botorch.*_candidates_func</c> given the <em>same tensors</em> under the
///     <em>same</em> <c>manual_seed</c> — so a wrong kwarg, a wrong <c>num_restarts</c>/<c>raw_samples</c>,
///     or a dropped <c>X_pending</c> in the ported snippet turns the gate red. Also pins the
///     default-selection logic, the tensor marshaling, and the error taxonomy.
///     <para>Reaches the package internals (<c>BoTorchCandidates</c> / <c>BoTorchInterop</c> /
///     <c>NativeCandidatesFunc</c>) via <c>InternalsVisibleTo</c>. Self-skips when botorch /
///     optuna_integration are absent.</para>
/// </summary>
[TestClass]
public class BoTorchApiParityTests : OptunaInteropTestBase
{
    // Pinned inputs (any values work — byte-exactness comes from handing BOTH sides the same tensors).
    private static readonly double[,] TrainX =
        { { 0.1, 0.2 }, { 0.7, 0.3 }, { 0.4, 0.9 }, { 0.55, 0.15 }, { 0.85, 0.6 } };
    private static readonly double[,] TrainObj = { { 1.0 }, { 2.0 }, { 0.5 }, { 1.5 }, { 0.8 } };
    private static readonly double[,] BoundsT = { { 0.0, 0.0 }, { 1.0, 1.0 } };   // (2, d)
    private static readonly double[,] Pending = { { 0.3, 0.6 }, { 0.8, 0.1 } };   // (n_pending, d)

    // ---- each candidates_func, byte-exact vs its optuna_integration twin ------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void LogEi_CandidatesFunc_ByteExact()
        => AssertCandidatesFuncByteExact(BoTorchCandidates.LogEi, "logei_candidates_func", pending: null, seed: 0);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Qei_CandidatesFunc_ByteExact()
        => AssertCandidatesFuncByteExact(BoTorchCandidates.Qei, "qei_candidates_func", pending: null, seed: 1);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Qei_WithPending_CandidatesFunc_ByteExact()
        => AssertCandidatesFuncByteExact(BoTorchCandidates.Qei, "qei_candidates_func", pending: Pending, seed: 2);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void QLogEi_CandidatesFunc_ByteExact()
        => AssertCandidatesFuncByteExact(BoTorchCandidates.QLogEi, "qlogei_candidates_func", pending: null, seed: 3);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void QNei_CandidatesFunc_ByteExact()
        => AssertCandidatesFuncByteExact(BoTorchCandidates.QNei, "qnei_candidates_func", pending: null, seed: 4);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void QNei_WithPending_CandidatesFunc_ByteExact()
        => AssertCandidatesFuncByteExact(BoTorchCandidates.QNei, "qnei_candidates_func", pending: Pending, seed: 5);

    // ---- default-selection parity (Resolve == _get_default_candidates_func) ----------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Resolve_SingleObjective_MatchesPythonDefaultSelection()
    {
        RequireBoTorchOrInconclusive();

        // C# enum → implementation.
        BoTorchCandidates.Resolve(BoTorchCandidatesFunc.Auto, 1, hasConstraint: false, considerRunningTrials: false)
            .Should().Be((NativeCandidatesFunc)BoTorchCandidates.LogEi);
        BoTorchCandidates.Resolve(BoTorchCandidatesFunc.Auto, 1, hasConstraint: false, considerRunningTrials: true)
            .Should().Be((NativeCandidatesFunc)BoTorchCandidates.Qei);
        BoTorchCandidates.Resolve(BoTorchCandidatesFunc.QLogEi, 1, false, false)
            .Should().Be((NativeCandidatesFunc)BoTorchCandidates.QLogEi);
        BoTorchCandidates.Resolve(BoTorchCandidatesFunc.QNei, 1, false, false)
            .Should().Be((NativeCandidatesFunc)BoTorchCandidates.QNei);

        // Python: _get_default_candidates_func picks the same-named function.
        PyExec("import optuna_integration as integration\n" +
               "from optuna_integration.botorch.botorch import _get_default_candidates_func\n");
        PyBool("_get_default_candidates_func(n_objectives=1, has_constraint=False, consider_running_trials=False) " +
               "is integration.botorch.logei_candidates_func").Should().BeTrue("Auto single-objective → logEI, like Optuna");
        PyBool("_get_default_candidates_func(n_objectives=1, has_constraint=False, consider_running_trials=True) " +
               "is integration.botorch.qei_candidates_func").Should().BeTrue("Auto + running trials → qEI, like Optuna");
    }

    // ---- tensor marshaling (double[,] → torch.Tensor) --------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void ToTensor_IsFloat64_AndRoundTrips()
    {
        RequireBoTorchOrInconclusive();
        var data = new double[,] { { 1.5, -2.25, 3e10 }, { 0.0, 1.0 / 3.0, -7.0 } };

        (string Dtype, double[] Flat) result = PythonRuntime.Run(() =>
        {
            using PyObject t = BoTorchInterop.ToTensor(data);
            using PyObject dtypeObj = t.GetAttr("dtype");
            using PyObject dtypeName = dtypeObj.InvokeMethod("__str__");
            return (dtypeName.As<string>(), Flatten(t));
        });

        result.Dtype.Should().Be("torch.float64", "BoTorch's SingleTaskGP fits in float64 — a float32 tensor would silently break parity");
        result.Flat.Should().Equal(new[] { 1.5, -2.25, 3e10, 0.0, 1.0 / 3.0, -7.0 }, "row-major values must round-trip exactly");
    }

    // ---- error taxonomy --------------------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void MultiValueObjective_RaisesValueError_LikePython()
    {
        RequireBoTorchOrInconclusive();
        var multiObj = new double[,] { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 }, { 9, 10 } };  // (n, 2)

        string csMessage = PythonRuntime.Run(() =>
        {
            using PyObject tx = BoTorchInterop.ToTensor(TrainX);
            using PyObject to = BoTorchInterop.ToTensor(multiObj);
            using PyObject bd = BoTorchInterop.ToTensor(BoundsT);
            try
            {
                using PyObject _ = BoTorchCandidates.LogEi(tx, to, null, bd, null);
                return "(no error)";
            }
            catch (NumSharp.ValueError e)
            {
                return e.Message;
            }
        });
        csMessage.Should().Be("Objective may only contain single values with logEI.");

        // Python raises the identical message.
        PyExec("import optuna_integration as integration, torch\n" +
               "_msg = None\n" +
               "try:\n" +
               "    integration.botorch.logei_candidates_func(torch.zeros((5, 2), dtype=torch.float64), " +
               "torch.zeros((5, 2), dtype=torch.float64), None, torch.tensor([[0., 0.], [1., 1.]], dtype=torch.float64), None)\n" +
               "except ValueError as e:\n" +
               "    _msg = str(e)\n");
        PyStr("_msg").Should().Be("Objective may only contain single values with logEI.");
    }

    [TestMethod]
    public void MultiObjective_Resolve_NotSupported_PhaseA()
        => new Action(() => BoTorchCandidates.Resolve(BoTorchCandidatesFunc.Auto, 2, false, false))
            .Should().Throw<NotSupportedException>("multi-objective is Phase B");

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Constraints_CandidatesFunc_NotSupported_PhaseA()
    {
        RequireBoTorchOrInconclusive();
        bool rejected = PythonRuntime.Run(() =>
        {
            using PyObject tx = BoTorchInterop.ToTensor(TrainX);
            using PyObject to = BoTorchInterop.ToTensor(TrainObj);
            using PyObject bd = BoTorchInterop.ToTensor(BoundsT);
            using PyObject con = BoTorchInterop.ToTensor(new double[,] { { 0.1 }, { -0.2 }, { 0.3 }, { -0.4 }, { 0.5 } });
            try
            {
                using PyObject _ = BoTorchCandidates.LogEi(tx, to, con, bd, null);
                return false;
            }
            catch (NotSupportedException)
            {
                return true;
            }
        });
        rejected.Should().BeTrue("constrained optimization is Phase C — the candidates_func must reject a non-null train_con for now");
    }

    // ---- harness --------------------------------------------------------------------------------

    private void AssertCandidatesFuncByteExact(
        NativeCandidatesFunc csFunc, string pyFuncName, double[,]? pending, int seed)
    {
        RequireBoTorchOrInconclusive();

        (double[] Cs, double[] Py) result = PythonRuntime.Run(() =>
        {
            using PyObject tx = BoTorchInterop.ToTensor(TrainX);
            using PyObject to = BoTorchInterop.ToTensor(TrainObj);
            using PyObject bd = BoTorchInterop.ToTensor(BoundsT);
            PyObject? pd = pending is null ? null : BoTorchInterop.ToTensor(pending);
            try
            {
                // C# side — the ported candidates_func over these tensors under manual_seed.
                double[] cs;
                PyObject csCand;
                using (Python.botorch.manual_seed(seed))
                    csCand = csFunc(tx, to, null, bd, pd);
                using (csCand)
                    cs = Flatten(csCand);

                // Python side — the real optuna_integration candidates_func over the SAME tensors + seed.
                using PyModule scope = Py.CreateScope();
                scope.Exec("import optuna_integration as integration\nfrom botorch.utils.sampling import manual_seed\n");
                scope.Set("tx", tx);
                scope.Set("to", to);
                scope.Set("bd", bd);
                scope.Set("pd", (PyObject?)pd ?? PyObject.None);
                scope.Exec(
                    $"with manual_seed({seed}):\n" +
                    $"    _c = integration.botorch.{pyFuncName}(tx, to, None, bd, pd)\n" +
                    "_c = _c.detach().reshape(-1).tolist()\n");
                double[] py;
                using (PyObject pyList = scope.Eval("_c"))
                    py = ToDoubles(pyList);

                return (cs, py);
            }
            finally
            {
                pd?.Dispose();
            }
        });

        result.Cs.Should().Equal(result.Py,
            $"the ported {pyFuncName} must byte-match optuna_integration.botorch.{pyFuncName} on the same tensors + seed");
    }

    private static double[] Flatten(PyObject tensor)
    {
        using PyObject detached = tensor.InvokeMethod("detach");
        using PyObject neg1 = (-1).ToPython();
        using PyObject flat = detached.InvokeMethod("reshape", neg1);
        using PyObject list = flat.InvokeMethod("tolist");
        return ToDoubles(list);
    }

    private static double[] ToDoubles(PyObject list)
    {
        long n = list.Length();
        var result = new double[n];
        for (long i = 0; i < n; i++)
            using (PyObject e = list[new PyInt(i)])
                result[i] = e.As<double>();
        return result;
    }

    private void RequireBoTorchOrInconclusive()
    {
        bool available;
        try
        {
            PyExec("try:\n    import botorch, optuna_integration\n    _have_botorch = True\n" +
                   "except Exception:\n    _have_botorch = False\n");
            available = PyBool("_have_botorch");
        }
        catch
        {
            available = false;
        }
        if (!available)
            Assert.Inconclusive("botorch / optuna_integration are not installed in the interop interpreter.");
    }
}
