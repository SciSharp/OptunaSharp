using System;
using AwesomeAssertions;
using Python.Runtime;
using OptunaSharp.Interop.BoTorch;
using OptunaSharp.Interop.PythonNet;
// `using Python.Runtime;` makes the bare `Python` resolve to that namespace, so the bridge's pythonic
// surface (the `OptunaSharp.Interop.BoTorch.Python` static class) needs an explicit alias here.
using PyBridge = OptunaSharp.Interop.BoTorch.Python;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     <b>Function-level</b> parity of the BoTorch API the bridge implements — the complement to the
///     end-to-end <see cref="BoTorchSamplerParityTests" />. Each ported <c>candidates_func</c> is called
///     directly over pinned <c>torch.Tensor</c>s and asserted <b>byte-for-byte identical</b> to the real
///     <c>optuna_integration.botorch.*_candidates_func</c> given the <em>same tensors</em> under the
///     <em>same</em> <c>manual_seed</c> — so a wrong kwarg, a wrong <c>num_restarts</c>/<c>raw_samples</c>,
///     a dropped <c>X_pending</c>, or a mis-ported constraint/hypervolume branch turns the gate red.
///     Covers the single-objective, multi-objective, constrained, and knowledge-gradient acquisitions;
///     also pins the default-selection logic, the tensor marshaling, and the error taxonomy.
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
    private static readonly double[,] TrainX6 =
        { { 0.1, 0.2 }, { 0.7, 0.3 }, { 0.4, 0.9 }, { 0.55, 0.15 }, { 0.85, 0.6 }, { 0.2, 0.5 } };
    private static readonly double[,] Obj1 = { { 1.0 }, { 2.0 }, { 0.5 }, { 1.5 }, { 0.8 } };
    private static readonly double[,] Obj2 =
        { { 1.0, 0.5 }, { 2.0, 0.2 }, { 0.5, 1.1 }, { 1.5, 0.9 }, { 0.8, 1.3 } };
    private static readonly double[,] Obj4 =
    {
        { 1.0, 0.5, 0.3, 0.9 }, { 2.0, 0.2, 1.1, 0.4 }, { 0.5, 1.1, 0.7, 0.2 },
        { 1.5, 0.9, 0.4, 1.2 }, { 0.8, 1.3, 0.6, 0.5 }, { 0.3, 0.7, 1.0, 0.8 },
    };
    private static readonly double[,] BoundsT = { { 0.0, 0.0 }, { 1.0, 1.0 } };   // (2, d)
    private static readonly double[,] Pending = { { 0.3, 0.6 }, { 0.8, 0.1 } };   // (n_pending, d)
    private static readonly double[,] Con1 = { { 0.1 }, { -0.2 }, { 0.3 }, { -0.4 }, { 0.5 } };  // mixed feas/infeas

    // ---- single-objective, byte-exact vs the optuna_integration twin -----------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void LogEi_CandidatesFunc_ByteExact()
        => AssertByteExact(BoTorchCandidates.LogEi, "logei_candidates_func", Obj1, con: null, pending: null, seed: 0);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void LogEi_Constrained_CandidatesFunc_ByteExact()
        => AssertByteExact(BoTorchCandidates.LogEi, "logei_candidates_func", Obj1, con: Con1, pending: null, seed: 10);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Qei_CandidatesFunc_ByteExact()
        => AssertByteExact(BoTorchCandidates.Qei, "qei_candidates_func", Obj1, con: null, pending: null, seed: 1);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Qei_WithPending_CandidatesFunc_ByteExact()
        => AssertByteExact(BoTorchCandidates.Qei, "qei_candidates_func", Obj1, con: null, pending: Pending, seed: 2);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Qei_Constrained_CandidatesFunc_ByteExact()
        => AssertByteExact(BoTorchCandidates.Qei, "qei_candidates_func", Obj1, con: Con1, pending: null, seed: 11);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void QLogEi_CandidatesFunc_ByteExact()
        => AssertByteExact(BoTorchCandidates.QLogEi, "qlogei_candidates_func", Obj1, con: null, pending: null, seed: 3);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void QLogEi_Constrained_CandidatesFunc_ByteExact()
        => AssertByteExact(BoTorchCandidates.QLogEi, "qlogei_candidates_func", Obj1, con: Con1, pending: null, seed: 12);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void QNei_CandidatesFunc_ByteExact()
        => AssertByteExact(BoTorchCandidates.QNei, "qnei_candidates_func", Obj1, con: null, pending: null, seed: 4);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void QNei_WithPending_CandidatesFunc_ByteExact()
        => AssertByteExact(BoTorchCandidates.QNei, "qnei_candidates_func", Obj1, con: null, pending: Pending, seed: 5);

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void QNei_Constrained_CandidatesFunc_ByteExact()
        => AssertByteExact(BoTorchCandidates.QNei, "qnei_candidates_func", Obj1, con: Con1, pending: null, seed: 13);

    // ---- multi-objective, byte-exact -------------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void Qehvi_TwoObjectives_ByteExact()
        => AssertByteExact(BoTorchCandidates.Qehvi, "qehvi_candidates_func", Obj2, con: null, pending: null, seed: 20);

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void Qehvi_WithPending_ByteExact()
        => AssertByteExact(BoTorchCandidates.Qehvi, "qehvi_candidates_func", Obj2, con: null, pending: Pending, seed: 21);

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void Qehvi_Constrained_ByteExact()
        => AssertByteExact(BoTorchCandidates.Qehvi, "qehvi_candidates_func", Obj2, con: Con1, pending: null, seed: 22);

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void Qehvi_ManyObjectives_AlphaBranch_ByteExact()
        => AssertByteExact(BoTorchCandidates.Qehvi, "qehvi_candidates_func", Obj4, con: null, pending: null, seed: 23, trainX: TrainX6);

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void Ehvi_TwoObjectives_ByteExact()
        => AssertByteExact(BoTorchCandidates.Ehvi, "ehvi_candidates_func", Obj2, con: null, pending: null, seed: 24);

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Ehvi_ManyObjectives_AlphaBranch_ByteExact()
        => AssertByteExact(BoTorchCandidates.Ehvi, "ehvi_candidates_func", Obj4, con: null, pending: null, seed: 25, trainX: TrainX6);

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void Qnehvi_TwoObjectives_ByteExact()
        => AssertByteExact(BoTorchCandidates.Qnehvi, "qnehvi_candidates_func", Obj2, con: null, pending: null, seed: 26);

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void Qnehvi_Constrained_ByteExact()
        => AssertByteExact(BoTorchCandidates.Qnehvi, "qnehvi_candidates_func", Obj2, con: Con1, pending: null, seed: 27);

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void QParEgo_TwoObjectives_ByteExact()
        => AssertByteExact(BoTorchCandidates.QParEgo, "qparego_candidates_func", Obj2, con: null, pending: null, seed: 28);

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void QParEgo_ManyObjectives_ByteExact()
        => AssertByteExact(BoTorchCandidates.QParEgo, "qparego_candidates_func", Obj4, con: null, pending: null, seed: 29, trainX: TrainX6);

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void QParEgo_Constrained_ByteExact()
        => AssertByteExact(BoTorchCandidates.QParEgo, "qparego_candidates_func", Obj2, con: Con1, pending: null, seed: 30);

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void QLogEiParEgo_TwoObjectives_ByteExact()
        => AssertByteExact(BoTorchCandidates.QLogEiParEgo, "qlogei_parego_candidates_func", Obj2, con: null, pending: null, seed: 31);

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void QLogEiParEgo_Constrained_ByteExact()
        => AssertByteExact(BoTorchCandidates.QLogEiParEgo, "qlogei_parego_candidates_func", Obj2, con: Con1, pending: null, seed: 32);

    // ---- knowledge gradient, byte-exact ---------------------------------------------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void Qkg_SingleObjective_ByteExact()
        => AssertByteExact(BoTorchCandidates.Qkg, "qkg_candidates_func", Obj1, con: null, pending: null, seed: 40);

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void Qhvkg_TwoObjectives_ByteExact()
        => AssertByteExact(BoTorchCandidates.Qhvkg, "qhvkg_candidates_func", Obj2, con: null, pending: null, seed: 42);

    [TestMethod]
    [TestCategory("RequiresPython")]
    [TestCategory("Slow")]
    public void Qhvkg_Constrained_ByteExact()
        => AssertByteExact(BoTorchCandidates.Qhvkg, "qhvkg_candidates_func", Obj2, con: Con1, pending: null, seed: 43);

    // ---- default-selection parity (Resolve == _get_default_candidates_func) ----------------------

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Resolve_MatchesPythonDefaultSelection()
    {
        RequireBoTorchOrInconclusive();

        // C# enum → implementation, over the full _get_default_candidates_func decision surface.
        Resolved(BoTorchCandidatesFunc.Auto, 1, false, false).Should().Be((NativeCandidatesFunc)BoTorchCandidates.LogEi);
        Resolved(BoTorchCandidatesFunc.Auto, 1, false, true).Should().Be((NativeCandidatesFunc)BoTorchCandidates.Qei);
        Resolved(BoTorchCandidatesFunc.Auto, 2, false, false).Should().Be((NativeCandidatesFunc)BoTorchCandidates.Qehvi);
        Resolved(BoTorchCandidatesFunc.Auto, 3, false, false).Should().Be((NativeCandidatesFunc)BoTorchCandidates.Qehvi);
        Resolved(BoTorchCandidatesFunc.Auto, 4, false, false).Should().Be((NativeCandidatesFunc)BoTorchCandidates.Ehvi);
        Resolved(BoTorchCandidatesFunc.Auto, 4, false, true).Should().Be((NativeCandidatesFunc)BoTorchCandidates.QParEgo);
        Resolved(BoTorchCandidatesFunc.Auto, 4, true, false).Should().Be((NativeCandidatesFunc)BoTorchCandidates.QParEgo);
        Resolved(BoTorchCandidatesFunc.QLogEi, 1, false, false).Should().Be((NativeCandidatesFunc)BoTorchCandidates.QLogEi);
        Resolved(BoTorchCandidatesFunc.Qnehvi, 2, false, false).Should().Be((NativeCandidatesFunc)BoTorchCandidates.Qnehvi);

        // Python: _get_default_candidates_func picks the same-named function on the same inputs.
        PyExec("import optuna_integration as integration\n" +
               "from optuna_integration.botorch.botorch import _get_default_candidates_func\n" +
               "b = integration.botorch\n");
        PyBool("_get_default_candidates_func(1, False, False) is b.logei_candidates_func").Should().BeTrue();
        PyBool("_get_default_candidates_func(1, False, True) is b.qei_candidates_func").Should().BeTrue();
        PyBool("_get_default_candidates_func(2, False, False) is b.qehvi_candidates_func").Should().BeTrue();
        PyBool("_get_default_candidates_func(3, False, False) is b.qehvi_candidates_func").Should().BeTrue();
        PyBool("_get_default_candidates_func(4, False, False) is b.ehvi_candidates_func").Should().BeTrue();
        PyBool("_get_default_candidates_func(4, False, True) is b.qparego_candidates_func").Should().BeTrue();
        PyBool("_get_default_candidates_func(4, True, False) is b.qparego_candidates_func").Should().BeTrue();
    }

    private static NativeCandidatesFunc Resolved(BoTorchCandidatesFunc s, int nObj, bool con, bool running)
        => BoTorchCandidates.Resolve(s, nObj, con, running);

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
    [TestCategory("RequiresPython")]
    public void Qkg_MultiValueObjective_RaisesValueError_LikePython()
    {
        RequireBoTorchOrInconclusive();
        string csMessage = PythonRuntime.Run(() =>
        {
            using PyObject tx = BoTorchInterop.ToTensor(TrainX);
            using PyObject to = BoTorchInterop.ToTensor(Obj2);
            using PyObject bd = BoTorchInterop.ToTensor(BoundsT);
            try
            {
                using PyObject _ = BoTorchCandidates.Qkg(tx, to, null, bd, null);
                return "(no error)";
            }
            catch (NumSharp.ValueError e)
            {
                return e.Message;
            }
        });
        csMessage.Should().Be("Objective may only contain single values with qKG.");
    }

    [TestMethod]
    [TestCategory("RequiresPython")]
    public void Ehvi_WithConstraints_RaisesValueError_LikePython()
    {
        RequireBoTorchOrInconclusive();
        string csMessage = PythonRuntime.Run(() =>
        {
            using PyObject tx = BoTorchInterop.ToTensor(TrainX);
            using PyObject to = BoTorchInterop.ToTensor(Obj2);
            using PyObject bd = BoTorchInterop.ToTensor(BoundsT);
            using PyObject con = BoTorchInterop.ToTensor(Con1);
            try
            {
                using PyObject _ = BoTorchCandidates.Ehvi(tx, to, con, bd, null);
                return "(no error)";
            }
            catch (NumSharp.ValueError e)
            {
                return e.Message;
            }
        });
        csMessage.Should().Be("Constraints are not supported with ehvi_candidates_func.");

        // Python raises the identical message.
        PyExec("import optuna_integration as integration, torch\n" +
               "_emsg = None\n" +
               "try:\n" +
               "    integration.botorch.ehvi_candidates_func(torch.zeros((5, 2), dtype=torch.float64), " +
               "torch.zeros((5, 2), dtype=torch.float64), torch.zeros((5, 1), dtype=torch.float64), " +
               "torch.tensor([[0., 0.], [1., 1.]], dtype=torch.float64), None)\n" +
               "except ValueError as e:\n" +
               "    _emsg = str(e)\n");
        PyStr("_emsg").Should().Be("Constraints are not supported with ehvi_candidates_func.");
    }

    // ---- harness --------------------------------------------------------------------------------

    private void AssertByteExact(
        NativeCandidatesFunc csFunc, string pyFuncName, double[,] trainObj, double[,]? con, double[,]? pending,
        int seed, double[,]? trainX = null)
    {
        RequireBoTorchOrInconclusive();
        trainX ??= TrainX;

        (double[] Cs, double[] Py) result = PythonRuntime.Run(() =>
        {
            using PyObject tx = BoTorchInterop.ToTensor(trainX);
            using PyObject to = BoTorchInterop.ToTensor(trainObj);
            using PyObject bd = BoTorchInterop.ToTensor(BoundsT);
            PyObject? tc = con is null ? null : BoTorchInterop.ToTensor(con);
            PyObject? pd = pending is null ? null : BoTorchInterop.ToTensor(pending);
            try
            {
                // C# side — the ported candidates_func over these tensors under manual_seed.
                double[] cs;
                PyObject csCand;
                using (PyBridge.botorch.manual_seed(seed))
                    csCand = csFunc(tx, to, tc, bd, pd);
                using (csCand)
                    cs = Flatten(csCand);

                // Python side — the real optuna_integration candidates_func over the SAME tensors + seed.
                using PyModule scope = Py.CreateScope();
                scope.Exec("import optuna_integration as integration\nfrom botorch.utils.sampling import manual_seed\n");
                scope.Set("tx", tx);
                scope.Set("to", to);
                scope.Set("bd", bd);
                scope.Set("tc", (PyObject?)tc ?? PyObject.None);
                scope.Set("pd", (PyObject?)pd ?? PyObject.None);
                scope.Exec(
                    $"with manual_seed({seed}):\n" +
                    $"    _c = integration.botorch.{pyFuncName}(tx, to, tc, bd, pd)\n" +
                    "_c = _c.detach().reshape(-1).tolist()\n");
                double[] py;
                using (PyObject pyList = scope.Eval("_c"))
                    py = ToDoubles(pyList);

                return (cs, py);
            }
            finally
            {
                tc?.Dispose();
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
