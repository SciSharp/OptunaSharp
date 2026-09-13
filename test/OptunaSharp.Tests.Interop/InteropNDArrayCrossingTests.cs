using System;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;
using OptunaSharp.Interop.PythonNet;
using BoTorch = OptunaSharp.Interop.BoTorch.BoTorchInterop;
using PyCma = OptunaSharp.Interop.PyCma.PyCmaInterop;

namespace OptunaSharp.Tests.Interop;

/// <summary>
///     Covers the <see cref="NDArray" />-accepting interop overloads — the ones added so a caller already
///     holding an <see cref="NDArray" /> crosses to Python WITHOUT a detour through a managed array:
///     <c>BoTorchInterop.ToTensor(NDArray)</c> / <c>ToTensor(NDArray, device)</c> and
///     <c>PyCmaInterop.ToPyList(NDArray)</c> / <c>ToNDArray(PyObject)</c>. (Pymoo was already
///     NDArray-native.) Each is asserted for correct values + dtype, PARITY with the pre-existing
///     raw-array overload (the NDArray path is not a behavioral fork), the zero-copy VIEW semantics (a
///     write through the tensor is visible in the source NDArray — proof no copy was taken), and the
///     BORROWED-ownership contract (the tensor keeps working after the caller disposes the source, because
///     the export roots the buffer for the tensor's life via a Python <c>weakref.finalize</c>).
///     <para>These are function-level companions to the end-to-end sampler parity suites. They reach the
///     package internals via <c>InternalsVisibleTo</c> and self-skip (Inconclusive) when the needed Python
///     module (torch / numpy) is absent, like every other <c>RequiresPython</c> test.</para>
/// </summary>
[TestClass]
public class InteropNDArrayCrossingTests : OptunaInteropTestBase
{
    // ---- BoTorch: NDArray -> torch.Tensor --------------------------------------------------------

    /// <summary>ToTensor(NDArray) yields a float64 tensor whose row-major values round-trip exactly
    /// (float64 because botorch.py's SingleTaskGP fits in it — a float32 tensor would break parity).</summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void BoTorch_ToTensor_NDArray_IsFloat64_AndRoundTrips()
    {
        RequireModulesOrInconclusive("torch");

        (string Dtype, double[] Flat) result = PythonRuntime.Run(() =>
        {
            using NDArray nd = np.array(new double[,] { { 1.5, -2.25, 3e10 }, { 0.0, 1.0 / 3.0, -7.0 } }, copy: false);
            using PyObject t = BoTorch.ToTensor(nd);
            using PyObject dtypeObj = t.GetAttr("dtype");
            using PyObject dtypeName = dtypeObj.InvokeMethod("__str__");
            return (dtypeName.As<string>(), Flatten(t));
        });

        result.Dtype.Should().Be("torch.float64");
        result.Flat.Should().Equal(1.5, -2.25, 3e10, 0.0, 1.0 / 3.0, -7.0);
    }

    /// <summary>ToTensor(NDArray) produces the SAME tensor as the double[,] overload for identical data —
    /// the NDArray path is not a behavioral fork.</summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void BoTorch_ToTensor_NDArray_MatchesDoubleMatrixOverload()
    {
        RequireModulesOrInconclusive("torch");
        var data = new double[,] { { 0.1, 0.2 }, { 0.7, 0.3 }, { 0.4, 0.9 } };

        (double[] ViaNd, double[] ViaArr) result = PythonRuntime.Run(() =>
        {
            using NDArray nd = np.array(data, copy: false);
            using PyObject tNd = BoTorch.ToTensor(nd);
            using PyObject tArr = BoTorch.ToTensor(data);
            return (Flatten(tNd), Flatten(tArr));
        });

        result.ViaNd.Should().Equal(result.ViaArr, "the NDArray and double[,] overloads must agree");
        result.ViaNd.Should().Equal(0.1, 0.2, 0.7, 0.3, 0.4, 0.9);
    }

    /// <summary>ToTensor(NDArray) is a zero-copy VIEW: a write THROUGH the tensor is visible in the source
    /// NDArray's buffer — so no copy was taken.</summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void BoTorch_ToTensor_NDArray_IsZeroCopyView()
    {
        RequireModulesOrInconclusive("torch");

        double[] afterWrite = PythonRuntime.Run(() =>
        {
            using NDArray nd = np.array(new double[] { 1.0, 2.0, 3.0 }, copy: false);
            using PyObject t = BoTorch.ToTensor(nd);
            // Write element 0 on the TENSOR side; a shared buffer means the NDArray sees it too.
            using (PyObject idx = new PyInt(0))
            using (PyObject val = (99.0).ToPython())
                t[idx] = val;
            return nd.ToArray<double>();
        });

        afterWrite.Should().Equal(new[] { 99.0, 2.0, 3.0 }, "torch.from_numpy shares the buffer NumSharp pinned");
    }

    /// <summary>The borrowed-ownership contract: ToTensor(NDArray) does NOT dispose the source, and the
    /// tensor keeps working after the caller disposes it — the export roots the buffer for the tensor's life.</summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void BoTorch_ToTensor_NDArray_SurvivesSourceDisposal()
    {
        RequireModulesOrInconclusive("torch");

        double[] flat = PythonRuntime.Run(() =>
        {
            NDArray nd = np.array(new double[] { 2.0, 4.0, 8.0, 16.0 }, copy: false);
            PyObject t = BoTorch.ToTensor(nd);
            nd.Dispose();               // caller disposes the borrowed source immediately after crossing
            using (t)
                return Flatten(t);      // the tensor must still read the correct values
        });

        flat.Should().Equal(2.0, 4.0, 8.0, 16.0);
    }

    /// <summary>ToTensor(NDArray, "cpu") returns the same zero-copy CPU view as the no-device overload.</summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void BoTorch_ToTensor_NDArray_CpuDevice_EqualsPlain()
    {
        RequireModulesOrInconclusive("torch");

        (double[] Cpu, double[] Plain) result = PythonRuntime.Run(() =>
        {
            using NDArray nd = np.array(new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 } }, copy: false);
            using PyObject tCpu = BoTorch.ToTensor(nd, "cpu");
            using PyObject tPlain = BoTorch.ToTensor(nd);
            return (Flatten(tCpu), Flatten(tPlain));
        });

        result.Cpu.Should().Equal(result.Plain);
        result.Cpu.Should().Equal(1.0, 2.0, 3.0, 4.0);
    }

    // ---- PyCma: NDArray -> Python list, numpy -> NDArray -----------------------------------------

    /// <summary>ToPyList(NDArray) on a vector matches the double[] overload and yields the exact floats.</summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void PyCma_ToPyList_NDArray_Vector_MatchesDoubleArray()
    {
        RequireModulesOrInconclusive("numpy");
        var vec = new double[] { 1.5, -2.5, 3.5, 0.0 };

        (bool Equal, double[] Values) result = PythonRuntime.Run(() =>
        {
            using NDArray nd = np.array(vec, copy: false);
            using PyObject fromNd = PyCma.ToPyList(nd);
            using PyObject fromArr = PyCma.ToPyList(vec);
            using PyObject eq = fromNd.InvokeMethod("__eq__", fromArr);
            return (eq.As<bool>(), ToDoubles(fromNd));
        });

        result.Equal.Should().BeTrue("the NDArray and double[] overloads must produce identical Python lists");
        result.Values.Should().Equal(1.5, -2.5, 3.5, 0.0);
    }

    /// <summary>ToPyList(NDArray) on a matrix matches the double[,] overload (a nested list of lists).</summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void PyCma_ToPyList_NDArray_Matrix_MatchesDoubleMatrix()
    {
        RequireModulesOrInconclusive("numpy");
        var mat = new double[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } };

        (bool Equal, string Repr) result = PythonRuntime.Run(() =>
        {
            using NDArray nd = np.array(mat, copy: false);
            using PyObject fromNd = PyCma.ToPyList(nd);
            using PyObject fromArr = PyCma.ToPyList(mat);
            using PyObject eq = fromNd.InvokeMethod("__eq__", fromArr);
            using PyObject repr = fromNd.InvokeMethod("__repr__");
            return (eq.As<bool>(), repr.As<string>());
        });

        result.Equal.Should().BeTrue("the NDArray and double[,] overloads must produce identical nested lists");
        result.Repr.Should().Be("[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]");
    }

    /// <summary>ToNDArray copies a numpy array back into an independent NDArray with the same values.</summary>
    [TestMethod]
    [TestCategory("RequiresPython")]
    public void PyCma_ToNDArray_RoundTripsNumpyValues()
    {
        RequireModulesOrInconclusive("numpy");

        double[] values = PythonRuntime.Run(() =>
        {
            using NDArray src = np.array(new double[] { 7.0, -1.5, 0.25 }, copy: false);
            using PyObject numpy = src.ToNumpy();          // a genuine numpy array PyObject (a view of src)
            using NDArray back = PyCma.ToNDArray(numpy);   // one bulk copy into an independent NDArray
            return back.ToArray<double>();
        });

        values.Should().Equal(7.0, -1.5, 0.25);
    }

    // ---- harness --------------------------------------------------------------------------------

    /// <summary>Flatten a torch tensor to a managed double[] (detach -> reshape(-1) -> tolist).</summary>
    /// <param name="tensor">The borrowed tensor to read.</param>
    /// <returns>The tensor's values in row-major order.</returns>
    private static double[] Flatten(PyObject tensor)
    {
        using PyObject detached = tensor.InvokeMethod("detach");
        using PyObject neg1 = (-1).ToPython();
        using PyObject flat = detached.InvokeMethod("reshape", neg1);
        using PyObject list = flat.InvokeMethod("tolist");
        return ToDoubles(list);
    }

    /// <summary>Read a flat Python list of numbers into a managed double[].</summary>
    /// <param name="list">The borrowed one-dimensional Python list.</param>
    /// <returns>The list's elements as doubles.</returns>
    private static double[] ToDoubles(PyObject list)
    {
        long n = list.Length();
        var result = new double[n];
        for (long i = 0; i < n; i++)
            using (PyObject e = list[new PyInt(i)])
                result[i] = e.As<double>();
        return result;
    }

    /// <summary>Skip the test (Inconclusive) unless every named Python module imports in the interop
    /// interpreter — keeps a bare machine (no torch / numpy) green, matching the suite's convention.</summary>
    /// <param name="modules">The Python module names that must be importable for the test to be meaningful.</param>
    private void RequireModulesOrInconclusive(params string[] modules)
    {
        foreach (string module in modules)
        {
            bool available;
            try
            {
                PyExec($"try:\n    import {module}\n    _have_mod = True\nexcept Exception:\n    _have_mod = False\n");
                available = PyBool("_have_mod");
            }
            catch
            {
                available = false;
            }
            if (!available)
                Assert.Inconclusive($"{module} is not installed in the interop interpreter.");
        }
    }
}
