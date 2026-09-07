using System;
using Python.Runtime;
using OptunaSharp.Interop.PythonNet;

namespace OptunaSharp.Interop.BoTorch;

/// <summary>
///     The BoTorch-specific pythonnet plumbing the bridge is built on: <c>double[,]</c>⇄
///     <c>torch.Tensor(float64)</c> marshaling and BoTorch's <c>manual_seed</c> scope (the context
///     manager that makes the acquisition optimization reproducible). Module import + caching, GIL, and
///     interpreter discovery come from the shared <see cref="PythonRuntime" /> (this type does not
///     duplicate them — it calls <see cref="PythonRuntime.Import" /> for <c>torch</c>/<c>botorch</c>/… and
///     runs inside <see cref="PythonRuntime.Run(Action)" />).
///     <para><b>GIL discipline:</b> every method here touches <see cref="PyObject" />s and MUST be
///     called under the GIL — i.e. from inside a <see cref="PythonRuntime.Run(Action)" /> body.</para>
/// </summary>
internal static class BoTorchInterop
{
    /// <summary>The <c>torch</c> module (cached by <see cref="PythonRuntime" />). Under the GIL.</summary>
    internal static PyObject Torch => PythonRuntime.Import("torch");

    // ---- tensor marshaling ----------------------------------------------------------------------

    /// <summary>
    ///     A C# <c>(rows, cols)</c> matrix → a <c>torch.Tensor</c> of dtype <c>float64</c>. The values are
    ///     exact <see cref="double" />s, so the tensor holds identical bits to
    ///     <c>torch.from_numpy(numpy.array(...))</c> — the spelling botorch.py uses. Under the GIL.
    /// </summary>
    internal static PyObject ToTensor(double[,] data)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);
        using var outer = new PyList();
        for (int i = 0; i < rows; i++)
        {
            using var row = new PyList();
            for (int j = 0; j < cols; j++)
                using (PyObject v = data[i, j].ToPython())
                    row.Append(v);
            outer.Append(row);
        }
        return TorchTensorFloat64(outer);
    }

    /// <summary>A 1-D array → a <c>torch.Tensor</c> of dtype <c>float64</c>. Under the GIL.</summary>
    internal static PyObject ToTensor(double[] data)
    {
        using var list = new PyList();
        foreach (double x in data)
            using (PyObject v = x.ToPython())
                list.Append(v);
        return TorchTensorFloat64(list);
    }

    private static PyObject TorchTensorFloat64(PyObject data)
    {
        using PyObject tensorFn = Torch.GetAttr("tensor");
        using PyObject float64 = Torch.GetAttr("float64");
        using var kwargs = new PyDict();
        kwargs["dtype"] = float64;
        return tensorFn.Invoke(new[] { data }, kwargs);
    }

    /// <summary>
    ///     A 1-D <c>torch.Tensor</c> → a C# array (via <c>.detach().cpu().tolist()</c> → Python
    ///     <c>float</c>s, which are C <c>double</c>s — lossless). Under the GIL.
    /// </summary>
    internal static double[] ToArray1D(PyObject tensor)
    {
        using PyObject detached = tensor.InvokeMethod("detach");
        using PyObject cpu = detached.InvokeMethod("cpu");
        using PyObject list = cpu.InvokeMethod("tolist");
        long n = list.Length();
        var result = new double[n];
        for (long i = 0; i < n; i++)
            using (PyObject item = list[new PyInt(i)])
                result[i] = item.As<double>();
        return result;
    }

    // ---- reproducibility scope -------------------------------------------------------------------

    /// <summary>
    ///     Run <paramref name="body" /> inside BoTorch's <c>manual_seed(seed)</c> context manager (a
    ///     no-op when <paramref name="seed" /> is <c>null</c>, matching Optuna) so the acquisition
    ///     optimization is reproducible. Returns the body's result. Under the GIL.
    /// </summary>
    internal static PyObject WithManualSeed(int? seed, Func<PyObject> body)
    {
        PyObject sampling = PythonRuntime.Import("botorch.utils.sampling");
        using PyObject manualSeed = sampling.GetAttr("manual_seed");
        using PyObject seedArg = seed is { } s ? s.ToPython() : PyObject.None;
        using PyObject cm = manualSeed.Invoke(seedArg);
        using PyObject enter = cm.GetAttr("__enter__");
        enter.Invoke().Dispose();
        try
        {
            return body();
        }
        finally
        {
            using PyObject exit = cm.GetAttr("__exit__");
            exit.Invoke(PyObject.None, PyObject.None, PyObject.None).Dispose();
        }
    }
}
