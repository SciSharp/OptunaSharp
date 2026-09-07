using Python;
using Python.Runtime;

namespace OptunaSharp.Interop.BoTorch;

/// <summary>
///     The BoTorch-specific pythonnet marshaling the bridge is built on: <c>double[,]</c>/<c>double[]</c>
///     ⇄ <c>torch.Tensor(float64)</c>. It builds tensors through the Pythonic <see cref="Python.torch" />
///     facade (<c>torch.tensor(list)</c>) and reads them back through the tensor extension members
///     (<c>t.detach().cpu().tolist()</c>), so this file is pure data crossing — the reproducibility scope
///     lives on the facade as <c>botorch.manual_seed</c>, and module import + caching / GIL / interpreter
///     discovery come from the shared <see cref="OptunaSharp.Interop.PythonNet.PythonRuntime" />.
///     <para><b>GIL discipline:</b> every method here touches <see cref="PyObject" />s and MUST be called
///     under the GIL — i.e. from inside a
///     <see cref="OptunaSharp.Interop.PythonNet.PythonRuntime.Run(System.Action)" /> body.</para>
/// </summary>
internal static class BoTorchInterop
{
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
        return torch.tensor(outer);
    }

    /// <summary>A 1-D array → a <c>torch.Tensor</c> of dtype <c>float64</c>. Under the GIL.</summary>
    internal static PyObject ToTensor(double[] data)
    {
        using var list = new PyList();
        foreach (double x in data)
            using (PyObject v = x.ToPython())
                list.Append(v);
        return torch.tensor(list);
    }

    /// <summary>
    ///     A 1-D <c>torch.Tensor</c> → a C# array (via <c>.detach().cpu().tolist()</c> → Python
    ///     <c>float</c>s, which are C <c>double</c>s — lossless). Under the GIL.
    /// </summary>
    internal static double[] ToArray1D(PyObject tensor)
    {
        using PyObject detached = tensor.detach();
        using PyObject cpu = detached.cpu();
        using PyObject list = cpu.tolist();
        long n = list.Length();
        var result = new double[n];
        for (long i = 0; i < n; i++)
            using (PyObject item = list[new PyInt(i)])
                result[i] = item.As<double>();
        return result;
    }
}
