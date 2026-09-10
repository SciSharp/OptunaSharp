using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace OptunaSharp.Interop.Pymoo;

/// <summary>
///     Marshaling between C# parameter/objective matrices and the numpy arrays pymoo consumes and
///     produces, done through NumSharp's zero-copy pythonnet interop (<c>NumSharp.Interop.pythonnet</c>)
///     — <c>NDArray</c> ⇄ numpy — rather than per-element Python-object loops (the anti-pattern the
///     <c>pythonic</c> skill §4 replaces). A <c>double[]</c>/<c>double[,]</c> is wrapped as an
///     <see cref="NDArray" /> view over the pinned managed array (<c>np.array(x, copy: false)</c> →
///     <c>GCHandle.Alloc(x, Pinned)</c>) and copied once into an independent numpy array via
///     <see cref="NDArrayPythonInterop.ToNumpyCopy" />; a numpy result crosses back to an independent
///     <see cref="NDArray" /> in one bulk copy via <see cref="NDArrayPythonInterop.ToNDArray" />.
///     <para><b>Why the copy (not a view) on the way in.</b> pymoo <em>retains</em> the arrays we hand
///     it — the problem's <c>xl</c>/<c>xu</c> for its whole life, and each generation's objective matrix
///     <c>F</c> as per-individual row views held by the <c>Population</c>. Handing pymoo an independent
///     numpy array (rather than a view over pinned CLR memory) decouples its lifetime from the CLR array
///     entirely, so nothing pymoo keeps can outlive a freed pin. The data is tiny (population × dimension
///     / objectives), so the copy is immaterial. The bulk data movement is still NumSharp's vectorized
///     crossing, never a hand-rolled element loop.</para>
///     <para><b>GIL discipline:</b> every method touches PyObjects and MUST be called under the GIL (from
///     a <see cref="OptunaSharp.Interop.PythonNet.PythonRuntime.Run(System.Action)" /> body); NumSharp's
///     crossing re-acquires the GIL re-entrantly and requires the engine to be up already, which
///     <c>PythonRuntime</c> guarantees.</para>
/// </summary>
internal static class PymooInterop
{
    /// <summary>A 1-D vector → an independent numpy 1-D <c>float64</c> array (bulk copy, no per-element
    /// loop). Dispose the returned object. Under the GIL.</summary>
    internal static PyObject ToNumpy(double[] data) => np.array(data, copy: false).ToNumpyCopy();

    /// <summary>A <c>(rows, cols)</c> matrix → an independent numpy 2-D <c>float64</c> array (bulk copy).
    /// Dispose the returned object. Under the GIL.</summary>
    internal static PyObject ToNumpy(double[,] data) => np.array(data, copy: false).ToNumpyCopy();

    /// <summary>A numpy 2-D array (e.g. a population's <c>X</c> matrix from <c>ask()</c>) → an
    /// independent <see cref="NDArray" /> in one bulk copy (no per-element loop), detached so the source
    /// may be disposed afterwards. Under the GIL.</summary>
    internal static NDArray ToNDArray(PyObject numpyMatrix) => numpyMatrix.ToNDArray();
}
