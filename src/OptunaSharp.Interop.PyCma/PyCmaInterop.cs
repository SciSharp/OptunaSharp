using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace OptunaSharp.Interop.PyCma;

/// <summary>
///     Marshaling between C# parameter vectors/matrices and the values pycma consumes, done through
///     NumSharp's zero-copy pythonnet interop (<c>NumSharp.Interop.pythonnet</c>) — <c>NDArray</c> ⇄ numpy
///     — rather than per-element Python-object loops. A <c>double[]</c>/<c>double[,]</c> is wrapped as an
///     <see cref="NDArray" /> view over the pinned managed array (<c>np.array(x, copy: false)</c> →
///     <c>GCHandle.Alloc(x, Pinned)</c>) and crosses to numpy as a VIEW via
///     <see cref="NDArrayPythonInterop.ToNumpy(NDArray, bool?)" />; a numpy result crosses back to an independent
///     <see cref="NDArray" /> in one bulk copy via <see cref="NDArrayPythonInterop.ToNDArray" />.
///     <para><b>Why the inputs cross as Python <c>list</c>s, not numpy arrays.</b> pycma's
///     <c>CMAEvolutionStrategy.tell</c> is numerically sensitive to the <em>type</em> of its
///     <c>solutions</c>/<c>function_values</c> arguments: feeding numpy arrays produces a covariance
///     update that drifts from feeding Python lists, and the drift compounds across generations (it first
///     shows several generations in). <c>optuna_integration.PyCmaSampler</c> passes plain Python lists
///     (<c>x0</c>, <c>bounds</c>, <c>CMA_stds</c>, and each <c>tell</c>'s <c>xs</c>/<c>ys</c>), so — to stay
///     byte-exact with it — this bridge does too. The C#→numpy crossing still does the bulk data movement
///     (no hand-rolled element loop); a final <c>numpy.tolist()</c> hands pycma the list it wants. The data
///     is tiny (population × dimension), so the extra materialization is immaterial.</para>
///     <para><b>GIL discipline:</b> every method touches PyObjects and MUST be called under the GIL (from a
///     <see cref="OptunaSharp.Interop.PythonNet.PythonRuntime.Run(System.Action)" /> body); NumSharp's
///     crossing re-acquires the GIL re-entrantly and requires the engine to be up already, which
///     <c>PythonRuntime</c> guarantees.</para>
/// </summary>
internal static class PyCmaInterop
{
    /// <summary>A 1-D vector → a Python <c>list</c> of floats (via a numpy view + <c>tolist()</c>, bit-exact,
    /// no per-element C# loop). Dispose the returned list. Under the GIL.</summary>
    /// <param name="data">Managed float64 candidate coordinates to expose as a Python list.</param>
    /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
    /// <exception cref="global::Python.Runtime.PythonException">NumPy array conversion or its tolist call fails.</exception>
    internal static PyObject ToPyList(double[] data)
    {
        // View the vector zero-copy, cross to numpy, then tolist() copies into an independent
        // Python list — so the NDArray and the numpy view are both transient. Dispose them
        // (reverse order: numpy view first, then the NDArray/pin); the list is already built.
        using NDArray nd = np.array(data, copy: false);
        using PyObject numpy = nd.ToNumpy();
        return numpy.InvokeMethod("tolist");
    }

    /// <summary>A <c>(rows, cols)</c> matrix → a Python <c>list</c> of <c>list</c>s of floats (via a numpy
    /// view + <c>tolist()</c>). Dispose the returned list. Under the GIL.</summary>
    /// <param name="data">Managed float64 matrix whose rows become nested Python lists.</param>
    /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
    /// <exception cref="global::Python.Runtime.PythonException">NumPy array conversion or its tolist call fails.</exception>
    internal static PyObject ToPyList(double[,] data)
    {
        // As the vector overload: view zero-copy, cross to numpy, tolist() copies out, then
        // dispose the numpy view and the NDArray/pin (the list no longer references them).
        using NDArray nd = np.array(data, copy: false);
        using PyObject numpy = nd.ToNumpy();
        return numpy.InvokeMethod("tolist");
    }

    /// <summary>A numpy 1-D array (one candidate solution from <c>es.ask()</c>) → an independent
    /// <see cref="double" /> array in one bulk copy (no per-element loop), detached so the source may be
    /// disposed afterwards. Under the GIL.</summary>
    /// <param name="numpyVector">The borrowed one-dimensional NumPy candidate vector.</param>
    /// <returns>An independent managed copy of the NumPy candidate coordinates.</returns>
    /// <exception cref="global::Python.Runtime.PythonException">Reading the NumPy array through the interop bridge fails.</exception>
    internal static double[] ToArray(PyObject numpyVector) => numpyVector.ToNDArray().ToArray<double>();

    /// <summary>
    ///     An <see cref="NDArray" /> (1-D vector → a Python <c>list</c> of floats; 2-D matrix → a
    ///     <c>list</c> of <c>list</c>s) via a numpy view + <c>tolist()</c> — bit-exact, no per-element C#
    ///     loop. The <see cref="NDArray" />-accepting counterpart of the <c>double[]</c>/<c>double[,]</c>
    ///     overloads, for a caller already holding an <see cref="NDArray" /> (it saves the round-trip
    ///     through a managed array). pycma consumes plain Python lists, not numpy arrays (see the type note
    ///     on the class) — so this still materializes a list.
    ///     <para><b>Ownership:</b> <paramref name="data" /> is the CALLER's and is NOT disposed here (unlike
    ///     the raw-array overloads, which own the <see cref="NDArray" /> they build). Only the transient
    ///     numpy view this method creates is disposed; <c>tolist()</c> has already copied the values into an
    ///     independent Python list, so the returned list does not alias <paramref name="data" />.</para>
    ///     Dispose the returned list. Under the GIL.
    /// </summary>
    /// <param name="data">The caller-owned NumSharp array to render as a (nested) Python list; viewed, not copied, and NOT disposed here.</param>
    /// <returns>A new owned Python reference (a <c>list</c>); dispose it while holding the GIL. <paramref name="data" /> remains owned by the caller.</returns>
    /// <exception cref="global::System.ArgumentNullException"><paramref name="data" /> is null.</exception>
    /// <exception cref="global::Python.Runtime.PythonException">NumPy array conversion or its tolist call fails.</exception>
    internal static PyObject ToPyList(NDArray data)
    {
        if (data is null) throw new System.ArgumentNullException(nameof(data));
        // data is BORROWED: view it zero-copy to numpy, tolist() copies into an independent Python list,
        // then dispose ONLY the numpy view we created (never data — the caller owns it).
        using PyObject numpy = data.ToNumpy();
        return numpy.InvokeMethod("tolist");
    }

    /// <summary>
    ///     A numpy array → an independent NumSharp <see cref="NDArray" /> in one bulk copy (no per-element
    ///     loop), detached so the source may be disposed afterwards. The <see cref="NDArray" />-returning
    ///     complement to <see cref="ToArray(PyObject)" /> (which returns a <see cref="double" />[]), for a
    ///     caller that stays in <see cref="NDArray" /> space rather than a managed vector. Under the GIL.
    /// </summary>
    /// <param name="numpy">The borrowed NumPy array whose data is copied into managed storage.</param>
    /// <returns>An independent, C-contiguous NumSharp array whose storage no longer depends on the Python input; the caller owns and disposes it.</returns>
    /// <exception cref="global::System.ArgumentNullException"><paramref name="numpy" /> is null.</exception>
    /// <exception cref="global::Python.Runtime.PythonException">Reading the NumPy array through the interop bridge fails.</exception>
    internal static NDArray ToNDArray(PyObject numpy)
    {
        if (numpy is null) throw new System.ArgumentNullException(nameof(numpy));
        // One bulk copy into a fresh C-contiguous NDArray; it ESCAPES (returned), so the caller owns it.
        return numpy.ToNDArray();
    }
}
