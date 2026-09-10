using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace OptunaSharp.Interop.Pymoo;

/// <summary>
///     The array crossing between OptunaSharp's <see cref="NDArray" /> world and the numpy arrays pymoo
///     consumes and produces, done through NumSharp's zero-copy pythonnet interop
///     (<c>NumSharp.Interop.pythonnet</c>) — <c>NDArray</c> ⇄ numpy — rather than per-element
///     Python-object loops (the anti-pattern the <c>pythonic</c> skill §4 replaces). <see cref="NDArray" />
///     is the currency on the C# side throughout: the search-space transform emits and consumes
///     <see cref="NDArray" /> (<c>SearchSpaceTransform.Transform</c> / <c>Untransform</c>), the problem
///     box and the reference points are built as <see cref="NDArray" />, and the ask/tell <c>X</c>/<c>F</c>
///     matrices cross as <see cref="NDArray" /> — so no code hand-rolls an element loop to (un)marshal.
///     <para><b>Why the copy (not a view) on the way in.</b> pymoo <em>retains</em> the arrays we hand
///     it — the problem's <c>xl</c>/<c>xu</c> for its whole life, the injected initial population, and
///     each generation's objective matrix <c>F</c> as per-individual row views held by the
///     <c>Population</c>. Handing pymoo an independent numpy array (via
///     <see cref="NDArrayPythonInterop.ToNumpyCopy" />) rather than a view over CLR-backed memory
///     decouples its lifetime from the C# side entirely, so nothing pymoo keeps can outlive a freed
///     buffer. The data is tiny (population × dimension / objectives), so the copy is immaterial, and the
///     bulk movement is still NumSharp's vectorized crossing, never a hand-rolled element loop.</para>
///     <para><b>GIL discipline:</b> every method touches PyObjects and MUST be called under the GIL (from
///     a <see cref="OptunaSharp.Interop.PythonNet.PythonRuntime.Run(System.Action)" /> body); NumSharp's
///     crossing re-acquires the GIL re-entrantly and requires the engine to be up already, which
///     <c>PythonRuntime</c> guarantees.</para>
/// </summary>
internal static class PymooInterop
{
    /// <summary>An <see cref="NDArray" /> (1-D vector or 2-D matrix) → an independent numpy array in one
    /// bulk copy (no per-element loop). Dispose the returned object. Under the GIL.</summary>
    internal static PyObject ToNumpy(NDArray array) => array.ToNumpyCopy();

    /// <summary>A numpy array (e.g. a population's <c>X</c> matrix from <c>ask()</c>) → an independent
    /// <see cref="NDArray" /> in one bulk copy (no per-element loop), detached so the source may be
    /// disposed afterwards. Under the GIL.</summary>
    internal static NDArray ToNDArray(PyObject numpy) => numpy.ToNDArray();
}
