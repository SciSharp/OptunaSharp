using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace OptunaSharp.Interop.BoTorch;

/// <summary>
///     Marshaling between C# matrices and torch tensors, done through NumSharp's zero-copy pythonnet
///     interop (<c>NumSharp.Interop.pythonnet</c>) — <c>NDArray</c> ⇄ numpy ⇄ torch — rather than
///     per-element Python-object loops. A <c>double[,]</c> becomes an <see cref="NDArray" /> (one bulk
///     <c>np.array</c> copy into NumSharp's buffer) and crosses to a <c>torch.Tensor</c> as a VIEW via
///     <see cref="TorchInterop.ToTorch" /> (<c>torch.from_numpy(nd.ToNumpy())</c>; NumSharp roots its
///     buffer for the tensor's lifetime through a Python <c>weakref.finalize</c>), and a result tensor
///     crosses back to an independent <see cref="NDArray" /> in one bulk copy via
///     <see cref="TorchInterop.ToTorchNDArray" /> — no manual element copying in either direction.
///     <para><b>GIL discipline:</b> every method touches PyObjects/torch and MUST be called under the
///     GIL (from a <see cref="OptunaSharp.Interop.PythonNet.PythonRuntime.Run(System.Action)" /> body);
///     NumSharp's crossing re-acquires the GIL re-entrantly and requires the engine to be up already,
///     which <c>PythonRuntime</c> guarantees.</para>
/// </summary>
internal static class BoTorchInterop
{
    /// <summary>
    ///     A C# <c>(rows, cols)</c> matrix → a float64 <c>torch.Tensor</c>, VIEW-shared with the managed
    ///     array itself — NO copy in either stage. <c>np.array(data, copy: false)</c> PINS the matrix
    ///     (<c>GCHandle.Alloc(data, GCHandleType.Pinned)</c>) and wraps its pinned address as an
    ///     <see cref="NDArray" /> view (row-major, float64), then <see cref="TorchInterop.ToTorch" /> is
    ///     <c>torch.from_numpy(nd.ToNumpy())</c> — so numpy and torch read the pinned managed memory
    ///     directly. NumSharp roots the NDArray (and thus the GCHandle pin) for the tensor's lifetime via a
    ///     Python <c>weakref.finalize</c>, and unpins when the last view is released. float64 matches
    ///     botorch.py (<c>SingleTaskGP</c> fits in float64; a float32 tensor would silently break parity).
    ///     Under the GIL.
    /// </summary>
    internal static PyObject ToTensor(double[,] data) => np.array(data, copy: false).ToTorch();

    /// <summary>A 1-D array → a float64 <c>torch.Tensor</c>, pinned-and-viewed like the matrix overload
    /// (no copy). Under the GIL.</summary>
    internal static PyObject ToTensor(double[] data) => np.array(data, copy: false).ToTorch();

    /// <summary>
    ///     A <c>torch.Tensor</c> → an independent NumSharp <see cref="NDArray" /> via NumSharp's torch
    ///     adapter (<c>detach().cpu().resolve_conj().resolve_neg().numpy()</c> → <c>ToNDArray</c>): one bulk
    ///     copy, no per-element loop, and detached from the tensor so the tensor may be disposed
    ///     afterwards. Under the GIL.
    /// </summary>
    internal static NDArray ToNDArray(PyObject tensor) => tensor.ToTorchNDArray(force: true);
}
