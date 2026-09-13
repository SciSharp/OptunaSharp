using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;
using OptunaSharp.Interop.PythonNet;

namespace OptunaSharp.Interop.BoTorch;

/// <summary>
///     Marshaling between C# matrices and torch tensors, done through NumSharp's zero-copy pythonnet
///     interop (<c>NumSharp.Interop.pythonnet</c>) — <c>NDArray</c> ⇄ numpy ⇄ torch — rather than
///     per-element Python-object loops. A <c>double[,]</c> becomes a pinned <see cref="NDArray" /> view
///     through <c>np.array(data, copy: false)</c> and crosses to a <c>torch.Tensor</c> as a VIEW via
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
    /// <param name="data">The managed float64 vector or row-major matrix passed across the Python boundary.</param>
    /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
    /// <exception cref="global::Python.Runtime.PythonException">The Python tensor, array, model-fitting, or acquisition operation fails; the original Python diagnostic is preserved.</exception>
    internal static PyObject ToTensor(double[,] data)
    {
        // The tensor is a VIEW over the pinned matrix; NumSharp roots the buffer (and the
        // GCHandle pin) for the tensor's lifetime via weakref.finalize, so disposing this
        // NDArray now is safe — the pin outlives it and releases with the last torch view.
        using NDArray nd = np.array(data, copy: false);
        return nd.ToTorch();
    }

    /// <summary>
    ///     A C# <c>(rows, cols)</c> matrix → a float64 <c>torch.Tensor</c> placed on <paramref name="device" />
    ///     — botorch.py's <c>torch.from_numpy(x).to(self._device)</c>. For CPU (or a <c>null</c>/empty device)
    ///     this is the zero-copy pinned view of the matrix overload; for any other device (e.g. <c>"cuda"</c>,
    ///     <c>"cuda:0"</c>) the pinned CPU view is created, copied to the device via <c>tensor.to(device)</c>
    ///     (a device tensor cannot alias host pinned memory — the copy is correct), and the CPU view (and its
    ///     pin) is released. Under the GIL.
    /// </summary>
    /// <param name="data">The managed float64 vector or row-major matrix passed across the Python boundary.</param>
    /// <param name="device">The torch device name; null, empty, or cpu keeps the pinned CPU view, while other devices receive a copy.</param>
    /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
    /// <exception cref="global::Python.Runtime.PythonException">The Python tensor, array, model-fitting, or acquisition operation fails; the original Python diagnostic is preserved.</exception>
    internal static PyObject ToTensor(double[,] data, string? device)
    {
        // Same zero-copy view as the CPU overload; the NDArray is disposed once the tensor
        // exists (its buffer is rooted for the tensor's life — see the (double[,]) overload).
        PyObject tensor;
        using (NDArray nd = np.array(data, copy: false))
            tensor = nd.ToTorch();
        if (string.IsNullOrEmpty(device) || device == "cpu")
            return tensor;
        using (tensor)
        using (PyObject dev = device.ToPython())
            return tensor.InvokeMethod("to", dev);
    }

    /// <summary>A 1-D array → a float64 <c>torch.Tensor</c>, pinned-and-viewed like the matrix overload
    /// (no copy). Under the GIL.</summary>
    /// <param name="data">The managed float64 vector or row-major matrix passed across the Python boundary.</param>
    /// <returns>A new owned Python reference; dispose it while holding the GIL. Input handles remain owned by their callers.</returns>
    /// <exception cref="global::Python.Runtime.PythonException">The Python tensor, array, model-fitting, or acquisition operation fails; the original Python diagnostic is preserved.</exception>
    internal static PyObject ToTensor(double[] data)
    {
        // Pinned-and-viewed like the matrix overload; dispose the NDArray once the tensor
        // exists (the buffer stays rooted for the tensor's life via weakref.finalize).
        using NDArray nd = np.array(data, copy: false);
        return nd.ToTorch();
    }

    /// <summary>
    ///     An <see cref="NDArray" /> (1-D vector or 2-D row-major matrix) → a <c>torch.Tensor</c>,
    ///     VIEW-shared with the array's buffer (NO copy): <see cref="TorchInterop.ToTorch" /> is
    ///     <c>torch.from_numpy(nd.ToNumpy())</c>, so numpy and torch read the buffer NumSharp already owns,
    ///     and NumSharp roots that buffer for the tensor's lifetime via a Python <c>weakref.finalize</c>.
    ///     The tensor keeps <paramref name="data" />'s dtype (float64 for a <c>double</c> array — what
    ///     botorch.py's <c>SingleTaskGP</c> fits in; a float32 source would silently break parity, so the
    ///     caller owns the dtype).
    ///     <para><b>Ownership — the key difference from the <c>double[,]</c>/<c>double[]</c> overloads:</b>
    ///     those OWN the <see cref="NDArray" /> they build from a managed array and dispose it; this one
    ///     BORROWS the caller's <paramref name="data" /> and does NOT dispose it. The caller may dispose
    ///     <paramref name="data" /> as soon as this returns — the export holds its own reference and keeps
    ///     the buffer alive for the tensor's life — but the tensor ALIASES that buffer, so
    ///     <paramref name="data" /> must not be MUTATED while the tensor is in use.</para>
    ///     <para><b>Layout:</b> the zero-copy view requires a writeable, non-negative-stride layout (true
    ///     for any freshly built or C-contiguous <see cref="NDArray" />); other layouts throw — materialize
    ///     a contiguous copy first. Under the GIL.</para>
    /// </summary>
    /// <param name="data">The caller-owned NumSharp array to expose as a tensor; viewed, not copied, and NOT disposed here.</param>
    /// <returns>A new owned Python reference (a <c>torch.Tensor</c> of <paramref name="data" />'s dtype); dispose it while holding the GIL. <paramref name="data" /> remains owned by the caller.</returns>
    /// <exception cref="global::System.ArgumentNullException"><paramref name="data" /> is null.</exception>
    /// <exception cref="global::System.InvalidOperationException">The array is non-writeable — PyTorch documents writing through a tensor over read-only numpy memory as undefined behavior.</exception>
    /// <exception cref="global::System.NotSupportedException">The array has a negative-stride layout, which torch tensors cannot view; materialize a C-contiguous copy first.</exception>
    /// <exception cref="global::Python.Runtime.PythonException">The underlying <c>torch.from_numpy</c> call fails; the original Python diagnostic is preserved.</exception>
    internal static PyObject ToTensor(NDArray data)
    {
        // data is BORROWED (caller-owned): cross it as a zero-copy view and return; do NOT dispose it.
        // The export roots the buffer for the tensor's life, so the caller may dispose data afterwards.
        if (data is null) throw new System.ArgumentNullException(nameof(data));
        return data.ToTorch();
    }

    /// <summary>
    ///     An <see cref="NDArray" /> → a <c>torch.Tensor</c> placed on <paramref name="device" /> —
    ///     botorch.py's <c>torch.from_numpy(x).to(self._device)</c>, for callers already holding an
    ///     <see cref="NDArray" />. For CPU (or a <c>null</c>/empty device) this is the zero-copy view of
    ///     the <see cref="ToTensor(NDArray)" /> overload; for any other device (e.g. <c>"cuda"</c>) the CPU
    ///     view is created, copied to the device via <c>tensor.to(device)</c> (a device tensor cannot alias
    ///     host memory — the copy is correct), and the throwaway CPU view is disposed. As with
    ///     <see cref="ToTensor(NDArray)" />, <paramref name="data" /> is BORROWED and NOT disposed here; on
    ///     the CPU path the returned tensor aliases it (do not mutate it while the tensor lives). Under the GIL.
    /// </summary>
    /// <param name="data">The caller-owned NumSharp array to expose as a tensor; viewed, not copied, and NOT disposed here.</param>
    /// <param name="device">The torch device name; null, empty, or <c>cpu</c> keeps the zero-copy CPU view, while any other device receives a copy.</param>
    /// <returns>A new owned Python reference (a <c>torch.Tensor</c> on <paramref name="device" />); dispose it while holding the GIL. <paramref name="data" /> remains owned by the caller.</returns>
    /// <exception cref="global::System.ArgumentNullException"><paramref name="data" /> is null.</exception>
    /// <exception cref="global::System.InvalidOperationException">The array is non-writeable (CPU/zero-copy path only).</exception>
    /// <exception cref="global::System.NotSupportedException">The array has a negative-stride layout (CPU/zero-copy path only).</exception>
    /// <exception cref="global::Python.Runtime.PythonException">The <c>torch.from_numpy</c> or device-transfer call fails; the original Python diagnostic is preserved.</exception>
    internal static PyObject ToTensor(NDArray data, string? device)
    {
        if (data is null) throw new System.ArgumentNullException(nameof(data));
        // Borrowed view as a CPU tensor (data is never disposed here). For CPU that view IS the result;
        // for any other device, copy it there (a device tensor cannot alias host memory) and dispose the
        // throwaway CPU view — mirrors the (double[,], device) overload once the NDArray is in hand.
        PyObject tensor = data.ToTorch();
        if (string.IsNullOrEmpty(device) || device == "cpu")
            return tensor;
        using (tensor)
        using (PyObject dev = device.ToPython())
            return tensor.InvokeMethod("to", dev);
    }

    /// <summary>
    ///     A <c>torch.Tensor</c> → an independent NumSharp <see cref="NDArray" /> via NumSharp's torch
    ///     adapter (<c>detach().cpu().resolve_conj().resolve_neg().numpy()</c> → <c>ToNDArray</c>): one bulk
    ///     copy, no per-element loop, and detached from the tensor so the tensor may be disposed
    ///     afterwards. Under the GIL.
    /// </summary>
    /// <param name="tensor">The borrowed torch tensor, accessed while the GIL is held.</param>
    /// <returns>An independent managed NumSharp array whose storage no longer depends on the Python input.</returns>
    /// <exception cref="global::Python.Runtime.PythonException">The Python tensor, array, model-fitting, or acquisition operation fails; the original Python diagnostic is preserved.</exception>
    internal static NDArray ToNDArray(PyObject tensor) => tensor.ToTorchNDArray(force: true);
}
