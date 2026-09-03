using System.Runtime.CompilerServices;
using NumSharp.Interop.OpenBLAS;

namespace OptunaSharp.Internal;

/// <summary>
///     Forces <c>NumSharp.Interop.OpenBLAS</c> to load at OptunaSharp startup so its native LAPACK/
///     CBLAS backend is registered on NumSharp's <c>TensorEngine.Blas</c> before any
///     <c>np.linalg</c> call runs. With the backend live, NumSharp's <c>cholesky</c>/<c>eigh</c>/
///     <c>solve</c>/<c>matrix_rank</c>/<c>svd</c>/<c>qr</c> compute through LAPACK and are
///     byte-identical to NumPy — the basis for OptunaSharp's byte-exact linalg-backed samplers
///     (NSGA-III's hyperplane solve, CMA-ES's eigendecomposition, the GP stack's Cholesky).
///
///     <para><b>Why a module initializer and not just the package reference.</b> The OpenBLAS
///     backend self-registers from its own <c>[ModuleInitializer]</c>, but the CLR runs a referenced
///     assembly's module initializer only when code first touches one of its types — a bare
///     project/package reference that nothing in OptunaSharp mentions leaves the assembly unloaded,
///     and <c>np.linalg.cholesky</c> then throws <c>OpenBlasMissingBackendException</c>. Touching
///     <see cref="OpenBlasEngine" /> here, from OptunaSharp.Core's own module initializer, loads the
///     assembly (running its registration) the moment OptunaSharp.Core loads. If the native binary
///     cannot be loaded on the host, the OpenBLAS initializer swallows it and
///     <see cref="OpenBlasEngine.LapackAvailable" /> is simply <c>false</c>.</para>
/// </summary>
internal static class OpenBlasActivation
{
    [ModuleInitializer]
    internal static void Activate()
    {
        // A static read of any OpenBLAS type forces the assembly load + its backend registration.
        _ = OpenBlasEngine.LapackAvailable;
    }
}
