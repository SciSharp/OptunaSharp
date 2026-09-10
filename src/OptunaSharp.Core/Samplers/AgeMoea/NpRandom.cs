using NumSharp;

namespace OptunaSharp.Samplers.AgeMoea;

/// <summary>
///     Thin adapters over NumSharp's legacy MT19937 <see cref="NumPyRandom" /> that return plain
///     managed arrays in NumPy's exact draw order and shape. Verified 0-ULP against numpy 2.0.2 for
///     <c>permutation</c>, <c>choice</c> (== <c>randint(0, n)</c>-index), <c>random((r, c))</c>
///     C-order, and scalar <c>randint</c> — the operations the AGE-MOEA operators consume.
///
///     Keeping the draw shape identical to pymoo's (<c>np.random.random((n_matings, n_var))</c>,
///     row-major) is load-bearing: a transposed or per-element loop would consume the MT19937 stream
///     in a different order and silently break byte-exactness.
/// </summary>
internal static class NpRandom
{
    /// <summary><c>np.random.random(n)</c> — a 1-D uniform draw.</summary>
    public static double[] Random(NumPyRandom rng, int n) => rng.rand(n).ToArray<double>();

    /// <summary>
    ///     <c>np.random.random((rows, cols))</c> as a jagged array, filled C-order (row-major) — the
    ///     same element order NumPy assigns, so the MT19937 stream is consumed identically.
    /// </summary>
    public static double[][] Random(NumPyRandom rng, int rows, int cols)
    {
        double[] flat = rng.rand(rows, cols).ToArray<double>();
        var result = new double[rows][];
        for (int i = 0; i < rows; i++)
        {
            var row = new double[cols];
            for (int j = 0; j < cols; j++)
                row[j] = flat[i * cols + j];
            result[i] = row;
        }
        return result;
    }

    /// <summary><c>np.random.permutation(n)</c> — a random permutation of <c>0..n-1</c>.</summary>
    public static int[] Permutation(NumPyRandom rng, int n)
    {
        long[] p = rng.permutation(n).ToArray<long>();
        var result = new int[p.Length];
        for (int i = 0; i < p.Length; i++)
            result[i] = (int)p[i];
        return result;
    }

    /// <summary><c>np.random.randint(low, high)</c> — a single integer in <c>[low, high)</c>.</summary>
    public static int RandInt(NumPyRandom rng, int low, int high) =>
        System.Convert.ToInt32(rng.randint(low, high).GetValue(0));
}
