using System;

namespace OptunaSharp.Internal;

internal static class MathUtil
{
    /// <summary>C99 / NumPy <c>nextafter(x, y)</c>: the next representable double after <paramref name="x" /> toward <paramref name="y" />.</summary>
    internal static double NextAfter(double x, double y)
    {
        if (double.IsNaN(x) || double.IsNaN(y))
            return x + y;
        if (x == y)
            return y;
        return y > x ? Math.BitIncrement(x) : Math.BitDecrement(x);
    }

    /// <summary>NumPy <c>np.clip(value, low, high)</c> for scalars (NaN propagates).</summary>
    internal static double Clip(double value, double low, double high)
        => Math.Min(Math.Max(value, low), high);
}
