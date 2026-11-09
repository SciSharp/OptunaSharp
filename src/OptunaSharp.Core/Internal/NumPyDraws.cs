using System;
using NumSharp;

namespace OptunaSharp.Internal;

/// <summary>
///     NumPy random draws read into managed values without leaving a NumSharp array to the finalizer (plan 14). Use these
///     on per-trial and per-parameter paths instead of casting a draw (<c>(double)rng.rand()</c>): the draws, their order,
///     their bits and the generator's next state are exactly the cast's.
/// </summary>
/// <remarks>
///     <para><b>Why it matters.</b> Every NumSharp <see cref="NDArray" /> carries a finalizer. One that is never disposed
///     survives its first garbage collection on the finalization queue and is promoted, so a hot path that drops arrays
///     turns almost every gen-0 collection into a gen-1 collection. Measured before plan 14 on 40-parameter studies: GC
///     pauses were 47 % of random search's time and 41 % of NSGA-II's, from exactly such per-parameter draws.</para>
///     <para><b>Two ways, one stream.</b> The uniform draws (<see cref="Rand(NumPyRandom)" />, <see cref="Rand(NumPyRandom, int)" />,
///     <see cref="RandomSample" />, <see cref="Uniform" />) read the generator's bit generator directly, as NumSharp's own
///     scalar and fill paths do: one <see cref="BitGenerator.NextDouble" /> per value (NumPy's <c>next_double</c>, for
///     MT19937 the 27 + 26 bits of two consecutive words), under the bit generator's <see cref="BitGenerator.@lock" />, so
///     no array is created at all. The draws whose algorithm carries more state (<see cref="RandInt" />'s masked rejection,
///     <see cref="Normal(NumPyRandom, double, double)" />'s cached polar Gaussian) call NumSharp and dispose the array it
///     returns. <c>NumPyDrawsTests</c> pins each helper against the NumSharp call it replaces, value and next
///     state, on generators seeded alike.</para>
///     <para><b>Concurrency.</b> Each call takes the bit generator's lock for its own draws, as a NumSharp call does; a
///     helper never caches the bit generator, so a reseed (<c>rng.seed(...)</c>) is seen by the next call.</para>
/// </remarks>
internal static class NumPyDraws
{
    /// <summary>NumPy <c>rng.rand()</c> (<c>random_sample()</c>): one uniform draw in [0, 1) as a <see cref="double" />.</summary>
    /// <param name="rng">The generator to draw from; its state advances exactly as <c>rng.rand()</c> advances it.</param>
    /// <returns>The drawn value.</returns>
    internal static double Rand(NumPyRandom rng)
    {
        BitGenerator bitGenerator = rng.get_bit_generator();
        lock (bitGenerator.@lock)
            return bitGenerator.NextDouble();
    }

    /// <summary>NumPy <c>rng.rand(n)</c>: <paramref name="n" /> uniform draws in [0, 1), in draw order.</summary>
    /// <param name="rng">The generator to draw from; its state advances exactly as <c>rng.rand(n)</c> advances it.</param>
    /// <param name="n">The number of draws; zero draws nothing and returns an empty array.</param>
    /// <returns>A new array holding the draws in the order NumPy fills them.</returns>
    /// <exception cref="ValueError"><paramref name="n" /> is negative (NumPy's negative-dimension error); nothing is drawn.</exception>
    internal static double[] Rand(NumPyRandom rng, int n) => RandomSample(rng, n);

    /// <summary>NumPy <c>rng.random_sample(n)</c>: <paramref name="n" /> uniform draws in [0, 1), in draw order.</summary>
    /// <param name="rng">The generator to draw from; its state advances exactly as <c>rng.random_sample(n)</c> advances it.</param>
    /// <param name="n">The number of draws; zero draws nothing and returns an empty array.</param>
    /// <returns>A new array holding the draws in the order NumPy fills them.</returns>
    /// <exception cref="ValueError"><paramref name="n" /> is negative (NumPy's negative-dimension error); nothing is drawn.</exception>
    internal static double[] RandomSample(NumPyRandom rng, long n)
    {
        if (n < 0)
            throw new ValueError("negative dimensions are not allowed");
        var draws = new double[n];
        BitGenerator bitGenerator = rng.get_bit_generator();
        // One lock for the whole fill, as NumSharp's random_sample(n) holds it: the n values are consecutive in the stream
        // even when another thread shares the generator.
        lock (bitGenerator.@lock)
        {
            for (long i = 0; i < n; i++)
                draws[i] = bitGenerator.NextDouble();
        }
        return draws;
    }

    /// <summary>NumPy <c>rng.randint(low, high)</c>: one integer draw in [<paramref name="low" />, <paramref name="high" />).</summary>
    /// <param name="rng">The generator to draw from; its state advances exactly as <c>rng.randint(low, high)</c> advances it.</param>
    /// <param name="low">The smallest value that can be drawn.</param>
    /// <param name="high">One past the largest value that can be drawn.</param>
    /// <returns>The drawn value, as the 64-bit integer NumPy's default <c>randint</c> dtype holds.</returns>
    /// <remarks>Calls NumSharp's <c>randint</c>, whose masked rejection sampling consumes a data-dependent number of words, and disposes the 0-d array it returns.</remarks>
    /// <exception cref="ValueError"><paramref name="high" /> is not greater than <paramref name="low" /> (NumPy's <c>low &gt;= high</c> error).</exception>
    internal static long RandInt(NumPyRandom rng, long low, long high)
    {
        using NDArray draw = rng.randint(low, high);
        return (long)draw;
    }

    /// <summary>NumPy <c>rng.uniform(low, high)</c> with scalar bounds: <c>low + (high - low) * U</c> for one uniform <c>U</c>.</summary>
    /// <param name="rng">The generator to draw from; its state advances exactly as <c>rng.uniform(low, high)</c> advances it.</param>
    /// <param name="low">The lower bound of the draw.</param>
    /// <param name="high">The upper bound of the draw.</param>
    /// <returns>The drawn value.</returns>
    /// <remarks>
    ///     NumPy's <c>random_uniform</c>, <c>lower + range * next_double</c>, with the range computed first and checked for
    ///     finiteness before anything is drawn, as both NumPy and NumSharp do. With one-element bound arrays NumPy computes the
    ///     same expression from the same uniform, so drawing a bounds column one row at a time with this helper reproduces
    ///     <c>rng.uniform(bounds[:, 0], bounds[:, 1])</c> bit for bit, as long as no range is infinite (the array form raises
    ///     before its first draw, a row-by-row loop only when it reaches the bad row).
    /// </remarks>
    /// <exception cref="OverflowException"><c>high - low</c> is infinite or NaN (NumPy's <c>OverflowError('Range exceeds valid bounds')</c>); nothing is drawn.</exception>
    internal static double Uniform(NumPyRandom rng, double low, double high)
    {
        double range = high - low;
        if (!double.IsFinite(range))
            throw new OverflowException("Range exceeds valid bounds");
        return low + range * Rand(rng);
    }

    /// <summary>NumPy <c>rng.normal(loc, scale)</c>: one Gaussian draw as a <see cref="double" />.</summary>
    /// <param name="rng">The generator to draw from; its state, including the cached second Gaussian of the polar method, advances exactly as <c>rng.normal(loc, scale)</c> advances it.</param>
    /// <param name="loc">The mean of the distribution.</param>
    /// <param name="scale">The standard deviation of the distribution.</param>
    /// <returns>The drawn value.</returns>
    /// <exception cref="ValueError"><paramref name="scale" /> is negative (NumPy's <c>scale &lt; 0</c> error).</exception>
    internal static double Normal(NumPyRandom rng, double loc, double scale)
    {
        using NDArray draw = rng.normal(loc, scale);
        return (double)draw;
    }

    /// <summary>NumPy <c>rng.normal(loc, scale, size=n)</c>: <paramref name="n" /> Gaussian draws, in draw order.</summary>
    /// <param name="rng">The generator to draw from; its state advances exactly as <c>rng.normal(loc, scale, n)</c> advances it.</param>
    /// <param name="loc">The mean of the distribution.</param>
    /// <param name="scale">The standard deviation of the distribution.</param>
    /// <param name="n">The number of draws.</param>
    /// <returns>A new array holding the draws in the order NumPy fills them.</returns>
    /// <exception cref="ValueError"><paramref name="scale" /> is negative, or <paramref name="n" /> is negative.</exception>
    internal static double[] Normal(NumPyRandom rng, double loc, double scale, int n)
    {
        using NDArray draws = rng.normal(loc, scale, new Shape(n));
        return draws.ToArray<double>();
    }
}
