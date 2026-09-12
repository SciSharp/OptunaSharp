using NumSharp;

namespace OptunaSharp.Samplers;

/// <summary>
///     Lazy Random State class. Port of <c>optuna/samplers/_lazy_random_state.py</c>.
///
///     This is a class to initialize the random state just before use to prevent duplication of
///     the same random state when a sampler instance is copied.
///
///     Every sampler consumes randomness through this wrapper over NumSharp's NumPy-exact
///     MT19937 <c>RandomState</c> — never <see cref="System.Random" /> — which is what makes
///     differential testing against Python bit-exact (plan §2 RNG discipline).
/// </summary>
public sealed class LazyRandomState
{
    private NumPyRandom? _rng;

    /// <summary>Creates a random-state holder, initializing immediately only when an explicit seed is supplied.</summary>
    /// <param name="seed">Seed for subsequent random draws; null uses the generator default when permitted.</param>
    public LazyRandomState(int? seed = null)
    {
        _rng = null;
        if (seed is not null)
            Rng.seed(seed.Value);
    }

    /// <summary>Allocates the NumPy-compatible random generator on first access.</summary>
    private void SetRng()
    {
        _rng = np.random.RandomState();
    }

    /// <summary>The underlying <c>np.random.RandomState</c>, created on first use.</summary>
    public NumPyRandom Rng
    {
        get
        {
            if (_rng is null)
                SetRng();
            return _rng!;
        }
    }
}
