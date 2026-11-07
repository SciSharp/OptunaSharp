using OptunaSharp.Samplers;

namespace TuningService.Infrastructure;

/// <summary>
///     Builds the sampler that <see cref="TuningServiceOptions.Sampler" /> names, so the search algorithm is a
///     configuration value rather than code.
/// </summary>
/// <remarks>
///     Samplers are value-configured objects (seeds, startup trials, population sizes) that keep per-study state,
///     so the container does not construct them by reflection. Program.cs registers this method as a transient
///     factory, and every study the container serves gets a sampler of its own.
/// </remarks>
public static class SamplerSelection
{
    /// <summary>Creates a new sampler of the configured kind.</summary>
    /// <param name="options">The validated service options.</param>
    /// <returns>A sampler that no other study uses.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="options" /> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    ///     The sampler name is not one of the allowed values. Startup validation refuses such a name first, so
    ///     this only fires when the options were built without validation.
    /// </exception>
    public static BaseSampler Create(TuningServiceOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        int? seed = options.SamplerSeed;
        return options.Sampler switch
        {
            "TPE" => new TPESampler(seed: seed),           // Optuna's default single-objective sampler
            "Random" => new RandomSampler(seed: seed),
            "QMC" => new QMCSampler(seed: seed),           // Sobol' sequence
            "GP" => new GPSampler(seed: seed),             // Gaussian-process Bayesian optimization
            "CmaEs" => new CmaEsSampler(seed: seed),
            "NSGAII" => new NSGAIISampler(seed: seed),     // Optuna's default multi-objective sampler
            _ => throw new ArgumentOutOfRangeException(nameof(options), options.Sampler, "Unknown sampler name."),
        };
    }
}
