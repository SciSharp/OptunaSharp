using Microsoft.Extensions.Diagnostics.HealthChecks;
using OptunaSharp.Storages;

namespace TuningService.Infrastructure;

/// <summary>
///     Reports whether the Optuna storage answers, for <c>GET /health</c>: it lists the studies, which needs a
///     working connection to the database.
/// </summary>
/// <remarks>
///     The storage is the container's singleton <see cref="BaseStorage" />, the same instance every study, job
///     and gRPC call uses, so the check exercises the real connection rather than opening a new one.
/// </remarks>
public sealed class StorageHealthCheck : IHealthCheck
{
    private readonly BaseStorage _storage;

    /// <summary>Creates the check.</summary>
    /// <param name="storage">The application's storage.</param>
    /// <exception cref="ArgumentNullException"><paramref name="storage" /> is <c>null</c>.</exception>
    public StorageHealthCheck(BaseStorage storage)
    {
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
    }

    /// <summary>Lists the studies and reports healthy with their count, or unhealthy with the error.</summary>
    /// <param name="context">The health-check registration being run.</param>
    /// <param name="cancellationToken">Unused: the storage API is synchronous, as Optuna's is.</param>
    /// <returns>The check result.</returns>
    public Task<HealthCheckResult> CheckHealthAsync(
        HealthCheckContext context, CancellationToken cancellationToken = default)
    {
        try
        {
            int studies = _storage.GetAllStudies().Count;
            return Task.FromResult(HealthCheckResult.Healthy($"{studies} studies"));
        }
        catch (Exception e)
        {
            // the exception goes to the health-check log; /health shows only the status
            return Task.FromResult(HealthCheckResult.Unhealthy("The Optuna storage did not answer.", e));
        }
    }
}
