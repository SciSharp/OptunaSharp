using System;
using System.IO;
using System.Threading;
using OptunaSharp.Logging;

namespace OptunaSharp.Artifacts;

/// <summary>
///     An artifact store's middleware for exponential backoff.
///     Port of <c>optuna/artifacts/_backoff.py: Backoff</c>.
/// </summary>
/// <example>
///     <code>
///     var artifactStore = new Backoff(new FileSystemArtifactStore("./artifacts"));
///     </code>
/// </example>
public class Backoff : IArtifactStore
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.artifacts._backoff");

    private readonly IArtifactStore _backend;
    private readonly int _maxRetries;
    private readonly double _multiplier;
    private readonly double _minDelay;
    private readonly double _maxDelay;

    // Default sleep seconds:
    // 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 30
    /// <summary>Wraps an artifact store with bounded exponential retry delays.</summary>
    /// <param name="backend">Shared store to call; ownership remains with the caller.</param>
    /// <param name="maxRetries">Maximum total attempts, including the first; must be positive.</param>
    /// <param name="multiplier">Positive factor applied to the delay after each failed attempt.</param>
    /// <param name="minDelay">Positive first retry delay in seconds.</param>
    /// <param name="maxDelay">Delay cap in seconds; must exceed minDelay.</param>
    /// <remarks>Constructor preconditions are debug assertions. ArtifactNotFound is never retried, and the final provider exception is rethrown.</remarks>
    public Backoff(
        IArtifactStore backend,
        int maxRetries = 10,
        double multiplier = 2,
        double minDelay = 0.1,
        double maxDelay = 30)
    {
        _backend = backend;
        System.Diagnostics.Debug.Assert(maxRetries > 0);
        System.Diagnostics.Debug.Assert(multiplier > 0);
        System.Diagnostics.Debug.Assert(minDelay > 0);
        System.Diagnostics.Debug.Assert(maxDelay > minDelay);
        _maxRetries = maxRetries;
        _multiplier = multiplier;
        _minDelay = minDelay;
        _maxDelay = maxDelay;
    }

    /// <summary>Calculates the capped delay after a failed attempt.</summary>
    /// <param name="nRetry">Zero-based failed-attempt index.</param>
    /// <returns>Delay in seconds after applying the multiplier and cap.</returns>
    private double GetSleepSecs(int nRetry) => Math.Min(_minDelay * Math.Pow(_multiplier, nRetry), _maxDelay);

    /// <param name="artifactId">Identifier of the artifact within the configured store.</param>
    /// <summary>Opens binary artifact content for reading.</summary>
    /// <returns>A readable stream owned by the caller; dispose it after reading.</returns>
    /// <exception cref="OptunaSharp.Artifacts.ArtifactNotFound">The artifact does not exist.</exception>
    /// <exception cref="System.Exception">The configured store cannot retrieve the artifact.</exception>
    /// <exception cref="System.InvalidOperationException">The attempt limit is not positive, violating the constructor precondition.</exception>
    public Stream OpenReader(string artifactId)
    {
        for (int i = 0; i < _maxRetries; i++)
        {
            try
            {
                return _backend.OpenReader(artifactId);
            }
            catch (ArtifactNotFound)
            {
                throw;
            }
            catch (Exception e)
            {
                if (i == _maxRetries - 1)
                    throw;
                _logger.Warning($"Failed to open artifact={artifactId} n_retry={i}", e);
            }
            Thread.Sleep(TimeSpan.FromSeconds(GetSleepSecs(i)));
        }
        throw new InvalidOperationException("must not reach here");
    }

    /// <param name="artifactId">Identifier of the artifact within the configured store.</param>
    /// <param name="contentBody">Readable content stream; ownership stays with the caller.</param>
    /// <summary>Copies artifact content into the store without taking ownership of the input stream.</summary>
    /// <exception cref="System.Exception">The backend or input stream fails during the upload.</exception>
    /// <exception cref="System.NotSupportedException">A retry attempts to rewind a non-seekable content stream.</exception>
    /// <remarks>Retries rewind contentBody to position zero, so the stream must be seekable when a retry occurs. A failed attempt can leave partial backend content.</remarks>
    public void Write(string artifactId, Stream contentBody)
    {
        for (int i = 0; i < _maxRetries; i++)
        {
            try
            {
                _backend.Write(artifactId, contentBody);
                break;
            }
            catch (ArtifactNotFound)
            {
                throw;
            }
            catch (Exception e)
            {
                if (i == _maxRetries - 1)
                    throw;
                _logger.Warning($"Failed to open artifact={artifactId} n_retry={i}", e);
                // A failed attempt may have consumed part of the stream. Restart from byte zero
                // so retrying does not silently upload only the unconsumed suffix.
                contentBody.Seek(0, SeekOrigin.Begin);
                Thread.Sleep(TimeSpan.FromSeconds(GetSleepSecs(i)));
            }
        }
    }

    /// <param name="artifactId">Identifier of the artifact within the configured store.</param>
    /// <summary>Deletes the artifact identified by its store key.</summary>
    /// <exception cref="OptunaSharp.Artifacts.ArtifactNotFound">The backend reports that the artifact is absent.</exception>
    /// <exception cref="System.Exception">The backend cannot delete the artifact.</exception>
    public void Remove(string artifactId)
    {
        // Python's loop retries even after a successful remove (missing `break`); the port
        // breaks after success — the retry-on-success is a Python bug, not behavior to keep.
        for (int i = 0; i < _maxRetries; i++)
        {
            try
            {
                _backend.Remove(artifactId);
                break;
            }
            catch (ArtifactNotFound)
            {
                throw;
            }
            catch (Exception e)
            {
                if (i == _maxRetries - 1)
                    throw;
                _logger.Warning($"Failed to delete artifact={artifactId}", e);
                Thread.Sleep(TimeSpan.FromSeconds(GetSleepSecs(i)));
            }
        }
    }
}
