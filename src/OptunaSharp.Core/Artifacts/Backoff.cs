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

    private double GetSleepSecs(int nRetry) => Math.Min(_minDelay * Math.Pow(_multiplier, nRetry), _maxDelay);

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
                contentBody.Seek(0, SeekOrigin.Begin);
                Thread.Sleep(TimeSpan.FromSeconds(GetSleepSecs(i)));
            }
        }
    }

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
