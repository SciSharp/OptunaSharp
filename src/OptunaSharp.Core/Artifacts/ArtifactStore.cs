using System.IO;

namespace OptunaSharp.Artifacts;

/// <summary>
///     Exception raised when an artifact is not found.
///     Port of <c>optuna/artifacts/exceptions.py: ArtifactNotFound</c>.
///
///     It is typically raised while calling <see cref="IArtifactStore.OpenReader" /> or
///     <see cref="IArtifactStore.Remove" />.
/// </summary>
public class ArtifactNotFound : OptunaError
{
    /// <summary>Creates an artifact lookup failure retaining supplied diagnostic details.</summary>
    public ArtifactNotFound() { }

    /// <param name="message">Error message explaining the failure.</param>
    /// <summary>Creates an artifact lookup failure retaining supplied diagnostic details.</summary>
    public ArtifactNotFound(string message) : base(message) { }

    /// <param name="message">Error message explaining the failure.</param>
    /// <param name="innerException">Original failure retained as the cause of this exception.</param>
    /// <summary>Creates an artifact lookup failure retaining supplied diagnostic details.</summary>
    public ArtifactNotFound(string message, System.Exception innerException) : base(message, innerException) { }
}

/// <summary>
///     A protocol defining the interface for an artifact backend.
///     Port of <c>optuna/artifacts/_protocol.py: ArtifactStore</c> (a Python Protocol → C#
///     interface).
///
///     The methods defined in this interface are not supposed to be directly called by library
///     users. An artifact backend is responsible for managing the storage and retrieval of
///     artifact data: opening, writing and removing artifacts.
/// </summary>
public interface IArtifactStore
{
    /// <summary>
    ///     Open the artifact identified by the <paramref name="artifactId" /> as a readable
    ///     binary stream. If the artifact does not exist, an <see cref="ArtifactNotFound" />
    ///     exception should be raised.
    /// </summary>
    /// <param name="artifactId">Identifier of the artifact within the configured store.</param>
    /// <returns>A readable stream owned by the caller; dispose it after reading.</returns>
    /// <exception cref="System.Exception">The configured store cannot retrieve the artifact.</exception>
    Stream OpenReader(string artifactId);

    /// <summary>Save the content to the backend.</summary>
    /// <param name="artifactId">Identifier of the artifact within the configured store.</param>
    /// <param name="contentBody">Readable content stream; ownership stays with the caller.</param>
    /// <exception cref="System.Exception">The backend or input stream fails during the upload.</exception>
    void Write(string artifactId, Stream contentBody);

    /// <summary>
    ///     Remove the artifact identified by the <paramref name="artifactId" />. If the artifact
    ///     does not exist, an <see cref="ArtifactNotFound" /> exception may be raised.
    /// </summary>
    /// <param name="artifactId">Identifier of the artifact within the configured store.</param>
    /// <exception cref="System.Exception">The backend cannot delete the artifact.</exception>
    void Remove(string artifactId);
}
