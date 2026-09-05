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
    public ArtifactNotFound() { }

    public ArtifactNotFound(string message) : base(message) { }

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
    Stream OpenReader(string artifactId);

    /// <summary>Save the content to the backend.</summary>
    void Write(string artifactId, Stream contentBody);

    /// <summary>
    ///     Remove the artifact identified by the <paramref name="artifactId" />. If the artifact
    ///     does not exist, an <see cref="ArtifactNotFound" /> exception may be raised.
    /// </summary>
    void Remove(string artifactId);
}
