using System;
using System.IO;
using NumSharp;

namespace OptunaSharp.Artifacts;

/// <summary>
///     An artifact store for file systems.
///     Port of <c>optuna/artifacts/_filesystem.py: FileSystemArtifactStore</c>.
/// </summary>
/// <example>
///     <code>
///     var basePath = "./artifacts";
///     Directory.CreateDirectory(basePath);
///     var artifactStore = new FileSystemArtifactStore(basePath: basePath);
///     // inside an objective:
///     ArtifactFunctions.UploadArtifact(artifactStore, filePath, trial);
///     </code>
/// </example>
public class FileSystemArtifactStore : IArtifactStore
{
    private readonly string _basePath;

    /// <param name="basePath">The base path to a directory to store artifacts.</param>
    /// <summary>Uses an existing directory as the root for artifact files.</summary>
    /// <remarks>The constructor stores the path without creating or validating the directory.</remarks>
    public FileSystemArtifactStore(string basePath)
    {
        _basePath = basePath;
    }

    /// <param name="artifactId">Identifier of the artifact within the configured store.</param>
    /// <summary>Resolves an artifact key and rejects paths escaping the configured directory.</summary>
    /// <returns>The combined artifact path.</returns>
    /// <exception cref="NumSharp.ValueError">The identifier is rooted or resolves outside the base directory.</exception>
    /// <exception cref="System.ArgumentException">The path has an unsupported format.</exception>
    private string GetFilepath(string artifactId)
    {
        if (Path.IsPathRooted(artifactId))
            throw new ValueError($"Invalid artifact_id: {artifactId}");

        string filepath = Path.Combine(_basePath, artifactId);
        // Python: os.path.commonpath check — reject ids escaping the base directory.
        string fullBase = Path.GetFullPath(_basePath).TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        string fullPath = Path.GetFullPath(filepath);
        if (!fullPath.StartsWith(fullBase + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)
            && !string.Equals(fullPath, fullBase, StringComparison.OrdinalIgnoreCase))
        {
            throw new ValueError($"Invalid artifact_id: {artifactId}");
        }
        return filepath;
    }

    /// <param name="artifactId">Identifier of the artifact within the configured store.</param>
    /// <summary>Opens binary artifact content for reading.</summary>
    /// <returns>A readable stream owned by the caller; dispose it after reading.</returns>
    /// <exception cref="OptunaSharp.Artifacts.ArtifactNotFound">The artifact does not exist.</exception>
    /// <exception cref="System.Exception">The configured store cannot retrieve the artifact.</exception>
    /// <exception cref="NumSharp.ValueError">The artifact identifier is rooted or escapes the configured directory.</exception>
    /// <exception cref="System.IO.IOException">A filesystem operation fails.</exception>
    /// <exception cref="System.UnauthorizedAccessException">Filesystem access is denied.</exception>
    public Stream OpenReader(string artifactId)
    {
        string filepath = GetFilepath(artifactId);
        try
        {
            return new FileStream(filepath, FileMode.Open, FileAccess.Read);
        }
        catch (FileNotFoundException e)
        {
            throw new ArtifactNotFound("not found", e);
        }
        catch (DirectoryNotFoundException e)
        {
            throw new ArtifactNotFound("not found", e);
        }
    }

    /// <param name="artifactId">Identifier of the artifact within the configured store.</param>
    /// <param name="contentBody">Readable content stream; ownership stays with the caller.</param>
    /// <summary>Copies artifact content into the store without taking ownership of the input stream.</summary>
    /// <exception cref="System.Exception">The backend or input stream fails during the upload.</exception>
    /// <exception cref="NumSharp.ValueError">The artifact identifier is rooted or escapes the configured directory.</exception>
    /// <exception cref="System.IO.IOException">A filesystem operation fails.</exception>
    /// <exception cref="System.UnauthorizedAccessException">Filesystem access is denied.</exception>
    public void Write(string artifactId, Stream contentBody)
    {
        string filepath = GetFilepath(artifactId);
        using var f = new FileStream(filepath, FileMode.Create, FileAccess.Write);
        contentBody.CopyTo(f);
    }

    /// <param name="artifactId">Identifier of the artifact within the configured store.</param>
    /// <summary>Deletes the artifact identified by its store key.</summary>
    /// <exception cref="OptunaSharp.Artifacts.ArtifactNotFound">The backend reports that the artifact is absent.</exception>
    /// <exception cref="System.Exception">The backend cannot delete the artifact.</exception>
    /// <exception cref="NumSharp.ValueError">The artifact identifier is rooted or escapes the configured directory.</exception>
    /// <exception cref="System.IO.IOException">A filesystem operation fails.</exception>
    /// <exception cref="System.UnauthorizedAccessException">Filesystem access is denied.</exception>
    public void Remove(string artifactId)
    {
        string filepath = GetFilepath(artifactId);
        if (!File.Exists(filepath))
            throw new ArtifactNotFound("not found");
        File.Delete(filepath);
    }
}
