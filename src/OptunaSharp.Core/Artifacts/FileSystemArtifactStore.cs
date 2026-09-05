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
    public FileSystemArtifactStore(string basePath)
    {
        _basePath = basePath;
    }

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

    public void Write(string artifactId, Stream contentBody)
    {
        string filepath = GetFilepath(artifactId);
        using var f = new FileStream(filepath, FileMode.Create, FileAccess.Write);
        contentBody.CopyTo(f);
    }

    public void Remove(string artifactId)
    {
        string filepath = GetFilepath(artifactId);
        if (!File.Exists(filepath))
            throw new ArtifactNotFound("not found");
        File.Delete(filepath);
    }
}
