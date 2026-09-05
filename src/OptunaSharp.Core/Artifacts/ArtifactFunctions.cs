using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;
using OptunaSharp.Storages;

namespace OptunaSharp.Artifacts;

/// <summary>
///     Meta information for an artifact.
///     Port of <c>optuna/artifacts/_upload.py: ArtifactMeta</c>.
/// </summary>
/// <param name="ArtifactId">The identifier of the artifact.</param>
/// <param name="Filename">The artifact file name used for the upload.</param>
/// <param name="Mimetype">
///     A MIME type of the artifact. If not specified, the MIME type is guessed from the file
///     extension.
/// </param>
/// <param name="Encoding">
///     An encoding of the artifact, suitable for use as a Content-Encoding header (e.g. gzip).
///     If not specified, the encoding is guessed from the file extension.
/// </param>
public sealed record ArtifactMeta(string ArtifactId, string Filename, string Mimetype, string? Encoding);

/// <summary>
///     Module-level artifact functions: <c>upload_artifact</c>, <c>download_artifact</c>,
///     <c>get_all_artifact_meta</c>.
/// </summary>
public static class ArtifactFunctions
{
    internal const string ArtifactsAttrPrefix = "artifacts:";
    internal const string DefaultMimeType = "application/octet-stream";

    /// <summary>
    ///     Upload an artifact to the artifact store (Python <c>upload_artifact</c>).
    /// </summary>
    /// <param name="artifactStore">An artifact store.</param>
    /// <param name="filePath">A path to the file to be uploaded.</param>
    /// <param name="studyOrTrial">A <see cref="Trial" />, <see cref="FrozenTrial" /> or <see cref="Study" />.</param>
    /// <param name="storage">A storage object. Required only if <paramref name="studyOrTrial" /> is a <see cref="FrozenTrial" />.</param>
    /// <param name="mimetype">A MIME type of the artifact; guessed from the file extension when omitted.</param>
    /// <param name="encoding">A Content-Encoding of the artifact; guessed from the file extension when omitted.</param>
    /// <returns>An artifact ID.</returns>
    public static string UploadArtifact(
        IArtifactStore artifactStore,
        string filePath,
        object studyOrTrial,
        BaseStorage? storage = null,
        string? mimetype = null,
        string? encoding = null)
    {
        string filename = Path.GetFileName(filePath);

        if (studyOrTrial is Trial trialWithStorage && storage is null)
            storage = trialWithStorage.Storage;
        else if (studyOrTrial is Study studyWithStorage && storage is null)
            storage = studyWithStorage.Storage;

        if (storage is null)
            throw new ValueError("storage is required for FrozenTrial.");

        string artifactId = Guid.NewGuid().ToString();
        var (guessMimetype, guessEncoding) = MimeTypes.GuessType(filename);
        var artifact = new ArtifactMeta(
            ArtifactId: artifactId,
            Filename: filename,
            Mimetype: mimetype ?? guessMimetype ?? DefaultMimeType,
            Encoding: encoding ?? guessEncoding);

        string attrKey = ArtifactsAttrPrefix + artifactId;
        string metaJson = PyJson.Dumps(new OrderedDict<string, object?>
        {
            ["artifact_id"] = artifact.ArtifactId,
            ["filename"] = artifact.Filename,
            ["mimetype"] = artifact.Mimetype,
            ["encoding"] = artifact.Encoding,
        });
        switch (studyOrTrial)
        {
            case Trial trial:
                storage.SetTrialSystemAttr(trial.TrialId, attrKey, metaJson);
                break;
            case FrozenTrial frozenTrial:
                storage.SetTrialSystemAttr(frozenTrial.TrialId, attrKey, metaJson);
                break;
            case Study study:
                storage.SetStudySystemAttr(study.StudyId, attrKey, metaJson);
                break;
            default:
                throw new TypeError(
                    $"study_or_trial must be a Trial, FrozenTrial or Study, but got {studyOrTrial.GetType().Name}.");
        }

        using (var f = new FileStream(filePath, FileMode.Open, FileAccess.Read))
        {
            artifactStore.Write(artifactId, f);
        }
        return artifactId;
    }

    /// <summary>
    ///     Download an artifact from the artifact store (Python <c>download_artifact</c>).
    /// </summary>
    /// <param name="artifactStore">An artifact store.</param>
    /// <param name="filePath">A path to save the downloaded artifact.</param>
    /// <param name="artifactId">The identifier of the artifact to download.</param>
    public static void DownloadArtifact(IArtifactStore artifactStore, string filePath, string artifactId)
    {
        if (File.Exists(filePath))
            throw new IOException($"File already exists: {filePath}");

        using var reader = artifactStore.OpenReader(artifactId);
        using var writer = new FileStream(filePath, FileMode.CreateNew, FileAccess.Write);
        reader.CopyTo(writer);
    }

    /// <summary>
    ///     List the associated artifact information of the provided trial or study
    ///     (Python <c>get_all_artifact_meta</c>). If a <see cref="Study" /> is provided, the
    ///     artifacts uploaded to the study are returned, not those of all its trials.
    /// </summary>
    public static List<ArtifactMeta> GetAllArtifactMeta(object studyOrTrial, BaseStorage? storage = null)
    {
        if (studyOrTrial is Trial trialWithStorage && storage is null)
            storage = trialWithStorage.Storage;
        else if (studyOrTrial is Study studyWithStorage && storage is null)
            storage = studyWithStorage.Storage;

        if (storage is null)
            throw new ValueError("storage is required for FrozenTrial.");

        IReadOnlyDictionary<string, object?> systemAttrs = studyOrTrial switch
        {
            Trial trial => storage.GetTrialSystemAttrs(trial.TrialId),
            FrozenTrial frozenTrial => storage.GetTrialSystemAttrs(frozenTrial.TrialId),
            Study study => storage.GetStudySystemAttrs(study.StudyId),
            _ => throw new TypeError(
                $"study_or_trial must be a Trial, FrozenTrial or Study, but got {studyOrTrial.GetType().Name}."),
        };

        var artifactMetaList = new List<ArtifactMeta>();
        foreach (var (attrKey, attrJsonString) in systemAttrs)
        {
            if (!attrKey.StartsWith(ArtifactsAttrPrefix, StringComparison.Ordinal))
                continue;

            var attrContent = (OrderedDict<string, object?>)PyJson.Loads((string)attrJsonString!)!;
            artifactMetaList.Add(new ArtifactMeta(
                ArtifactId: (string)attrContent["artifact_id"]!,
                Filename: (string)attrContent["filename"]!,
                Mimetype: (string)attrContent["mimetype"]!,
                Encoding: (string?)attrContent["encoding"]));
        }

        return artifactMetaList;
    }
}

/// <summary>
///     A minimal port of Python's <c>mimetypes.guess_type</c> covering the common table —
///     (mimetype, encoding) guessed from a filename.
/// </summary>
internal static class MimeTypes
{
    private static readonly Dictionary<string, string> _encodingsMap = new(StringComparer.OrdinalIgnoreCase)
    {
        [".gz"] = "gzip",
        [".Z"] = "compress",
        [".bz2"] = "bzip2",
        [".xz"] = "xz",
        [".br"] = "br",
    };

    private static readonly Dictionary<string, string> _typesMap = new(StringComparer.OrdinalIgnoreCase)
    {
        [".txt"] = "text/plain",
        [".csv"] = "text/csv",
        [".html"] = "text/html",
        [".htm"] = "text/html",
        [".css"] = "text/css",
        [".js"] = "text/javascript",
        [".json"] = "application/json",
        [".xml"] = "text/xml",
        [".pdf"] = "application/pdf",
        [".zip"] = "application/zip",
        [".tar"] = "application/x-tar",
        [".png"] = "image/png",
        [".jpg"] = "image/jpeg",
        [".jpeg"] = "image/jpeg",
        [".gif"] = "image/gif",
        [".svg"] = "image/svg+xml",
        [".bmp"] = "image/bmp",
        [".tiff"] = "image/tiff",
        [".webp"] = "image/webp",
        [".mp3"] = "audio/mpeg",
        [".wav"] = "audio/x-wav",
        [".mp4"] = "video/mp4",
        [".avi"] = "video/x-msvideo",
        [".bin"] = "application/octet-stream",
        [".pkl"] = "application/octet-stream",
        [".npy"] = "application/octet-stream",
        [".npz"] = "application/octet-stream",
        [".md"] = "text/markdown",
        [".py"] = "text/x-python",
        [".yaml"] = "application/yaml",
        [".yml"] = "application/yaml",
    };

    internal static (string? Mimetype, string? Encoding) GuessType(string filename)
    {
        string? encoding = null;
        string ext = Path.GetExtension(filename);
        // Python strips encoding suffixes first: "model.tar.gz" → encoding gzip, type of "model.tar".
        while (_encodingsMap.TryGetValue(ext, out var enc))
        {
            encoding = enc;
            filename = filename.Substring(0, filename.Length - ext.Length);
            ext = Path.GetExtension(filename);
        }
        string? mimetype = _typesMap.TryGetValue(ext, out var type) ? type : null;
        return (mimetype, encoding);
    }
}
