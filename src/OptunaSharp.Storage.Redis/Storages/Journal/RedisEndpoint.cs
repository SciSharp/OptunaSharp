using System;
using System.Collections.Generic;
using NumSharp;                 // ValueError

namespace OptunaSharp.Storages.Journal;

/// <summary>
///     A neutral, client-agnostic Redis connection target parsed from a <c>redis-py</c>-style URL.
///     Each Redis journal backend (StackExchange / NRedisStack / ServiceStack) builds its own client
///     from this — so the one bit of URL parsing that <c>redis-py</c>'s <c>from_url</c> gives Optuna
///     for free is shared, not triplicated. This is shared <em>data</em>, not an abstraction: no
///     backend implements a common interface, they only agree on the parsed endpoint.
/// </summary>
public sealed record RedisEndpoint
{
    /// <summary>Host name (empty when <see cref="UnixSocketPath" /> is set).</summary>
    public string Host { get; init; } = "";

    /// <summary>Port (defaults to 6379).</summary>
    public int Port { get; init; } = 6379;

    /// <summary>TLS (the <c>rediss://</c> scheme).</summary>
    public bool Ssl { get; init; }

    /// <summary>Database index, if the URL specified one (<c>/db</c> or <c>?db=</c>).</summary>
    public int? Db { get; init; }

    /// <summary>ACL user name, if supplied.</summary>
    public string? User { get; init; }

    /// <summary>Password, if supplied.</summary>
    public string? Password { get; init; }

    /// <summary>When set, a unix-domain-socket target (<see cref="Host" />/<see cref="Port" /> unused).</summary>
    public string? UnixSocketPath { get; init; }
}

/// <summary>
///     Parses <c>redis-py</c> connection URLs into a neutral <see cref="RedisEndpoint" />. Accepts the
///     schemes <c>redis-py</c> accepts — <c>redis://</c>, <c>rediss://</c> (TLS), <c>unix://</c> — with
///     an optional <c>[username]:[password]@</c> and <c>/db</c> (or <c>?db=</c>); query parameters
///     (<c>?db=</c>/<c>?username=</c>/<c>?password=</c>) override the URL components, matching redis-py.
/// </summary>
public static class RedisUrl
{
    public static RedisEndpoint Parse(string url)
    {
        Uri uri;
        try
        {
            uri = new Uri(url, UriKind.Absolute);
        }
        catch (UriFormatException err)
        {
            throw new ValueError($"Invalid Redis URL: '{url}' ({err.Message})");
        }

        string scheme = uri.Scheme.ToLowerInvariant();
        var query = ParseQuery(uri.Query);

        string host = "";
        int port = 6379;
        bool ssl = false;
        int? db = null;
        string? user = null;
        string? password = null;
        string? unix = null;

        switch (scheme)
        {
            case "redis":
            case "rediss":
                ssl = scheme == "rediss";
                host = uri.Host;
                port = uri.Port >= 0 ? uri.Port : 6379;
                (user, password) = ApplyUserInfo(uri, user, password);
                db = DbFromPath(uri.AbsolutePath);
                break;

            case "unix":
                unix = uri.LocalPath;
                (user, password) = ApplyUserInfo(uri, user, password);
                break;

            default:
                // Verbatim redis-py error text.
                throw new ValueError(
                    "Redis URL must specify one of the following schemes " +
                    $"(redis://, rediss://, unix://): '{url}'");
        }

        // Query parameters override the URL components (redis-py precedence).
        if (query.TryGetValue("username", out string? qUser) && !string.IsNullOrEmpty(qUser))
            user = qUser;
        if (query.TryGetValue("password", out string? qPass) && !string.IsNullOrEmpty(qPass))
            password = qPass;
        if (query.TryGetValue("db", out string? qDb) && int.TryParse(qDb, out int dbFromQuery))
            db = dbFromQuery;

        return new RedisEndpoint
        {
            Host = host,
            Port = port,
            Ssl = ssl,
            Db = db,
            User = user,
            Password = password,
            UnixSocketPath = unix,
        };
    }

    private static (string? User, string? Password) ApplyUserInfo(Uri uri, string? user, string? password)
    {
        if (string.IsNullOrEmpty(uri.UserInfo))
            return (user, password);
        // userinfo is "[username]:[password]" — either half may be empty. Uri keeps them
        // percent-encoded; unescape like redis-py.
        string[] parts = uri.UserInfo.Split(new[] { ':' }, 2);
        string u = Uri.UnescapeDataString(parts[0]);
        string? p = parts.Length > 1 ? Uri.UnescapeDataString(parts[1]) : null;
        return (string.IsNullOrEmpty(u) ? user : u, string.IsNullOrEmpty(p) ? password : p);
    }

    private static int? DbFromPath(string absolutePath)
    {
        string path = absolutePath.Trim('/');
        return !string.IsNullOrEmpty(path) && int.TryParse(path, out int db) ? db : null;
    }

    /// <summary>Parses a <c>?a=b&amp;c=d</c> query string into a case-sensitive map (last value wins).</summary>
    private static Dictionary<string, string> ParseQuery(string query)
    {
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        if (string.IsNullOrEmpty(query))
            return result;
        foreach (string pair in query.TrimStart('?').Split('&'))
        {
            if (pair.Length == 0)
                continue;
            int eq = pair.IndexOf('=');
            string key = eq >= 0 ? pair.Substring(0, eq) : pair;
            string value = eq >= 0 ? pair.Substring(eq + 1) : string.Empty;
            result[Uri.UnescapeDataString(key)] = Uri.UnescapeDataString(value);
        }
        return result;
    }
}
