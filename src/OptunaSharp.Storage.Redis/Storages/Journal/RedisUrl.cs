using System;
using System.Collections.Generic;
using System.Net.Sockets;
using NumSharp;                 // ValueError
using StackExchange.Redis;

namespace OptunaSharp.Storages.Journal;

/// <summary>
///     Translates a <c>redis-py</c>-style connection URL into StackExchange.Redis'
///     <see cref="ConfigurationOptions" />. This is the one thing <c>redis.Redis.from_url(url)</c>
///     gives Optuna for free that has no direct StackExchange.Redis equivalent (its native config
///     is a comma-separated <c>host:port,option=value</c> string, not a URI).
///
///     Accepts the schemes <c>redis-py</c> accepts:
///     <list type="bullet">
///       <item><c>redis://[[username]:[password]@]host[:port][/db]</c></item>
///       <item><c>rediss://…</c> — the same, over TLS</item>
///       <item><c>unix://[[username]:[password]@]/path/to/socket[?db=N]</c></item>
///     </list>
///     <c>db</c>/<c>password</c>/<c>username</c> may also be supplied as query parameters
///     (<c>?db=1&amp;password=…</c>), which take precedence over the URL components — matching
///     <c>redis-py</c>, where query args map onto the connection kwargs.
/// </summary>
internal static class RedisUrl
{
    internal static ConfigurationOptions BuildConfiguration(string url)
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
        var options = new ConfigurationOptions
        {
            // redis-py's Redis.from_url connects lazily and defers connection errors to first use;
            // mirror that so constructing the backend does not throw if the server is momentarily
            // unreachable (StackExchange.Redis will connect/reconnect on demand).
            AbortOnConnectFail = false,
        };

        var query = ParseQuery(uri.Query);

        switch (scheme)
        {
            case "redis":
            case "rediss":
                options.Ssl = scheme == "rediss";
                options.EndPoints.Add(uri.Host, uri.Port >= 0 ? uri.Port : 6379);
                if (options.Ssl)
                    options.SslHost = uri.Host;
                ApplyUserInfo(options, uri);
                ApplyDbFromPath(options, uri.AbsolutePath);
                break;

            case "unix":
                // redis-py encodes the socket path in the URL path; the host is empty.
                options.EndPoints.Add(new UnixDomainSocketEndPoint(uri.LocalPath));
                ApplyUserInfo(options, uri);
                break;

            default:
                // Verbatim redis-py error text.
                throw new ValueError(
                    "Redis URL must specify one of the following schemes " +
                    $"(redis://, rediss://, unix://): '{url}'");
        }

        // Query parameters override the URL components (redis-py precedence).
        if (query.TryGetValue("username", out string? qUser) && !string.IsNullOrEmpty(qUser))
            options.User = qUser;
        if (query.TryGetValue("password", out string? qPass) && !string.IsNullOrEmpty(qPass))
            options.Password = qPass;
        if (query.TryGetValue("db", out string? qDb) && int.TryParse(qDb, out int dbFromQuery))
            options.DefaultDatabase = dbFromQuery;

        return options;
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

    private static void ApplyUserInfo(ConfigurationOptions options, Uri uri)
    {
        if (string.IsNullOrEmpty(uri.UserInfo))
            return;
        // userinfo is "[username]:[password]" — either half may be empty (redis-py accepts
        // ":password@" with no user). Uri keeps them percent-encoded; unescape like redis-py.
        string[] parts = uri.UserInfo.Split(new[] { ':' }, 2);
        string user = Uri.UnescapeDataString(parts[0]);
        string? pass = parts.Length > 1 ? Uri.UnescapeDataString(parts[1]) : null;
        if (!string.IsNullOrEmpty(user))
            options.User = user;
        if (!string.IsNullOrEmpty(pass))
            options.Password = pass;
    }

    private static void ApplyDbFromPath(ConfigurationOptions options, string absolutePath)
    {
        string path = absolutePath.Trim('/');
        if (!string.IsNullOrEmpty(path) && int.TryParse(path, out int db))
            options.DefaultDatabase = db;
    }
}
