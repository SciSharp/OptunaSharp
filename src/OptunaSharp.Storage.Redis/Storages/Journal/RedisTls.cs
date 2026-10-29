using System;
using System.IO;
using System.Linq;
using System.Net.Security;
using System.Security.Authentication;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using System.Text.RegularExpressions;
using NumSharp;                 // TypeError

namespace OptunaSharp.Storages.Journal;

/// <summary>
///     The TLS settings of a <c>rediss://</c> endpoint, applied as redis-py's <c>SSLConnection</c> applies them through
///     Python's <c>ssl</c> module: the system's certificate authorities plus any the URL names, the host name check,
///     a client certificate, a minimum TLS version, and no revocation check.
/// </summary>
/// <remarks>
///     Python loads these files on every connect, so a missing or malformed file fails the first command there; the
///     backends load them once, while they are constructed, and fail there instead.
/// </remarks>
internal static class RedisTls
{
    /// <summary>The server-authentication extended key usage, which Python's default context requires of a server certificate.</summary>
    private const string ServerAuthOid = "1.3.6.1.5.5.7.3.1";

    /// <summary>The file names OpenSSL's hashed-directory lookup reads: <c>&lt;8 hex digits&gt;.&lt;n&gt;</c>, as <c>c_rehash</c> writes them.</summary>
    private static readonly Regex HashedCertificateName = new("^[0-9a-f]{8}\\.[0-9]+$", RegexOptions.CultureInvariant);

    /// <summary>
    ///     The TLS versions <c>ssl_min_version</c> allows, or <see langword="null" /> for the platform's default (no
    ///     <c>ssl_min_version</c>, or -2, the minimum the platform supports).
    /// </summary>
    /// <param name="ep">The endpoint.</param>
    /// <returns>The allowed protocols, or <see langword="null" />.</returns>
    /// <remarks>SSL 3.0 (768) is not available in .NET, so it allows the same as TLS 1.0, the lowest that is.</remarks>
    internal static SslProtocols? Protocols(RedisEndpoint ep)
    {
#pragma warning disable SYSLIB0039 // TLS 1.0 and 1.1 are exactly what an ssl_min_version of 768-770 allows.
        return ep.SslMinVersion switch
        {
            null or -2 => null,
            768 or 769 => SslProtocols.Tls | SslProtocols.Tls11 | SslProtocols.Tls12 | SslProtocols.Tls13,
            770 => SslProtocols.Tls11 | SslProtocols.Tls12 | SslProtocols.Tls13,
            771 => SslProtocols.Tls12 | SslProtocols.Tls13,
            // 772 (TLS 1.3) and -1 (the maximum supported).
            _ => SslProtocols.Tls13,
        };
#pragma warning restore SYSLIB0039
    }

    /// <summary>
    ///     Whether .NET's own certificate check (the system's authorities and the host name) is exactly Python's for this
    ///     endpoint, so no callback is needed.
    /// </summary>
    /// <param name="ep">The endpoint.</param>
    /// <returns><see langword="true" /> when certificates are checked against the system's authorities, host name included.</returns>
    internal static bool UsesDefaultValidation(RedisEndpoint ep)
        => ep.SslCertReqs != "none" && ep.SslCheckHostname && ep.SslCaCerts is null && ep.SslCaData is null && ep.SslCaPath is null;

    /// <summary>
    ///     The certificate check for the endpoint, or <see langword="null" /> when .NET's own check is already Python's.
    /// </summary>
    /// <param name="ep">The endpoint.</param>
    /// <returns>The callback, or <see langword="null" />.</returns>
    /// <exception cref="FileNotFoundException"><c>ssl_ca_certs</c> names a missing file.</exception>
    /// <exception cref="CryptographicException"><c>ssl_ca_certs</c> or <c>ssl_ca_data</c> holds no certificate, or a malformed one.</exception>
    internal static RemoteCertificateValidationCallback? Validation(RedisEndpoint ep)
    {
        if (UsesDefaultValidation(ep))
            return null;
        if (ep.SslCertReqs == "none")
            return static (_, _, _, _) => true;
        X509Certificate2Collection authorities = LoadAuthorities(ep);
        bool checkHostname = ep.SslCheckHostname;
        return (_, certificate, chain, errors) => Accept(certificate, chain, errors, authorities, checkHostname);
    }

    /// <summary>
    ///     The client certificate (<c>ssl_certfile</c>, with its key from <c>ssl_keyfile</c> or the same file), or
    ///     <see langword="null" /> when the URL names none.
    /// </summary>
    /// <param name="ep">The endpoint.</param>
    /// <returns>A certificate with its private key, or <see langword="null" />.</returns>
    /// <exception cref="TypeError">A key file without a certificate file; Python's message.</exception>
    /// <exception cref="FileNotFoundException">A missing file.</exception>
    /// <exception cref="CryptographicException">A malformed file, a key that does not match, or an encrypted key with a wrong or missing password.</exception>
    internal static X509Certificate2? ClientCertificate(RedisEndpoint ep)
    {
        if (ep.SslCertFile is null && ep.SslKeyFile is null)
            return null;
        if (ep.SslCertFile is null)
            throw new TypeError("certfile should be a valid filesystem path");
        X509Certificate2 loaded;
        if (ep.SslPassword is { } password)
        {
            try
            {
                loaded = X509Certificate2.CreateFromEncryptedPemFile(ep.SslCertFile, password, ep.SslKeyFile);
            }
            catch (CryptographicException encrypted)
            {
                // Python ignores the password of an unencrypted key; keep the first error if that fails too.
                try
                {
                    loaded = X509Certificate2.CreateFromPemFile(ep.SslCertFile, ep.SslKeyFile);
                }
                catch (CryptographicException)
                {
                    throw encrypted;
                }
            }
        }
        else
        {
            loaded = X509Certificate2.CreateFromPemFile(ep.SslCertFile, ep.SslKeyFile);
        }
        using (loaded)
        {
            // A PEM import leaves the key ephemeral, which Windows' TLS stack cannot use for a client certificate; a
            // PKCS#12 round trip gives it a stored key (StackExchange.Redis' own PEM helper does the same).
            byte[] pkcs12 = loaded.Export(X509ContentType.Pkcs12);
#if NET9_0_OR_GREATER
            return X509CertificateLoader.LoadPkcs12(pkcs12, null);
#else
            return new X509Certificate2(pkcs12);
#endif
        }
    }

    /// <summary>
    ///     The certificate authorities the URL adds, in Python's loading order: <c>ssl_ca_data</c>, then
    ///     <c>ssl_ca_certs</c>, then <c>ssl_ca_path</c>.
    /// </summary>
    /// <param name="ep">The endpoint.</param>
    /// <returns>The authorities; empty when the URL adds none.</returns>
    /// <exception cref="FileNotFoundException"><c>ssl_ca_certs</c> names a missing file.</exception>
    /// <exception cref="CryptographicException"><c>ssl_ca_data</c> or <c>ssl_ca_certs</c> holds no certificate, or a malformed one.</exception>
    internal static X509Certificate2Collection LoadAuthorities(RedisEndpoint ep)
    {
        var authorities = new X509Certificate2Collection();
        if (ep.SslCaData is { } data)
        {
            int before = authorities.Count;
            authorities.ImportFromPem(data);
            if (authorities.Count == before)
                throw new CryptographicException("no start line: cadata does not contain a certificate");
        }
        if (ep.SslCaCerts is { } file)
        {
            if (!File.Exists(file))
                throw new FileNotFoundException("[Errno 2] No such file or directory", file);
            int before = authorities.Count;
            authorities.ImportFromPemFile(file);
            if (authorities.Count == before)
                throw new CryptographicException("[X509: NO_CERTIFICATE_OR_CRL_FOUND] no certificate or crl found");
        }
        // OpenSSL reads a hashed directory lazily, during a handshake: a missing directory or an unreadable entry
        // only leaves those authorities out, it is never an error.
        if (ep.SslCaPath is { } directory && Directory.Exists(directory))
        {
            foreach (string path in Directory.EnumerateFiles(directory).Where(p => HashedCertificateName.IsMatch(Path.GetFileName(p))))
            {
                try
                {
                    authorities.ImportFromPemFile(path);
                }
                catch (Exception e) when (e is CryptographicException or IOException or UnauthorizedAccessException)
                {
                    // Skipped, as OpenSSL skips an entry it cannot read.
                }
            }
        }
        return authorities;
    }

    /// <summary>
    ///     Python's verdict on a server certificate .NET has already checked against the system's authorities: the host
    ///     name check is dropped when <c>ssl_check_hostname</c> is off, and a chain the system does not trust is accepted
    ///     when it ends at one of the URL's authorities.
    /// </summary>
    /// <param name="certificate">The server certificate.</param>
    /// <param name="chain">The chain .NET built, whose certificates (the server's intermediates included) help build the second one.</param>
    /// <param name="errors">.NET's verdict.</param>
    /// <param name="authorities">The URL's certificate authorities.</param>
    /// <param name="checkHostname">Whether the certificate must name the host.</param>
    /// <returns><see langword="true" /> to accept the certificate.</returns>
    internal static bool Accept(X509Certificate? certificate, X509Chain? chain, SslPolicyErrors errors,
        X509Certificate2Collection authorities, bool checkHostname)
    {
        if (certificate is null || (errors & SslPolicyErrors.RemoteCertificateNotAvailable) != 0)
            return false;
        if (!checkHostname)
            errors &= ~SslPolicyErrors.RemoteCertificateNameMismatch;
        if ((errors & SslPolicyErrors.RemoteCertificateChainErrors) != 0 && authorities.Count > 0)
        {
            using var custom = new X509Chain();
            custom.ChainPolicy.TrustMode = X509ChainTrustMode.CustomRootTrust;
            custom.ChainPolicy.CustomTrustStore.AddRange(authorities);
            // Python's default context never checks revocation.
            custom.ChainPolicy.RevocationMode = X509RevocationMode.NoCheck;
            custom.ChainPolicy.ApplicationPolicy.Add(new Oid(ServerAuthOid));
            if (chain is not null)
            {
                custom.ChainPolicy.ExtraStore.AddRange(chain.ChainPolicy.ExtraStore);
                foreach (X509ChainElement element in chain.ChainElements)
                    custom.ChainPolicy.ExtraStore.Add(element.Certificate);
            }
            X509Certificate2 leaf = certificate as X509Certificate2 ?? new X509Certificate2(certificate);
            if (custom.Build(leaf))
                errors &= ~SslPolicyErrors.RemoteCertificateChainErrors;
        }
        return errors == SslPolicyErrors.None;
    }
}
