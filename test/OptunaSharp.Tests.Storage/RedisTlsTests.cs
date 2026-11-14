using System;
using System.IO;
using System.Net.Security;
using System.Security.Cryptography;
using System.Security.Cryptography.X509Certificates;
using AwesomeAssertions;
using NumSharp;                          // TypeError, FileNotFoundError
using OptunaSharp.Storages.Journal;      // RedisTls (friend access), RedisEndpoint

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Unit tests (no server) for the shared TLS policy the StackExchange and NRedisStack backends apply to a
///     <c>rediss://</c> endpoint: Python's <c>ssl</c> semantics for redis-py's <c>ssl_*</c> arguments — the system's
///     authorities plus the URL's, the host name check unless turned off, Python's file-loading errors, a client
///     certificate from PEM files, and the TLS versions <c>ssl_min_version</c> allows.
/// </summary>
[TestClass]
public class RedisTlsTests
{
    /// <summary>The scratch folder for this test's PEM files, deleted afterwards.</summary>
    private string _dir = null!;

    /// <summary>Creates the scratch folder.</summary>
    [TestInitialize]
    public void CreateFolder()
    {
        _dir = Path.Combine(Path.GetTempPath(), $"optunasharp-tls-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_dir);
    }

    /// <summary>Deletes the scratch folder.</summary>
    [TestCleanup]
    public void DeleteFolder()
    {
        try
        {
            Directory.Delete(_dir, recursive: true);
        }
        catch (IOException)
        {
            // A handle still closing on Windows; the temp folder is cleaned by the OS eventually.
        }
    }

    /// <summary>
    ///     A test certificate authority and a server certificate it issued for <paramref name="host" />, with the
    ///     server's private key exported as unencrypted and as password-protected PKCS#8 PEM.
    /// </summary>
    /// <param name="host">The DNS name the server certificate names.</param>
    /// <returns>The authority, the server certificate (with its key), and the key's two PEM forms.</returns>
    private static (X509Certificate2 Ca, X509Certificate2 Leaf, string KeyPem, string EncryptedKeyPem) IssueChain(string host)
    {
        using RSA caKey = RSA.Create(2048);
        var caRequest = new CertificateRequest("CN=OptunaSharp Test CA", caKey, HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
        caRequest.CertificateExtensions.Add(new X509BasicConstraintsExtension(true, false, 0, true));
        caRequest.CertificateExtensions.Add(new X509KeyUsageExtension(X509KeyUsageFlags.KeyCertSign | X509KeyUsageFlags.CrlSign, true));
        X509Certificate2 ca = caRequest.CreateSelfSigned(DateTimeOffset.UtcNow.AddDays(-1), DateTimeOffset.UtcNow.AddDays(30));

        using RSA leafKey = RSA.Create(2048);
        var leafRequest = new CertificateRequest($"CN={host}", leafKey, HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
        var names = new SubjectAlternativeNameBuilder();
        names.AddDnsName(host);
        leafRequest.CertificateExtensions.Add(names.Build());
        leafRequest.CertificateExtensions.Add(new X509BasicConstraintsExtension(false, false, 0, false));
        leafRequest.CertificateExtensions.Add(new X509EnhancedKeyUsageExtension(new OidCollection { new Oid("1.3.6.1.5.5.7.3.1"), new Oid("1.3.6.1.5.5.7.3.2") }, false));
        using X509Certificate2 issued = leafRequest.Create(ca, DateTimeOffset.UtcNow.AddDays(-1), DateTimeOffset.UtcNow.AddDays(29), new byte[] { 1, 2, 3, 4 });
        X509Certificate2 leaf = issued.CopyWithPrivateKey(leafKey);
        string keyPem = leafKey.ExportPkcs8PrivateKeyPem();
        string encrypted = leafKey.ExportEncryptedPkcs8PrivateKeyPem(
            "pw", new PbeParameters(PbeEncryptionAlgorithm.Aes256Cbc, HashAlgorithmName.SHA256, 10_000));
        return (ca, leaf, keyPem, encrypted);
    }

    /// <summary>Writes <paramref name="text" /> to a file in the scratch folder.</summary>
    /// <param name="name">The file name.</param>
    /// <param name="text">The contents.</param>
    /// <returns>The file's full path.</returns>
    private string WriteFile(string name, string text)
    {
        string path = Path.Combine(_dir, name);
        File.WriteAllText(path, text);
        return path;
    }

    /// <summary>
    ///     <c>ssl_min_version</c> maps onto the TLS versions .NET may negotiate: none (the platform's default) for
    ///     no value or -2, every version from the minimum up otherwise, TLS 1.3 alone for 772 and -1.
    /// </summary>
    [TestMethod]
    public void Protocols_FollowSslMinVersion()
    {
        RedisTls.Protocols(new RedisEndpoint { Ssl = true }).Should().BeNull();
        RedisTls.Protocols(new RedisEndpoint { Ssl = true, SslMinVersion = -2 }).Should().BeNull();
        RedisTls.Protocols(new RedisEndpoint { Ssl = true, SslMinVersion = 771 }).Should().Be(System.Security.Authentication.SslProtocols.Tls12 | System.Security.Authentication.SslProtocols.Tls13);
        RedisTls.Protocols(new RedisEndpoint { Ssl = true, SslMinVersion = 772 }).Should().Be(System.Security.Authentication.SslProtocols.Tls13);
        RedisTls.Protocols(new RedisEndpoint { Ssl = true, SslMinVersion = -1 }).Should().Be(System.Security.Authentication.SslProtocols.Tls13);
    }

    /// <summary>
    ///     With the default settings no callback is installed, so .NET's own check (the system's authorities and the
    ///     host name, exactly Python's default context) applies; <c>ssl_cert_reqs=none</c> accepts any certificate.
    /// </summary>
    [TestMethod]
    public void Validation_DefaultIsDotNetsOwn_AndNoneAcceptsAnything()
    {
        RedisTls.Validation(RedisUrl.Parse("rediss://h")).Should().BeNull();
        RemoteCertificateValidationCallback none = RedisTls.Validation(RedisUrl.Parse("rediss://h?ssl_cert_reqs=none"))!;
        none(this, null, null, SslPolicyErrors.RemoteCertificateChainErrors | SslPolicyErrors.RemoteCertificateNameMismatch)
            .Should().BeTrue();
    }

    /// <summary>
    ///     A chain the system does not trust is accepted when it ends at one of the URL's authorities, and refused when
    ///     it does not; the host name check still applies unless <c>ssl_check_hostname</c> turns it off.
    /// </summary>
    [TestMethod]
    public void Accept_TrustsTheUrlsAuthorities_AndChecksTheHostName()
    {
        var (ca, leaf, _, _) = IssueChain("redis.test");
        var (otherCa, _, _, _) = IssueChain("other.test");
        using (ca)
        using (leaf)
        using (otherCa)
        {
            var authorities = new X509Certificate2Collection(ca);
            RedisTls.Accept(leaf, null, SslPolicyErrors.RemoteCertificateChainErrors, authorities, checkHostname: true).Should().BeTrue();
            RedisTls.Accept(leaf, null, SslPolicyErrors.RemoteCertificateChainErrors, new X509Certificate2Collection(otherCa), checkHostname: true)
                .Should().BeFalse();
            RedisTls.Accept(leaf, null, SslPolicyErrors.RemoteCertificateNameMismatch, authorities, checkHostname: true).Should().BeFalse();
            RedisTls.Accept(leaf, null, SslPolicyErrors.RemoteCertificateNameMismatch, authorities, checkHostname: false).Should().BeTrue();
            RedisTls.Accept(null, null, SslPolicyErrors.RemoteCertificateNotAvailable, authorities, checkHostname: false).Should().BeFalse();
        }
    }

    /// <summary>
    ///     The URL's authorities load in Python's order and with Python's errors: <c>ssl_ca_data</c> without a
    ///     certificate, a missing <c>ssl_ca_certs</c> file, one without a certificate; an <c>ssl_ca_path</c> directory
    ///     contributes its hash-named files only, and a missing directory contributes nothing, as OpenSSL reads it lazily.
    /// </summary>
    [TestMethod]
    public void LoadAuthorities_FollowsPythonsLoadVerifyLocations()
    {
        var (ca, leaf, _, _) = IssueChain("redis.test");
        using (ca)
        using (leaf)
        {
            string caPem = ca.ExportCertificatePem();
            RedisTls.LoadAuthorities(new RedisEndpoint { Ssl = true, SslCaData = caPem }).Count.Should().Be(1);
            RedisTls.LoadAuthorities(new RedisEndpoint { Ssl = true, SslCaCerts = WriteFile("ca.pem", caPem) }).Count.Should().Be(1);

            Action noData = () => RedisTls.LoadAuthorities(new RedisEndpoint { Ssl = true, SslCaData = "x" });
            noData.Should().Throw<CryptographicException>().WithMessage("no start line: cadata does not contain a certificate");
            Action missing = () => RedisTls.LoadAuthorities(new RedisEndpoint { Ssl = true, SslCaCerts = Path.Combine(_dir, "none.pem") });
            missing.Should().ThrowExactly<FileNotFoundError>().WithMessage("[Errno 2] No such file or directory");
            Action empty = () => RedisTls.LoadAuthorities(new RedisEndpoint { Ssl = true, SslCaCerts = WriteFile("empty.pem", "hello") });
            empty.Should().Throw<CryptographicException>().WithMessage("[X509: NO_CERTIFICATE_OR_CRL_FOUND] no certificate or crl found");

            string hashed = Path.Combine(_dir, "hashed");
            Directory.CreateDirectory(hashed);
            File.WriteAllText(Path.Combine(hashed, "0a1b2c3d.0"), caPem);
            File.WriteAllText(Path.Combine(hashed, "ca.pem"), caPem);
            RedisTls.LoadAuthorities(new RedisEndpoint { Ssl = true, SslCaPath = hashed }).Count.Should().Be(1);
            RedisTls.LoadAuthorities(new RedisEndpoint { Ssl = true, SslCaPath = Path.Combine(_dir, "nowhere") }).Count.Should().Be(0);
        }
    }

    /// <summary>
    ///     The client certificate loads from PEM as Python's <c>load_cert_chain</c> does: the key from
    ///     <c>ssl_keyfile</c> or from the certificate file itself, a password for an encrypted key (ignored for a plain
    ///     one), Python's <c>TypeError</c> for a key file alone, and a failure for a wrong password.
    /// </summary>
    [TestMethod]
    public void ClientCertificate_FollowsPythonsLoadCertChain()
    {
        var (ca, leaf, keyPem, encryptedKeyPem) = IssueChain("client.test");
        using (ca)
        using (leaf)
        {
            string certPem = leaf.ExportCertificatePem();
            string certFile = WriteFile("client.crt", certPem);
            string keyFile = WriteFile("client.key", keyPem);
            string encryptedKeyFile = WriteFile("client-encrypted.key", encryptedKeyPem);
            string combinedFile = WriteFile("client-combined.pem", certPem + "\n" + keyPem);

            RedisTls.ClientCertificate(new RedisEndpoint { Ssl = true }).Should().BeNull();
            using (X509Certificate2 separate = RedisTls.ClientCertificate(new RedisEndpoint { Ssl = true, SslCertFile = certFile, SslKeyFile = keyFile })!)
                separate.HasPrivateKey.Should().BeTrue();
            using (X509Certificate2 combined = RedisTls.ClientCertificate(new RedisEndpoint { Ssl = true, SslCertFile = combinedFile })!)
                combined.HasPrivateKey.Should().BeTrue();
            using (X509Certificate2 encrypted = RedisTls.ClientCertificate(
                       new RedisEndpoint { Ssl = true, SslCertFile = certFile, SslKeyFile = encryptedKeyFile, SslPassword = "pw" })!)
                encrypted.HasPrivateKey.Should().BeTrue();
            using (X509Certificate2 ignored = RedisTls.ClientCertificate(
                       new RedisEndpoint { Ssl = true, SslCertFile = certFile, SslKeyFile = keyFile, SslPassword = "unused" })!)
                ignored.HasPrivateKey.Should().BeTrue();

            Action keyOnly = () => RedisTls.ClientCertificate(new RedisEndpoint { Ssl = true, SslKeyFile = keyFile });
            keyOnly.Should().Throw<TypeError>().WithMessage("certfile should be a valid filesystem path");
            Action wrongPassword = () => RedisTls.ClientCertificate(
                new RedisEndpoint { Ssl = true, SslCertFile = certFile, SslKeyFile = encryptedKeyFile, SslPassword = "wrong" });
            wrongPassword.Should().Throw<CryptographicException>();
        }
    }
}
