using System;
using AwesomeAssertions;
using NumSharp;                          // ValueError, TypeError
using OptunaSharp.Storages.Journal;      // RedisUrl / RedisEndpoint (shared package)

namespace OptunaSharp.Tests.Storage;

/// <summary>
///     Unit tests for the shared <c>redis://</c> → <see cref="RedisEndpoint" /> parser — the one bit of
///     bespoke glue (StackExchange.Redis / ServiceStack have no common URI parser; redis-py's
///     <c>from_url</c> gave Optuna this). All three backends build their client from this, so parsing
///     once here is what keeps their connection interpretation identical. No server required.
/// </summary>
[TestClass]
public class RedisUrlTests
{
    [TestMethod]
    public void PlainHostPort()
    {
        var e = RedisUrl.Parse("redis://localhost:6379");
        e.Host.Should().Be("localhost");
        e.Port.Should().Be(6379);
        e.Ssl.Should().BeFalse();
        e.Db.Should().BeNull();
        e.User.Should().BeNull();
        e.Password.Should().BeNull();
        e.UnixSocketPath.Should().BeNull();
    }

    [TestMethod]
    public void DefaultsPortWhenOmitted() => RedisUrl.Parse("redis://localhost").Port.Should().Be(6379);

    [TestMethod]
    public void PasswordOnly_NoUser_WithDbInPath()
    {
        var e = RedisUrl.Parse("redis://:secret@127.0.0.1:6380/3");
        e.Host.Should().Be("127.0.0.1");
        e.Port.Should().Be(6380);
        e.Password.Should().Be("secret");
        e.User.Should().BeNull();
        e.Db.Should().Be(3);
    }

    [TestMethod]
    public void UserAndPercentEncodedPassword()
    {
        var e = RedisUrl.Parse("redis://alice:p%40ss@redis.example.com:6379/1");
        e.User.Should().Be("alice");
        e.Password.Should().Be("p@ss");   // %40 → '@'
        e.Db.Should().Be(1);
    }

    [TestMethod]
    public void RedissEnablesTls()
    {
        var e = RedisUrl.Parse("rediss://user:pw@redis.example.com:6380");
        e.Ssl.Should().BeTrue();
        e.Host.Should().Be("redis.example.com");
        e.Port.Should().Be(6380);
    }

    [TestMethod]
    public void QueryDbOverridesPathDb() => RedisUrl.Parse("redis://host:6379/2?db=5").Db.Should().Be(5);

    [TestMethod]
    public void QueryUsernameAndPassword()
    {
        var e = RedisUrl.Parse("redis://host:6379?username=alice&password=secret");
        e.User.Should().Be("alice");
        e.Password.Should().Be("secret");
    }

    [TestMethod]
    public void UserOnly_NoPassword()
    {
        var e = RedisUrl.Parse("redis://alice@host:6379");
        e.User.Should().Be("alice");
        e.Password.Should().BeNull();
    }

    [TestMethod]
    public void UnixSocket()
    {
        var e = RedisUrl.Parse("unix:///var/run/redis/redis.sock?db=0&password=abc");
        e.UnixSocketPath.Should().Be("/var/run/redis/redis.sock");
        e.Db.Should().Be(0);
        e.Password.Should().Be("abc");
    }

    [TestMethod]
    public void NonNumericDbPath_IsIgnored() => RedisUrl.Parse("redis://host:6379/not-a-db").Db.Should().BeNull();

    [TestMethod]
    public void ValuelessQueryFlag_IsIgnored() => RedisUrl.Parse("redis://host:6379?verbose&db=2").Db.Should().Be(2);

    [TestMethod]
    public void UnknownScheme_ThrowsValueError()
    {
        var act = () => RedisUrl.Parse("http://localhost:6379");
        act.Should().Throw<ValueError>()
            .WithMessage("*Redis URL must specify one of the following schemes (redis://, rediss://, unix://)*");
    }

    /// <summary>A string without one of redis-py's three schemes fails redis-py's scheme check, whatever else it holds.</summary>
    [TestMethod]
    public void Malformed_ThrowsValueError()
    {
        var act = () => RedisUrl.Parse("not-a-url");
        act.Should().Throw<ValueError>()
            .WithMessage("Redis URL must specify one of the following schemes (redis://, rediss://, unix://)");
    }

    // ------------------------------------------------------------------ redis-py 8.1 semantics
    //
    // Every expectation below is what redis-py 8.1's ConnectionPool.from_url plus its first connect make of the URL
    // (observed with the live twin in the interop suite's RedisUrlStandardParityTests).

    /// <summary>
    ///     redis-py's scheme test is a case-sensitive prefix test: an upper-case scheme, a leading space or a single
    ///     slash fails it.
    /// </summary>
    /// <param name="url">A URL redis-py refuses.</param>
    [TestMethod]
    [DataRow("REDIS://h")]
    [DataRow("Redis://h")]
    [DataRow(" redis://h")]
    [DataRow("redis:/h")]
    public void SchemeCheck_IsACaseSensitivePrefixTest(string url)
    {
        var act = () => RedisUrl.Parse(url);
        act.Should().Throw<ValueError>().WithMessage("Redis URL must specify one of the following schemes*");
    }

    /// <summary>
    ///     redis-py 8's defaults: 5-second command and connect timeouts (TCP and unix sockets alike), port 6379 for
    ///     <c>rediss://</c> too, no protocol pinned, certificates required and host names checked.
    /// </summary>
    [TestMethod]
    public void Defaults_AreRedisPy8s()
    {
        var e = RedisUrl.Parse("redis://h");
        e.SocketTimeout.Should().Be(5);
        e.SocketConnectTimeout.Should().Be(5);
        e.Protocol.Should().BeNull();
        RedisUrl.Parse("unix:///s").SocketTimeout.Should().Be(5);
        var tls = RedisUrl.Parse("rediss://h");
        tls.Port.Should().Be(6379);
        tls.SslCertReqs.Should().Be("required");
        tls.SslCheckHostname.Should().BeTrue();
    }

    /// <summary>
    ///     Typed query options reach the endpoint: floats for the timeouts (Python's <c>float</c>, so <c>%201_0%20</c>
    ///     is 10), ints for <c>protocol</c>, text for the names.
    /// </summary>
    [TestMethod]
    public void QueryOptions_AreTypedAsRedisPyTypesThem()
    {
        var e = RedisUrl.Parse("redis://h?socket_timeout=2.5&socket_connect_timeout=1&client_name=c&lib_name=l&protocol=2");
        e.SocketTimeout.Should().Be(2.5);
        e.SocketConnectTimeout.Should().Be(1);
        e.ClientName.Should().Be("c");
        e.LibraryName.Should().Be("l");
        e.Protocol.Should().Be(2);
        RedisUrl.Parse("redis://h?socket_timeout=%201_0%20").SocketTimeout.Should().Be(10);
        RedisUrl.Parse("redis://h?protocol=%203").Protocol.Should().Be(3);
    }

    /// <summary>
    ///     The user information wins over <c>?username=</c>/<c>?password=</c>, the first of a repeated key wins, and
    ///     query values are form-decoded (<c>+</c> is a space, <c>%2B</c> a plus).
    /// </summary>
    [TestMethod]
    public void QueryPrecedenceAndDecoding_FollowParseQs()
    {
        var e = RedisUrl.Parse("redis://u:p@h?username=x&password=y");
        e.User.Should().Be("u");
        e.Password.Should().Be("p");
        RedisUrl.Parse("redis://h?db=1&db=2").Db.Should().Be(1);
        RedisUrl.Parse("redis://h?password=a+b").Password.Should().Be("a b");
        RedisUrl.Parse("redis://h?password=a%2Bb").Password.Should().Be("a+b");
        RedisUrl.Parse("redis://u:p@s@h").Password.Should().Be("p@s");
    }

    /// <summary><c>?host=</c> and <c>?port=</c> are used when the URL itself has no host or port; the URL's own win otherwise.</summary>
    [TestMethod]
    public void QueryHostAndPort_FillWhatTheUrlLeavesOut()
    {
        var e = RedisUrl.Parse("redis://?host=qh&port=6390");
        e.Host.Should().Be("qh");
        e.Port.Should().Be(6390);
        RedisUrl.Parse("redis://h?host=qh").Host.Should().Be("h");
        RedisUrl.Parse("redis://h:6391?port=6390").Port.Should().Be(6391);
        RedisUrl.Parse("redis://?port=%206390%20").Port.Should().Be(6390);
    }

    /// <summary>
    ///     The path is the database only without <c>?db=</c>, and only when the slash-less, percent-decoded path is a
    ///     Python int; a <c>unix://</c> path is the socket, never a database.
    /// </summary>
    [TestMethod]
    public void PathDatabase_FollowsPythonInt()
    {
        RedisUrl.Parse("redis://h/1/2").Db.Should().Be(12);
        RedisUrl.Parse("redis://h/%31").Db.Should().Be(1);
        RedisUrl.Parse("redis://h/ 1").Db.Should().Be(1);
        RedisUrl.Parse("redis://h/+1").Db.Should().Be(1);
        RedisUrl.Parse("redis://h/1_0").Db.Should().Be(10);
        RedisUrl.Parse("redis://h/abc").Db.Should().BeNull();
        var unix = RedisUrl.Parse("unix:///tmp/redis.sock/3");
        unix.UnixSocketPath.Should().Be("/tmp/redis.sock/3");
        unix.Db.Should().BeNull();
    }

    /// <summary>The host is lower-cased and then percent-decoded (so <c>%4A</c> stays upper-case); IPv6 brackets are dropped.</summary>
    [TestMethod]
    public void Host_IsLowerCasedThenDecoded()
    {
        RedisUrl.Parse("redis://H.Example.COM").Host.Should().Be("h.example.com");
        RedisUrl.Parse("redis://H%4A").Host.Should().Be("hJ");
        RedisUrl.Parse("redis://u:p@[::1]:7000").Host.Should().Be("::1");
        RedisUrl.Parse("redis://h:0").Port.Should().Be(6379);
        RedisUrl.Parse("redis://h:").Port.Should().Be(6379);
    }

    /// <summary>
    ///     TLS options: <c>ssl_cert_reqs=none</c> turns the host name check off even when the URL asks for it,
    ///     <c>ssl_check_hostname</c> takes redis-py's <c>to_bool</c> words, and <c>ssl_min_version</c> keeps Python's
    ///     <c>TLSVersion</c> value.
    /// </summary>
    [TestMethod]
    public void TlsOptions_FollowSslConnection()
    {
        var none = RedisUrl.Parse("rediss://h?ssl_cert_reqs=none&ssl_check_hostname=true");
        none.SslCertReqs.Should().Be("none");
        none.SslCheckHostname.Should().BeFalse();
        RedisUrl.Parse("rediss://h?ssl_check_hostname=no").SslCheckHostname.Should().BeFalse();
        RedisUrl.Parse("rediss://h?ssl_check_hostname=maybe").SslCheckHostname.Should().BeTrue();
        RedisUrl.Parse("rediss://h?ssl_min_version=771").SslMinVersion.Should().Be(771);
        var files = RedisUrl.Parse("rediss://h?ssl_ca_certs=/x/ca.pem&ssl_certfile=c&ssl_keyfile=k&ssl_password=pw");
        files.SslCaCerts.Should().Be("/x/ca.pem");
        files.SslCertFile.Should().Be("c");
        files.SslKeyFile.Should().Be("k");
        files.SslPassword.Should().Be("pw");
    }

    /// <summary>
    ///     Options that only tune redis-py's own pool, retries, health checks or decoding are accepted without effect,
    ///     as is an <c>encoding</c> that names UTF-8 and a <c>connection_class</c> that <c>rediss://</c> replaces.
    /// </summary>
    [TestMethod]
    public void PoolAndDecodingOptions_AreAcceptedWithoutEffect()
    {
        var e = RedisUrl.Parse(
            "redis://h?max_connections=5&retry_on_timeout=yes&health_check_interval=30&decode_responses=true&encoding=UTF8&lib_version=9");
        e.Host.Should().Be("h");
        RedisUrl.Parse("rediss://h?connection_class=x").Ssl.Should().BeTrue();
        RedisUrl.Parse("redis://h?socket_keepalive=0&socket_keepalive_options=x").Host.Should().Be("h");
    }

    /// <summary>
    ///     What redis-py rejects is rejected with redis-py's (or Python's) exception kind and message; redis-py's own
    ///     <c>ConnectionError</c>/<c>RedisError</c> arrive as <see cref="ValueError" /> with the same text.
    /// </summary>
    /// <param name="url">The URL.</param>
    /// <param name="kind">The expected exception type.</param>
    /// <param name="message">The expected message, exactly.</param>
    [TestMethod]
    [DataRow("redis://h?db=abc", typeof(ValueError), "Invalid value for 'db' in connection URL.")]
    [DataRow("redis://h?socket_timeout=1,5", typeof(ValueError), "Invalid value for 'socket_timeout' in connection URL.")]
    [DataRow("rediss://h?ssl_include_verify_flags=BOGUS", typeof(ValueError), "Invalid value for 'ssl_include_verify_flags' in connection URL.")]
    [DataRow("redis://h?foo=bar", typeof(TypeError), "AbstractConnection.__init__() got an unexpected keyword argument 'foo'")]
    [DataRow("redis://h?bar=2&foo=1", typeof(TypeError), "AbstractConnection.__init__() got an unexpected keyword argument 'bar'")]
    [DataRow("redis://h?timeout=3", typeof(TypeError), "AbstractConnection.__init__() got an unexpected keyword argument 'timeout'")]
    [DataRow("redis://h?ssl_cert_reqs=none", typeof(TypeError), "AbstractConnection.__init__() got an unexpected keyword argument 'ssl_cert_reqs'")]
    [DataRow("unix:///s?host=x", typeof(TypeError), "AbstractConnection.__init__() got an unexpected keyword argument 'host'")]
    [DataRow("redis://h?path=/x", typeof(TypeError), "AbstractConnection.__init__() got an unexpected keyword argument 'path'")]
    [DataRow("redis://h?self=1", typeof(TypeError), "ConnectionPool.__init__() got multiple values for argument 'self'")]
    [DataRow("redis://h?connection_class=x", typeof(TypeError), "'str' object is not callable")]
    [DataRow("redis://h?parser_class=x", typeof(TypeError), "'str' object is not callable")]
    [DataRow("redis://h?driver_info=x", typeof(TypeError), "driver_info must be a DriverInfo instance or None")]
    [DataRow("redis://h?protocol=4", typeof(ValueError), "protocol must be either 2 or 3")]
    [DataRow("rediss://h?ssl_cert_reqs=NONE", typeof(ValueError), "Invalid SSL Certificate Requirements Flag: NONE")]
    [DataRow("redis://h?max_connections=-1", typeof(ValueError), "\"max_connections\" must be a positive integer")]
    [DataRow("redis://h:99999", typeof(ValueError), "Port out of range 0-65535")]
    [DataRow("redis://h:abc", typeof(ValueError), "Port could not be cast to integer value as 'abc'")]
    [DataRow("redis://?port=abc", typeof(ValueError), "invalid literal for int() with base 10: 'abc'")]
    [DataRow("redis://[::1", typeof(ValueError), "Invalid IPv6 URL")]
    [DataRow("redis://h?socket_timeout=-1", typeof(ValueError), "Timeout value out of range")]
    [DataRow("redis://h?socket_connect_timeout=nan", typeof(ValueError), "Invalid value NaN (not a number)")]
    [DataRow("redis://h?socket_timeout=inf", typeof(OverflowException), "timestamp out of range for platform time_t")]
    [DataRow("rediss://h?ssl_min_version=999", typeof(ValueError), "Unsupported TLS/SSL version 0x3e7")]
    [DataRow("rediss://h?ssl_min_version=-3", typeof(ValueError), "Unsupported TLS/SSL version 0xfffffffd")]
    [DataRow("rediss://h?ssl_validate_ocsp=1&ssl_validate_ocsp_stapled=1", typeof(ValueError), "Either an OCSP staple or pure OCSP connection must be validated - not both.")]
    [DataRow("rediss://h?ssl_keyfile=k", typeof(TypeError), "certfile should be a valid filesystem path")]
    [DataRow("rediss://h?ssl_keyfile=k&ssl_min_version=999", typeof(TypeError), "certfile should be a valid filesystem path")]
    public void RedisPyRejections_KeepTheirKindAndMessage(string url, Type kind, string message)
    {
        var act = () => RedisUrl.Parse(url);
        Exception e = act.Should().Throw<Exception>().Which;
        e.GetType().Should().Be(kind);
        e.Message.Should().Be(message);
    }

    /// <summary>
    ///     Values redis-py accepts but its server or socket then refuses are refused up front, with the server's
    ///     message: a client name with a space, a database index outside 0–2147483647, a <c>?port=</c> outside 1–65535.
    /// </summary>
    /// <param name="url">The URL.</param>
    /// <param name="message">The expected message prefix.</param>
    [TestMethod]
    [DataRow("redis://h?client_name=a%20b", "Client names cannot contain spaces, newlines or special characters.")]
    [DataRow("redis://h?db=-1", "DB index is out of range")]
    [DataRow("redis://h/99999999999", "DB index is out of range")]
    [DataRow("redis://?port=0", "port must be 1-65535, not 0")]
    [DataRow("redis://?port=99999", "port must be 1-65535, not 99999")]
    public void ServerRules_AreAppliedUpFront(string url, string message)
    {
        var act = () => RedisUrl.Parse(url);
        act.Should().Throw<ValueError>().WithMessage(message);
    }

    /// <summary>
    ///     Arguments that take a Python object, a non-UTF-8 encoding, TLS settings no .NET client offers, and a zero
    ///     (non-blocking) timeout are refused with <see cref="NotSupportedException" /> rather than silently ignored.
    /// </summary>
    /// <param name="url">The URL.</param>
    [TestMethod]
    [DataRow("redis://h?retry=x")]
    [DataRow("redis://h?credential_provider=x")]
    [DataRow("redis://h?cache_config=x")]
    [DataRow("redis://h?encoding=utf-16")]
    [DataRow("rediss://h?ssl_ciphers=HIGH")]
    [DataRow("rediss://h?ssl_include_verify_flags=VERIFY_X509_STRICT")]
    [DataRow("rediss://h?ssl_validate_ocsp_stapled=true")]
    [DataRow("redis://h?socket_keepalive_options=x")]
    [DataRow("redis://h?socket_timeout=0")]
    public void UnhonorableOptions_AreRefused(string url)
    {
        var act = () => RedisUrl.Parse(url);
        act.Should().Throw<NotSupportedException>();
    }

    /// <summary>
    ///     A user name or password next to a <c>credential_provider</c> raises redis-py's <c>DataError</c> text, before
    ///     the provider itself is refused.
    /// </summary>
    [TestMethod]
    public void CredentialsWithCredentialProvider_RaiseDataError()
    {
        var act = () => RedisUrl.Parse("redis://u:p@h?credential_provider=x");
        act.Should().Throw<ValueError>().WithMessage("'username' and 'password' cannot be passed along with 'credential_provider'.*");
    }

    /// <summary>Timeouts become whole milliseconds for the .NET clients: rounded up, at least 1, capped at <see cref="int.MaxValue" />.</summary>
    [TestMethod]
    public void TimeoutMilliseconds_RoundUpAndCap()
    {
        RedisEndpoint.ToMilliseconds(5).Should().Be(5000);
        RedisEndpoint.ToMilliseconds(0.0001).Should().Be(1);
        RedisEndpoint.ToMilliseconds(2.0004).Should().Be(2001);
        RedisEndpoint.ToMilliseconds(1e9).Should().Be(int.MaxValue);
    }
}
