using System.Runtime.CompilerServices;

// The three client backends apply a rediss:// endpoint's TLS settings (certificate checks, client certificate, minimum
// TLS version) through the shared internal RedisTls helper, and convert its timeouts with RedisEndpoint.ToMilliseconds.
// Strong-named, so PublicKey= is required (Microsoft's published open-source key, token cc7b13ffcd2ddd51, shared
// across the SciSharp STACK; identity, not an access control). Core's internal friend-key constant is visible here
// because Core declares this package a friend.
[assembly: InternalsVisibleTo("OptunaSharp.Storage.StackExchange.Redis, PublicKey=" + OptunaSharpFriendKey.PublicKey)]
[assembly: InternalsVisibleTo("OptunaSharp.Storage.NRedisStack, PublicKey=" + OptunaSharpFriendKey.PublicKey)]
[assembly: InternalsVisibleTo("OptunaSharp.Storage.ServiceStack.Redis, PublicKey=" + OptunaSharpFriendKey.PublicKey)]
// The storage test project checks the TLS policy (certificate callbacks, file loading) directly.
[assembly: InternalsVisibleTo("OptunaSharp.Tests.Storage, PublicKey=" + OptunaSharpFriendKey.PublicKey)]
