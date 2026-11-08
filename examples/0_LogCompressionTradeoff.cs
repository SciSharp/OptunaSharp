#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Log-archive compression — a two-objective search (smaller archive vs. faster compression) over real
// System.IO.Compression codecs, with NSGA-II and a resumable JournalStorage file: run it again and the study
// picks up where it left off and keeps refining the same Pareto front.
//
// source: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html
//         (multi-objective studies, `directions=[...]`, `study.best_trials`)
//         https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.journal.JournalFileBackend.html
//         (the append-only journal file a study can resume from)
//         https://learn.microsoft.com/en-us/dotnet/api/system.io.compression.brotliencoder
//
// Real objective: every trial compresses the same deterministic 768 KiB web-server access log (generated
// below, seeded) with the codec the trial picks — Brotli (quality 0–11, window 10–24) or GZip/ZLib/Deflate
// (a CompressionLevel) — in independent chunks of 16–512 KiB, as a log shipper does so each chunk can be
// decompressed alone. Objective 1 is the compressed size as a percentage of the raw size (minimize),
// objective 2 the measured throughput in MB/s (maximize; best of 2 runs to damp timer noise). The search
// space is conditional (Brotli's knobs only exist under codec = brotli); NSGA-II evolves the parameters all
// trials share and re-draws the branch-only ones. Timings are real, so the front differs run to run.
//
//   Run:  dotnet run examples/0_LogCompressionTradeoff.cs          (run it twice: the trial count grows)

using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;
using System.Text;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

byte[] corpus = MakeAccessLog(768 * 1024);       // the data every trial compresses (identical across runs)

// a journal file next to the system temp folder: an append-only log of every study operation. A second
// process (or a second run) that opens the same file sees the same study.
string journalPath = Path.Combine(Path.GetTempPath(), "optunasharp-log-compression.journal");
var storage = new JournalStorage(new JournalFileBackend(journalPath));

var study = Optuna.CreateStudy(
    storage: storage,
    sampler: new NSGAIISampler(populationSize: 16, seed: 11),
    studyName: "log-compression",
    loadIfExists: true,                                                     // resume instead of failing
    directions: new[] { StudyDirection.Minimize, StudyDirection.Maximize }); // (size %, MB/s)

int before = study.Trials.Count;
study.Optimize(t => Objective(t, corpus), nTrials: 48);

var front = study.BestTrials.OrderBy(t => t.Values![0]).ToList();
Console.WriteLine($"journal: {journalPath}");
Console.WriteLine($"trials: {before} before this run, {study.Trials.Count} now; Pareto front: {front.Count} trials");
Console.WriteLine();
Console.WriteLine("  size %     MB/s   codec     settings");
foreach (var t in front)
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0,7:F2} {1,8:F1}   {2,-8}  {3}",
        t.Values![0], t.Values[1], t.Params["codec"], Settings(t)));

/// <summary>Compresses the corpus with the trial's codec settings and measures size and speed.</summary>
/// <param name="t">The trial proposing the codec, its settings and the chunk size.</param>
/// <param name="data">The raw log bytes to compress.</param>
/// <returns>Compressed size as a percentage of the raw size, and throughput in MB/s.</returns>
static IReadOnlyList<double> Objective(Trial t, byte[] data)
{
    string codec = t.SuggestCategorical("codec", new[] { "brotli", "gzip", "zlib", "deflate" });
    int chunk = (int)t.SuggestInt("chunk_kib", 16, 512, log: true) * 1024;   // independent chunks
    Func<byte[], int, int, int> compress;
    if (codec == "brotli")
    {
        int quality = (int)t.SuggestInt("quality", 0, 11);                  // brotli's speed/ratio dial
        int window = (int)t.SuggestInt("window", 10, 24);                   // log2 of the sliding window
        compress = (buf, off, len) => BrotliSize(buf, off, len, quality, window);
    }
    else
    {
        string level = t.SuggestCategorical("level", new[] { "Fastest", "Optimal", "SmallestSize" });
        var lvl = Enum.Parse<CompressionLevel>(level);
        compress = (buf, off, len) => StreamSize(codec, buf, off, len, lvl);
    }

    long total = 0;
    double bestSeconds = double.MaxValue;
    for (int rep = 0; rep < 2; rep++)                                       // best of 2: damp timer noise
    {
        var sw = Stopwatch.StartNew();
        total = 0;
        for (int off = 0; off < data.Length; off += chunk)
            total += compress(data, off, Math.Min(chunk, data.Length - off));
        bestSeconds = Math.Min(bestSeconds, sw.Elapsed.TotalSeconds);
    }
    double sizePct = 100.0 * total / data.Length;
    double mbPerSec = data.Length / 1e6 / Math.Max(bestSeconds, 1e-6);
    return new[] { sizePct, mbPerSec };
}

/// <summary>Brotli-compresses one chunk with an explicit quality and window and returns its size.</summary>
/// <param name="buf">The source buffer.</param>
/// <param name="off">Offset of the chunk.</param>
/// <param name="len">Length of the chunk.</param>
/// <param name="quality">Brotli quality, 0 (fastest) to 11 (densest).</param>
/// <param name="window">Brotli window bits, 10 to 24.</param>
/// <returns>The compressed byte count.</returns>
/// <exception cref="InvalidOperationException">The encoder reported a failure (it never should for a valid buffer).</exception>
static int BrotliSize(byte[] buf, int off, int len, int quality, int window)
{
    var dest = new byte[BrotliEncoder.GetMaxCompressedLength(len)];
    if (!BrotliEncoder.TryCompress(buf.AsSpan(off, len), dest, out int written, quality, window))
        throw new InvalidOperationException("brotli compression failed");
    return written;
}

/// <summary>Compresses one chunk through a GZip, ZLib or Deflate stream and returns its size.</summary>
/// <param name="codec">gzip, zlib or deflate.</param>
/// <param name="buf">The source buffer.</param>
/// <param name="off">Offset of the chunk.</param>
/// <param name="len">Length of the chunk.</param>
/// <param name="level">The .NET compression level.</param>
/// <returns>The compressed byte count, container framing included.</returns>
static int StreamSize(string codec, byte[] buf, int off, int len, CompressionLevel level)
{
    using var ms = new MemoryStream();
    // the compression stream must be disposed before the length is read: disposal flushes the final block.
    using (Stream z = codec switch
           {
               "gzip" => new GZipStream(ms, level, leaveOpen: true),
               "zlib" => new ZLibStream(ms, level, leaveOpen: true),
               _ => new DeflateStream(ms, level, leaveOpen: true),
           })
        z.Write(buf, off, len);
    return (int)ms.Length;
}

/// <summary>Formats the branch-specific settings of a trial.</summary>
/// <param name="t">A finished trial.</param>
/// <returns>A short human-readable settings string.</returns>
static string Settings(FrozenTrial t) => (string)t.Params["codec"]! == "brotli"
    ? $"q={t.Params["quality"]} window={t.Params["window"]} chunk={t.Params["chunk_kib"]} KiB"
    : $"level={t.Params["level"]} chunk={t.Params["chunk_kib"]} KiB";

/// <summary>Generates a deterministic, realistic combined-format web access log.</summary>
/// <param name="bytes">Approximate size of the log to produce.</param>
/// <returns>The UTF-8 bytes of the log (identical on every call, so runs compress the same data).</returns>
static byte[] MakeAccessLog(int bytes)
{
    var rng = new Random(12345);   // data generation only — samplers never use System.Random
    string[] paths = { "/", "/api/v1/orders", "/api/v1/orders/{0}", "/static/app.{0}.js", "/login", "/health",
        "/api/v1/users/{0}/cart", "/search?q=shoes&page={0}", "/img/p/{0}.webp" };
    string[] agents = { "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
        "curl/8.9.1", "kube-probe/1.31", "Googlebot/2.1 (+http://www.google.com/bot.html)" };
    int[] statuses = { 200, 200, 200, 200, 304, 301, 404, 500 };
    var sb = new StringBuilder();
    var time = new DateTime(2026, 11, 8, 0, 0, 0, DateTimeKind.Utc);
    while (sb.Length < bytes)
    {
        time = time.AddMilliseconds(rng.Next(1, 400));
        string path = string.Format(CultureInfo.InvariantCulture, paths[rng.Next(paths.Length)], rng.Next(1, 5000));
        sb.Append(CultureInfo.InvariantCulture,
            $"10.{rng.Next(0, 4)}.{rng.Next(0, 256)}.{rng.Next(1, 255)} - - [{time:dd/MMM/yyyy:HH:mm:ss} +0000] ");
        sb.Append(CultureInfo.InvariantCulture,
            $"\"GET {path} HTTP/1.1\" {statuses[rng.Next(statuses.Length)]} {rng.Next(120, 90000)} ");
        sb.Append(CultureInfo.InvariantCulture, $"\"-\" \"{agents[rng.Next(agents.Length)]}\" rt={rng.NextDouble():F3}\n");
    }
    return Encoding.UTF8.GetBytes(sb.ToString(0, bytes));
}
