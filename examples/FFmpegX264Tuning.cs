#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false
#:property SkipNDScopeWeave=true
#:property GeneratePackageOnBuild=false

// FFmpeg / x264 encoder-parameter tuning — an OptunaSharp port of the most-starred public Optuna
// gist (25★), by Optuna maintainer `sile`:
//   https://gist.github.com/sile/8aa1ff7808dd55298f51dd70c8b83092
//   "Optunaを使ってFFmpegのエンコードパラメータを最適化してみる"
//
// Faithful to the gist, this runs a REAL ffmpeg encode and maximizes the actual SSIM libx264 reports —
// over the gist's conditional x264 search space (psy-rd under trellis>0; deblock under not-no-deblock;
// dynamic partitions-*/feature-flag names), with a SuccessiveHalvingPruner that stops hopeless trials
// at short cut-points. The test clip is the rickroll fetched from YouTube (dQw4w9WgXcQ) — sile tuned
// Big Buck Bunny; we tune "Never Gonna Give You Up".
//
// Requires ffmpeg (SSIM via libx264 `ssim=1`). Video source, in order: $FFMPEG_TUNE_VIDEO → cached
// download → yt-dlp of the rickroll (first 30s, low-res) → an ffmpeg `testsrc2` synthetic clip if
// yt-dlp is unavailable. Knobs: $FFMPEG_TUNE_TRIALS (default 12), $FFMPEG_TUNE_URL.
// (suggest_uniform, removed in Optuna 3.0, is written as its modern equivalent SuggestFloat.)
//
//   Run:  dotnet run examples/FFmpegX264Tuning.cs
//         FFMPEG_TUNE_TRIALS=6 dotnet run examples/FFmpegX264Tuning.cs

using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using OptunaSharp;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

var url = Environment.GetEnvironmentVariable("FFMPEG_TUNE_URL") is { Length: > 0 } u
    ? u : "https://www.youtube.com/watch?v=dQw4w9WgXcQ";
int trials = int.TryParse(Environment.GetEnvironmentVariable("FFMPEG_TUNE_TRIALS"), out int n) ? n : 12;
int[] rungs = { 2, 4, 8 };   // seconds encoded at each SuccessiveHalving rung
var inv = CultureInfo.InvariantCulture;

if (!Which("ffmpeg"))
{
    Console.Error.WriteLine("ffmpeg not found on PATH — this example encodes real video. Install ffmpeg and retry.");
    return;
}

var (source, label) = AcquireVideo();
Console.WriteLine($"Tuning x264 to maximize SSIM on: {label}");
Console.WriteLine($"  source: {source}");

var study = Optuna.CreateStudy(
    storage: new InMemoryStorage(),
    sampler: new TPESampler(seed: 0),   // the gist's default sampler (seeded here for reproducibility)
    pruner: new OptunaSharp.Pruners.SuccessiveHalvingPruner(minResource: 2, reductionFactor: 2),
    direction: StudyDirection.Maximize,
    studyName: "ffmpeg-x264");

study.Optimize(Objective, nTrials: trials);

Console.WriteLine($"\nFinished trials: {study.Trials.Count}");
Console.WriteLine($"Best SSIM: {study.BestValue.ToString("0.000000", inv)}");
Console.WriteLine("Best x264 params:");
foreach (var kv in study.BestParams)
    Console.WriteLine($"    {kv.Key}: {kv.Value}");

// The gist's x264 search space; each set is encoded for real and scored by libx264's SSIM.
double Objective(Trial t)
{
    long refs = t.SuggestInt("refs", 1, 16);
    double qcomp = t.SuggestFloat("qcomp", 0.0, 1.0);
    long bAdapt = t.SuggestInt("b-adapt", 0, 2);
    long subme = t.SuggestInt("subme", 1, 9);
    long trellis = t.SuggestInt("trellis", 0, 2);
    string psy = "";
    if (trellis > 0)
        psy = $":psy-rd={t.SuggestFloat("psy-rd.0", 0, 1).ToString("0.###", inv)}," +
              $"{t.SuggestFloat("psy-rd.1", 0, 1).ToString("0.###", inv)}";

    var parts = new List<string>();
    foreach (var p in new[] { "p8x8", "p4x4", "b8x8", "i8x8", "i4x4" })
        if (t.SuggestCategorical($"partitions-{p}", new[] { 0, 1 }) != 0)
            parts.Add(p);
    string partitions = parts.Count == 0 ? "none" : string.Join(",", parts);

    string me = t.SuggestCategorical("me", new[] { "dia", "hex", "umh", "esa" });
    long meRange = t.SuggestInt("me_range", 4, 16);
    string direct = t.SuggestCategorical("direct", new[] { "none", "spatial", "temporal", "auto" });
    long rcLookahead = t.SuggestInt("rc-lookahead", 10, 100);
    long minKeyint = t.SuggestInt("min-keyint", 1, 12);
    long scenecut = t.SuggestInt("scenecut", 10, 100);
    t.SuggestInt("qdiff", 1, 51);   // suggested for search-space fidelity (no direct x264-params key)

    bool noDeblock = t.SuggestCategorical("no-deblock", new[] { 0, 1 }) != 0;
    string deblock = noDeblock
        ? "no-deblock=1"
        : $"deblock={t.SuggestInt("deblockalpha", -6, 6)},{t.SuggestInt("deblockbeta", -6, 6)}";

    // each feature flag is a [0,1] categorical; when it's on AND maps to a real x264 option, append it
    var flags = new StringBuilder();
    foreach (var f in new[]
             {
                 "b-pyramid", "no-cabac", "mixed-refs", "no-chroma-me", "8x8dct",
                 "no-fast-pskip", "no-dct-decimate", "deadzone-inter", "deadzone-intra", "nr",
             })
        if (t.SuggestCategorical(f, new[] { 0, 1 }) != 0 && FlagOpt(f) is { } opt)
            flags.Append(':').Append(opt);

    // assemble the full x264-params string: fixed keyint + every tuned knob + ssim=1 so libx264 reports SSIM
    string x264 =
        $"keyint=12:ref={refs}:qcomp={qcomp.ToString("0.####", inv)}:bframes=3:b-adapt={bAdapt}:" +
        $"subme={subme}:trellis={trellis}:partitions={partitions}:me={me}:merange={meRange}:" +
        $"direct={direct}:rc-lookahead={rcLookahead}:min-keyint={minKeyint}:scenecut={scenecut}:" +
        $"{deblock}{flags}{psy}:ssim=1";

    // SuccessiveHalving rungs: encode a longer slice each time, report SSIM, and prune trials that are
    // clearly behind before spending time on the longer encodes
    double ssim = 0;
    foreach (int dur in rungs)
    {
        ssim = Encode(source, dur, x264);   // real ffmpeg encode → libx264 SSIM (0 if the combo fails)
        t.Report(ssim, dur);
        if (t.ShouldPrune())
            throw new TrialPruned();
    }
    return ssim;
}

// Encode `dur` seconds of `src` with the given x264 params and return the SSIM libx264 prints.
double Encode(string src, int dur, string x264Params)
{
    var (code, err) = Run("ffmpeg",
        $"-y -hide_banner -nostdin -i \"{src}\" -t {dur} -an -c:v libx264 -b:v 100k -s 320x240 " +
        $"-x264-params \"{x264Params}\" -f null -", 120);
    if (code != 0)
        return 0.0;                              // an invalid x264 combo → treat as worst (SSIM 0)
    // libx264 prints e.g. "SSIM Mean Y:0.987654 (19.1db)" on stderr when ssim=1 — pull the number out
    var m = Regex.Match(err, @"SSIM Mean Y:\s*([0-9.]+)");
    return m.Success ? double.Parse(m.Groups[1].Value, inv) : 0.0;
}

(string path, string label) AcquireVideo()
{
    if (Environment.GetEnvironmentVariable("FFMPEG_TUNE_VIDEO") is { Length: > 0 } given && File.Exists(given))
        return (given, "user-provided video ($FFMPEG_TUNE_VIDEO)");

    string dir = Path.Combine(Path.GetTempPath(), "optunasharp-ffmpeg-tune");
    Directory.CreateDirectory(dir);
    string cache = Path.Combine(dir, "source.mp4");
    if (File.Exists(cache))
        return (cache, "cached rickroll clip");

    if (Which("yt-dlp"))
    {
        Console.WriteLine("Fetching the rickroll (first 30s) via yt-dlp…");
        var (code, _) = Run("yt-dlp",
            $"--no-warnings --no-playlist --download-sections \"*0-30\" " +
            $"-f \"worst[ext=mp4]/worst\" -o \"{cache}\" \"{url}\"", 240);
        if (code == 0 && File.Exists(cache))
            return (cache, "Rick Astley — Never Gonna Give You Up (YouTube, first 30s)");
        Console.WriteLine("yt-dlp fetch failed; falling back to a synthetic ffmpeg testsrc clip.");
    }
    else
    {
        Console.WriteLine("yt-dlp not found; using a synthetic ffmpeg testsrc clip instead of the rickroll.");
    }

    Run("ffmpeg", $"-y -hide_banner -loglevel error -f lavfi -i \"testsrc2=size=640x480:rate=25\" -t 12 \"{cache}\"", 60);
    return (cache, "synthetic ffmpeg testsrc2 clip (rickroll unavailable)");
}

// Map the gist's feature flags to valid x264-params (unmapped ones stay suggested but unapplied).
static string? FlagOpt(string flag) => flag switch
{
    "b-pyramid" => "b-pyramid=normal",
    "no-cabac" => "cabac=0",
    "mixed-refs" => "mixed-refs=1",
    "no-chroma-me" => "chroma-me=0",
    "8x8dct" => "8x8dct=1",
    "no-fast-pskip" => "fast-pskip=0",
    "no-dct-decimate" => "dct-decimate=0",
    _ => null,   // deadzone-inter/intra, nr take values — left unapplied
};

static bool Which(string exe)
{
    foreach (var d in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator))
    {
        if (string.IsNullOrEmpty(d)) continue;
        if (File.Exists(Path.Combine(d, exe)) || File.Exists(Path.Combine(d, exe + ".exe")))
            return true;
    }
    return false;
}

static (int code, string stderr) Run(string exe, string args, int timeoutSec)
{
    var psi = new ProcessStartInfo(exe, args)
    {
        RedirectStandardError = true,
        RedirectStandardOutput = true,
        UseShellExecute = false,
        CreateNoWindow = true,
    };
    using var p = Process.Start(psi)!;
    var err = new StringBuilder();
    p.ErrorDataReceived += (_, e) => { if (e.Data != null) err.AppendLine(e.Data); };
    p.BeginErrorReadLine();
    p.StandardOutput.ReadToEnd();
    if (!p.WaitForExit(timeoutSec * 1000))
    {
        try { p.Kill(true); } catch { /* best effort */ }
        return (-1, err.ToString());
    }
    p.WaitForExit();
    return (p.ExitCode, err.ToString());
}
