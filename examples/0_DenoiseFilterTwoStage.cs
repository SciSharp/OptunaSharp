#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Image-denoising pipeline, tuned in two stages — stage 1 explores every knob of a median-prefilter +
// bilateral-filter pipeline with TPE; stage 2 freezes the structural choices (prefilter, window radius,
// passes) at the stage-1 winner with PartialFixedSampler and spends the remaining budget refining only the
// two continuous sigmas, in the same study.
//
// source: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.PartialFixedSampler.html
//         (fix some parameters, keep sampling the rest; swapped in with `study.sampler = ...`)
//         https://en.wikipedia.org/wiki/Bilateral_filter
//
// Real objective: a deterministic 96×96 grayscale test card (disc, bars, a smooth gradient, fine stripes)
// corrupted by Gaussian sensor noise (σ = 0.08) and 1 % salt-and-pepper defects; the trial's pipeline
// restores it and the score is the PSNR against the clean card (higher is better). Salt-and-pepper is why a
// median prefilter can matter, and the radius/passes trade detail for smoothing — the structure the first
// stage settles, after which the sigmas are what is left to polish.
//
//   Run:  dotnet run examples/0_DenoiseFilterTwoStage.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var (clean, noisy) = TestCard.Make(size: 96, seed: 31);
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "noisy input PSNR: {0:F2} dB", Image.Psnr(clean, noisy)));

var study = Optuna.CreateStudy(sampler: new TPESampler(seed: 2, nStartupTrials: 15), direction: StudyDirection.Maximize, studyName: "denoise");

// stage 1: everything is open.
study.Optimize(t => Objective(t, clean, noisy), nTrials: 50);
var stage1 = study.BestTrial;
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "stage 1 best: {0:F2} dB  {1}", stage1.Value, Describe(stage1)));

// stage 2: freeze the structural decisions, keep tuning the sigmas with the same TPE underneath.
var frozen = new Dictionary<string, object?>
{
    ["prefilter"] = stage1.Params["prefilter"],
    ["radius"] = stage1.Params["radius"],
    ["passes"] = stage1.Params["passes"],
};
study.Sampler = new PartialFixedSampler(frozen, study.Sampler);
study.Optimize(t => Objective(t, clean, noisy), nTrials: 30);
var stage2 = study.BestTrial;
bool structureHeld = study.Trials.Skip(50).All(t => frozen.All(kv => Equals(t.Params[kv.Key], kv.Value)));
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "stage 2 best: {0:F2} dB  {1}", stage2.Value, Describe(stage2)));
Console.WriteLine($"every stage-2 trial kept the frozen structure: {structureHeld}");

/// <summary>Restores the noisy card with the trial's pipeline and scores it.</summary>
/// <param name="t">The trial proposing the pipeline.</param>
/// <param name="clean">The ground-truth image.</param>
/// <param name="noisy">The corrupted image.</param>
/// <returns>PSNR in dB of the restored image against the clean one.</returns>
static double Objective(Trial t, double[,] clean, double[,] noisy)
{
    string prefilter = t.SuggestCategorical("prefilter", new[] { "none", "median3" });
    int radius = (int)t.SuggestInt("radius", 1, 6);                       // window half-width in pixels
    int passes = (int)t.SuggestInt("passes", 1, 3);                       // repeated filtering
    double sigmaS = t.SuggestFloat("sigma_space", 0.5, 6.0, log: true);   // spatial falloff (pixels)
    double sigmaR = t.SuggestFloat("sigma_range", 0.01, 0.5, log: true);  // intensity falloff (edge keeping)
    var img = prefilter == "median3" ? Image.Median3(noisy) : noisy;
    for (int p = 0; p < passes; p++)
        img = Image.Bilateral(img, radius, sigmaS, sigmaR);
    return Image.Psnr(clean, img);
}

/// <summary>Formats a trial's pipeline.</summary>
/// <param name="t">A finished trial.</param>
/// <returns>A one-line description.</returns>
static string Describe(FrozenTrial t) => string.Format(CultureInfo.InvariantCulture,
    "prefilter={0} radius={1} passes={2} σs={3:F2} σr={4:F3}",
    t.Params["prefilter"], t.Params["radius"], t.Params["passes"], t.Params["sigma_space"], t.Params["sigma_range"]);

/// <summary>The synthetic test card and its corruption.</summary>
static class TestCard
{
    /// <summary>Draws the card and a noisy copy; the same seed gives the same noise.</summary>
    /// <param name="size">Width and height in pixels.</param>
    /// <param name="seed">Noise seed.</param>
    /// <returns>The clean and the noisy image, intensities in [0, 1].</returns>
    public static (double[,] Clean, double[,] Noisy) Make(int size, int seed)
    {
        var clean = new double[size, size];
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                double v = 0.15 + 0.5 * x / size;                                   // smooth gradient
                if (Math.Pow(x - 30, 2) + Math.Pow(y - 30, 2) < 18 * 18) v = 0.9;    // bright disc
                if (x > 55 && x < 85 && y > 15 && y < 45) v = ((x / 6) % 2 == 0) ? 0.1 : 0.8;   // bars
                if (y > 60 && x > 10 && x < 86) v = ((x / 2) % 2 == 0) ? 0.35 : 0.65;          // fine stripes
                clean[y, x] = v;
            }
        var rng = new Random(seed);   // noise generation only — samplers never use System.Random
        var noisy = new double[size, size];
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++)
            {
                double g = Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());
                double v = clean[y, x] + 0.08 * g;
                double u = rng.NextDouble();
                if (u < 0.005) v = 0; else if (u < 0.01) v = 1;                  // salt-and-pepper defects
                noisy[y, x] = Math.Clamp(v, 0, 1);
            }
        return (clean, noisy);
    }
}

/// <summary>The image operations of the pipeline.</summary>
static class Image
{
    /// <summary>Peak signal-to-noise ratio for intensities in [0, 1].</summary>
    /// <param name="a">Reference image.</param>
    /// <param name="b">Test image of the same size.</param>
    /// <returns>PSNR in dB.</returns>
    public static double Psnr(double[,] a, double[,] b)
    {
        double mse = 0;
        foreach (var (y, x) in Pixels(a))
            mse += Math.Pow(a[y, x] - b[y, x], 2);
        mse /= a.Length;
        return 10 * Math.Log10(1 / mse);
    }

    /// <summary>3×3 median filter (edges replicate).</summary>
    /// <param name="img">Input image.</param>
    /// <returns>The filtered image.</returns>
    public static double[,] Median3(double[,] img)
    {
        int h = img.GetLength(0), w = img.GetLength(1);
        var o = new double[h, w];
        var win = new double[9];
        foreach (var (y, x) in Pixels(img))
        {
            int k = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                    win[k++] = img[Math.Clamp(y + dy, 0, h - 1), Math.Clamp(x + dx, 0, w - 1)];
            Array.Sort(win);
            o[y, x] = win[4];
        }
        return o;
    }

    /// <summary>Bilateral filter: a spatial Gaussian times an intensity Gaussian, normalized.</summary>
    /// <param name="img">Input image.</param>
    /// <param name="radius">Window half-width.</param>
    /// <param name="sigmaS">Spatial standard deviation in pixels.</param>
    /// <param name="sigmaR">Intensity standard deviation.</param>
    /// <returns>The filtered image.</returns>
    public static double[,] Bilateral(double[,] img, int radius, double sigmaS, double sigmaR)
    {
        int h = img.GetLength(0), w = img.GetLength(1);
        var o = new double[h, w];
        var spatial = new double[2 * radius + 1, 2 * radius + 1];
        for (int dy = -radius; dy <= radius; dy++)
            for (int dx = -radius; dx <= radius; dx++)
                spatial[dy + radius, dx + radius] = Math.Exp(-(dx * dx + dy * dy) / (2 * sigmaS * sigmaS));
        double inv2r = 1 / (2 * sigmaR * sigmaR);
        foreach (var (y, x) in Pixels(img))
        {
            double c = img[y, x], num = 0, den = 0;
            for (int dy = -radius; dy <= radius; dy++)
            {
                int yy = Math.Clamp(y + dy, 0, h - 1);
                for (int dx = -radius; dx <= radius; dx++)
                {
                    double v = img[yy, Math.Clamp(x + dx, 0, w - 1)];
                    double wgt = spatial[dy + radius, dx + radius] * Math.Exp(-(v - c) * (v - c) * inv2r);
                    num += wgt * v;
                    den += wgt;
                }
            }
            o[y, x] = num / den;
        }
        return o;
    }

    /// <summary>Enumerates every pixel coordinate in row-major order.</summary>
    /// <param name="img">The image.</param>
    /// <returns>(row, column) pairs.</returns>
    private static IEnumerable<(int Y, int X)> Pixels(double[,] img)
    {
        for (int y = 0; y < img.GetLength(0); y++)
            for (int x = 0; x < img.GetLength(1); x++)
                yield return (y, x);
    }
}
