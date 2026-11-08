#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Spam filter on a time budget — "you have ten seconds": TPE tunes a multinomial naive-Bayes spam filter
// (smoothing, vocabulary cut-offs, bigrams, lower-casing) with only a wall-clock `timeout`, no trial count,
// and the progress bar on, the way a nightly job is given a fixed slot.
//
// source: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
//         (`timeout` without `n_trials`, `show_progress_bar`)
//         https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering
//
// Real objective, real text classification: a deterministic corpus of 2,000 short messages — ham drawn from
// everyday vocabulary, spam from marketing/phishing vocabulary, with shared words, CamelCase/ALL-CAPS
// variants and misspellings so the knobs matter — tokenized, vectorized and classified by a hand-written
// multinomial naive Bayes. The score is the 4-fold cross-validated F1 of the spam class. How many trials fit
// in ten seconds depends on the machine; the study stops when the budget runs out.
//
//   Run:  dotnet run examples/0_SpamFilterTimeBudget.cs

using System.Diagnostics;
using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // the progress bar replaces the per-trial INFO lines

var corpus = Corpus.Generate(messages: 2000, seed: 29);
var study = Optuna.CreateStudy(sampler: new TPESampler(seed: 30, nStartupTrials: 10), direction: StudyDirection.Maximize, studyName: "spam-nb");
var sw = Stopwatch.StartNew();
study.Optimize(t => Objective(t, corpus), timeout: 10, showProgressBar: true);   // the budget is time, not trials
sw.Stop();

Console.WriteLine();
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0} trials in {1:F1} s; best spam F1 {2:F4} (trial #{3})",
    study.Trials.Count, sw.Elapsed.TotalSeconds, study.BestValue, study.BestTrial.Number));
foreach (var kv in study.BestParams)
    Console.WriteLine($"    {kv.Key,-12} {kv.Value}");

/// <summary>Cross-validates one filter configuration.</summary>
/// <param name="t">The trial proposing smoothing, vocabulary limits and tokenization options.</param>
/// <param name="c">The labelled corpus.</param>
/// <returns>Mean 4-fold F1 of the spam class.</returns>
static double Objective(Trial t, (string Text, bool Spam)[] c)
{
    double alpha = t.SuggestFloat("alpha", 1e-3, 5, log: true);           // Laplace/Lidstone smoothing
    long minDf = t.SuggestInt("min_df", 1, 10);                             // drop words seen in fewer messages
    long maxFeatures = t.SuggestInt("max_features", 50, 5000, log: true);   // keep the most frequent words
    bool bigrams = t.SuggestCategorical("bigrams", new[] { false, true });
    bool lower = t.SuggestCategorical("lowercase", new[] { false, true });

    double f1Sum = 0;
    for (int fold = 0; fold < 4; fold++)
    {
        var train = c.Where((_, i) => i % 4 != fold).ToArray();
        var test = c.Where((_, i) => i % 4 == fold).ToArray();
        var model = NaiveBayes.Fit(train, alpha, (int)minDf, (int)maxFeatures, bigrams, lower);
        int tp = 0, fp = 0, fn = 0;
        foreach (var (text, spam) in test)
        {
            bool pred = model.IsSpam(text);
            if (pred && spam) tp++; else if (pred) fp++; else if (spam) fn++;
        }
        f1Sum += tp == 0 ? 0 : 2.0 * tp / (2 * tp + fp + fn);
    }
    return f1Sum / 4;
}

/// <summary>The synthetic message corpus.</summary>
static class Corpus
{
    /// <summary>Generates labelled messages; the same seed gives the same corpus.</summary>
    /// <param name="messages">Number of messages (about 30 % spam).</param>
    /// <param name="seed">Corpus seed.</param>
    /// <returns>The messages with their labels.</returns>
    public static (string Text, bool Spam)[] Generate(int messages, int seed)
    {
        var rng = new Random(seed);   // corpus generation only — samplers never use System.Random
        string[] ham = "meeting lunch tomorrow project report thanks please review attached call later family weekend dinner schedule update notes team budget plan coffee".Split(' ');
        string[] spam = "free winner prize claim urgent offer click limited cash bonus account verify password deal discount crypto guaranteed act now".Split(' ');
        string[] shared = "you your the today this new please now".Split(' ');
        var corpus = new (string, bool)[messages];
        for (int i = 0; i < messages; i++)
        {
            bool isSpam = rng.NextDouble() < 0.3;
            var words = new List<string>();
            int n = 6 + rng.Next(10);
            for (int k = 0; k < n; k++)
            {
                double u = rng.NextDouble();
                string[] pool = u < 0.35 ? shared : (isSpam ^ (u > 0.9)) ? spam : ham;   // 10 % cross-over words
                string w = pool[rng.Next(pool.Length)];
                if (rng.NextDouble() < 0.15) w = w.ToUpperInvariant();                      // SHOUTING
                else if (rng.NextDouble() < 0.1) w = char.ToUpperInvariant(w[0]) + w[1..];   // sentence case
                if (rng.NextDouble() < 0.05 && w.Length > 3) w = w.Remove(rng.Next(w.Length), 1);   // typo
                words.Add(w);
            }
            corpus[i] = (string.Join(' ', words), isSpam);
        }
        return corpus;
    }
}

/// <summary>A multinomial naive-Bayes text classifier.</summary>
/// <param name="LogPrior">log P(spam) − log P(ham).</param>
/// <param name="LogRatio">Per-token log P(token | spam) − log P(token | ham).</param>
/// <param name="Bigrams">Whether bigram tokens are produced.</param>
/// <param name="Lower">Whether text is lower-cased before tokenizing.</param>
sealed record NaiveBayes(double LogPrior, Dictionary<string, double> LogRatio, bool Bigrams, bool Lower)
{
    /// <summary>Fits the classifier on labelled messages.</summary>
    /// <param name="train">Training messages.</param>
    /// <param name="alpha">Smoothing pseudo-count.</param>
    /// <param name="minDf">Minimum document frequency for a token to enter the vocabulary.</param>
    /// <param name="maxFeatures">Vocabulary size cap (most frequent tokens kept).</param>
    /// <param name="bigrams">Add bigram tokens.</param>
    /// <param name="lower">Lower-case the text first.</param>
    /// <returns>The fitted classifier.</returns>
    public static NaiveBayes Fit((string Text, bool Spam)[] train, double alpha, int minDf, int maxFeatures, bool bigrams, bool lower)
    {
        var df = new Dictionary<string, int>();
        var tokenized = train.Select(m => (Tokens: Tokenize(m.Text, bigrams, lower), m.Spam)).ToArray();
        foreach (var (tokens, _) in tokenized)
            foreach (var tok in tokens.Distinct())
                df[tok] = df.GetValueOrDefault(tok) + 1;
        var vocab = df.Where(kv => kv.Value >= minDf).OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key, StringComparer.Ordinal)
            .Take(maxFeatures).Select(kv => kv.Key).ToHashSet();
        var spamCounts = new Dictionary<string, double>();
        var hamCounts = new Dictionary<string, double>();
        double spamTotal = 0, hamTotal = 0;
        foreach (var (tokens, spam) in tokenized)
            foreach (var tok in tokens.Where(vocab.Contains))
            {
                var counts = spam ? spamCounts : hamCounts;
                counts[tok] = counts.GetValueOrDefault(tok) + 1;
                if (spam) spamTotal++; else hamTotal++;
            }
        var ratio = vocab.ToDictionary(tok => tok, tok =>
            Math.Log((spamCounts.GetValueOrDefault(tok) + alpha) / (spamTotal + alpha * vocab.Count))
            - Math.Log((hamCounts.GetValueOrDefault(tok) + alpha) / (hamTotal + alpha * vocab.Count)));
        double spamDocs = train.Count(m => m.Spam);
        return new NaiveBayes(Math.Log(spamDocs / (train.Length - spamDocs)), ratio, bigrams, lower);
    }

    /// <summary>Classifies one message.</summary>
    /// <param name="text">The message.</param>
    /// <returns><c>true</c> when the spam log-odds are positive.</returns>
    public bool IsSpam(string text) =>
        LogPrior + Tokenize(text, Bigrams, Lower).Sum(tok => LogRatio.GetValueOrDefault(tok)) > 0;

    /// <summary>Splits a message into word (and optionally bigram) tokens.</summary>
    /// <param name="text">The message.</param>
    /// <param name="bigrams">Add "w1 w2" tokens for adjacent words.</param>
    /// <param name="lower">Lower-case first.</param>
    /// <returns>The tokens, in order.</returns>
    private static List<string> Tokenize(string text, bool bigrams, bool lower)
    {
        var words = (lower ? text.ToLowerInvariant() : text).Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var tokens = new List<string>(words);
        if (bigrams)
            for (int i = 0; i + 1 < words.Length; i++)
                tokens.Add(words[i] + " " + words[i + 1]);
        return tokens;
    }
}
