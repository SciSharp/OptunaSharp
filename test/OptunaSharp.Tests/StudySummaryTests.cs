using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;

namespace OptunaSharp.Tests;

/// <summary>
///     Pins <see cref="StudySummary" /> equality and ordering to Optuna's
///     <c>optuna/study/_study_summary.py</c>: <c>__eq__</c> is value equality over every field
///     (<c>other.__dict__ == self.__dict__</c>), and <c>__lt__</c> / <c>__le__</c> order by
///     <c>study_id</c>. (Optuna's summary is unhashable; the port instead exposes a usable
///     <see cref="StudySummary.GetHashCode" /> consistent with equality, matching the FrozenTrial port.)
/// </summary>
[TestClass]
public class StudySummaryTests
{
    private static StudySummary Make(
        int studyId,
        string? name = null,
        int nTrials = 0,
        FrozenTrial? bestTrial = null,
        OrderedDict<string, object?>? userAttrs = null)
        => new(
            studyName: name ?? $"study-{studyId}",
            direction: StudyDirection.Minimize,
            bestTrial: bestTrial,
            userAttrs: userAttrs ?? new OrderedDict<string, object?>(),
            nTrials: nTrials,
            datetimeStart: null,
            studyId: studyId);

    [TestMethod]
    public void Equality_Is_Value_Based_Over_All_Fields()
    {
        var best = FrozenTrial.CreateTrial(value: 1.0);
        var a = Make(7, name: "s", nTrials: 3, bestTrial: best,
            userAttrs: new OrderedDict<string, object?> { ["k"] = 1L });
        // Same values, independent dictionary instance (Python compares dicts by value).
        var b = Make(7, name: "s", nTrials: 3, bestTrial: best,
            userAttrs: new OrderedDict<string, object?> { ["k"] = 1L });

        a.Equals(b).Should().BeTrue();
        (a == b).Should().BeTrue();
        (a != b).Should().BeFalse();
        a.GetHashCode().Should().Be(b.GetHashCode());

        // Differ in a single field -> not equal (Optuna's __eq__ compares every field).
        var differsInCount = Make(7, name: "s", nTrials: 4, bestTrial: best,
            userAttrs: new OrderedDict<string, object?> { ["k"] = 1L });
        a.Equals(differsInCount).Should().BeFalse();

        var differsInAttrs = Make(7, name: "s", nTrials: 3, bestTrial: best,
            userAttrs: new OrderedDict<string, object?> { ["k"] = 2L });
        a.Equals(differsInAttrs).Should().BeFalse();

        // best_trial present vs absent is a difference.
        Make(7, name: "s", nTrials: 3, bestTrial: null,
            userAttrs: new OrderedDict<string, object?> { ["k"] = 1L })
            .Equals(a).Should().BeFalse();
    }

    [TestMethod]
    public void Equality_Against_Null_And_Other_Types_Is_False()
    {
        var a = Make(1);
        a.Equals(null).Should().BeFalse();
        a.Equals("not a summary").Should().BeFalse();
        (a == null).Should().BeFalse();
        (null == a).Should().BeFalse();
    }

    [TestMethod]
    public void Ordering_Is_By_StudyId()
    {
        var s1 = Make(1);
        var s2 = Make(2);
        var s3 = Make(3);

        (s1 < s2).Should().BeTrue();
        (s3 > s2).Should().BeTrue();
        (s1 <= Make(1)).Should().BeTrue();
        (s2 >= Make(2)).Should().BeTrue();
        s1.CompareTo(s2).Should().BeNegative();
        s3.CompareTo(s2).Should().BePositive();

        var sorted = new List<StudySummary> { s3, s1, s2 };
        sorted.Sort();
        sorted.Select(s => s.StudyId).Should().Equal(1, 2, 3);
    }

    [TestMethod]
    public void CompareTo_And_Equals_Disagree_On_StudyId_Tie_Like_Optuna()
    {
        // Optuna orders by study_id but tests equality over all fields, so a study_id tie with a
        // differing field is CompareTo == 0 yet not Equals.
        var a = Make(5, name: "a", nTrials: 1);
        var b = Make(5, name: "b", nTrials: 2);

        a.CompareTo(b).Should().Be(0);
        a.Equals(b).Should().BeFalse();
    }
}
