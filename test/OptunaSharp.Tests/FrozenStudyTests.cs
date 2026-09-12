using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Collections;

namespace OptunaSharp.Tests;

/// <summary>
///     Pins <see cref="FrozenStudy" /> equality and ordering to Optuna's
///     <c>optuna/study/_frozen.py</c>: <c>__eq__</c> is value equality over every field
///     (study name, directions, user attrs, system attrs, study id) and <c>__lt__</c> / <c>__le__</c>
///     order by <c>study_id</c>. (Optuna's <c>FrozenStudy</c> is unhashable; the port instead exposes
///     a usable <see cref="FrozenStudy.GetHashCode" /> consistent with equality.)
/// </summary>
[TestClass]
public class FrozenStudyTests
{
    private static FrozenStudy Make(
        int studyId,
        string? name = null,
        OrderedDict<string, object?>? userAttrs = null,
        OrderedDict<string, object?>? systemAttrs = null)
        => new(
            studyName: name ?? $"study-{studyId}",
            direction: StudyDirection.Minimize,
            userAttrs: userAttrs ?? new OrderedDict<string, object?>(),
            systemAttrs: systemAttrs ?? new OrderedDict<string, object?>(),
            studyId: studyId);

    [TestMethod]
    public void Equality_Is_Value_Based_Over_All_Fields()
    {
        var a = Make(4, "s",
            userAttrs: new OrderedDict<string, object?> { ["u"] = 1L },
            systemAttrs: new OrderedDict<string, object?> { ["sys"] = "x" });
        var b = Make(4, "s",
            userAttrs: new OrderedDict<string, object?> { ["u"] = 1L },
            systemAttrs: new OrderedDict<string, object?> { ["sys"] = "x" });

        a.Equals(b).Should().BeTrue();
        (a == b).Should().BeTrue();
        a.GetHashCode().Should().Be(b.GetHashCode());

        // system_attrs is part of FrozenStudy.__dict__, so it participates in equality.
        var differsInSystemAttrs = Make(4, "s",
            userAttrs: new OrderedDict<string, object?> { ["u"] = 1L },
            systemAttrs: new OrderedDict<string, object?> { ["sys"] = "y" });
        a.Equals(differsInSystemAttrs).Should().BeFalse();

        var differsInUserAttrs = Make(4, "s",
            userAttrs: new OrderedDict<string, object?> { ["u"] = 2L },
            systemAttrs: new OrderedDict<string, object?> { ["sys"] = "x" });
        a.Equals(differsInUserAttrs).Should().BeFalse();

        a.Equals(Make(4, "other")).Should().BeFalse();  // differs in study name
        a.Equals(null).Should().BeFalse();
        a.Equals("not a study").Should().BeFalse();
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
        s1.CompareTo(s2).Should().BeNegative();

        var sorted = new List<FrozenStudy> { s3, s1, s2 };
        sorted.Sort();
        sorted.Select(s => s.Direction).Should().HaveCount(3);
        sorted[0].Should().BeSameAs(s1);
        sorted[2].Should().BeSameAs(s3);
    }
}
