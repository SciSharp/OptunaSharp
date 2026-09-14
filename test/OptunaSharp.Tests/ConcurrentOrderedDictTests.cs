using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AwesomeAssertions;
using OptunaSharp.Collections;
using Clone = OptunaSharp.Collections.Concurrent;

namespace OptunaSharp.Tests;

/// <summary>
///     Pins the contract of <see cref="ConcurrentOrderedDict{TKey,TValue}" /> — the thread-safe, two-path
///     (key + insertion-order index) collection built on the vendored concurrent dictionary. These tests are the
///     guard rails for the load-bearing invariants: that the two paths observe one value, that enumeration is
///     value-order like a list, that keys stay unique, that removal preserves order and re-indexes, and that
///     concurrent use neither corrupts nor throws.
/// </summary>
[TestClass]
public class ConcurrentOrderedDictTests
{
    /// <summary>A 16-byte value type that is not written atomically, so value replacement takes the clone-and-publish path.</summary>
    private struct Pair16
    {
        /// <summary>First 8 bytes.</summary>
        public long A;

        /// <summary>Second 8 bytes.</summary>
        public long B;
    }

    /// <summary>The key indexer and the index indexer must read the same value for the same entry.</summary>
    [TestMethod]
    public void TwoPaths_KeyAndIndex_SeeSameValues()
    {
        var d = new ConcurrentOrderedDict<string, int>();
        d.Add("x", 1);
        d.Add("y", 2);

        d["x"].Should().Be(1);
        d[0].Should().Be(1);
        d["y"].Should().Be(2);
        d[1].Should().Be(2);
    }

    /// <summary>Default enumeration, <c>ToArray</c> and <c>Values</c> yield the values in insertion order (list-shaped).</summary>
    [TestMethod]
    public void Enumeration_ToArray_Values_AreValuesInInsertionOrder()
    {
        var d = new ConcurrentOrderedDict<string, int>();
        d.Add("a", 10);
        d.Add("b", 20);
        d.Add("c", 30);

        d.Should().Equal(10, 20, 30);
        d.ToArray().Should().Equal(10, 20, 30);
        d.Values.Should().Equal(10, 20, 30);
        d.Keys.Should().Equal("a", "b", "c");
        d.Pairs.Select(p => (p.Key, p.Value)).Should().Equal(("a", 10), ("b", 20), ("c", 30));
    }

    /// <summary>Uniqueness is enforced by key: a second add of the same key fails (<c>TryAdd</c>) or throws (<c>Add</c>).</summary>
    [TestMethod]
    public void Dedup_TryAddReturnsFalse_AddThrows()
    {
        var d = new ConcurrentOrderedDict<string, int>();
        d.TryAdd("k", 1).Should().BeTrue();
        d.TryAdd("k", 2).Should().BeFalse();
        d.Count.Should().Be(1);
        d["k"].Should().Be(1);

        Action act = () => d.Add("k", 3);
        act.Should().Throw<ArgumentException>();
    }

    /// <summary>The pair constructor mirrors Python <c>dict(pairs)</c>: duplicate keys keep the first position but take the last value.</summary>
    [TestMethod]
    public void SourceCtor_LastValueWins_FirstPosition()
    {
        var src = new[]
        {
            new KeyValuePair<string, int>("a", 1),
            new KeyValuePair<string, int>("b", 2),
            new KeyValuePair<string, int>("a", 9),
        };
        var d = new ConcurrentOrderedDict<string, int>(src);

        d.Keys.Should().Equal("a", "b");
        d["a"].Should().Be(9);
        d[0].Should().Be(9);
    }

    /// <summary>Setting an existing key updates the value in place on both paths and never moves the key.</summary>
    [TestMethod]
    public void KeySet_UpdatesInPlace_KeepsPosition_OnBothPaths()
    {
        var d = new ConcurrentOrderedDict<string, int>();
        d.Add("x", 1);
        d.Add("y", 2);

        d["x"] = 11;

        d.Keys.Should().Equal("x", "y");
        d[0].Should().Be(11);
        d["x"].Should().Be(11);
    }

    /// <summary>The index setter replaces the value at a position (keeping its key) and is seen by the key path.</summary>
    [TestMethod]
    public void IndexSet_ReplacesValue_KeepsKey_OnBothPaths()
    {
        var d = new ConcurrentOrderedDict<string, int>();
        d.Add("x", 1);
        d.Add("y", 2);

        d[1] = 22;

        d["y"].Should().Be(22);
        d.GetKeyAt(1).Should().Be("y");
        d.Keys.Should().Equal("x", "y");
    }

    /// <summary>The key-to-index mapping is exact for present keys and reports -1 (or false) for absent ones.</summary>
    [TestMethod]
    public void IndexOf_TryGetIndex_GetKeyAt()
    {
        var d = new ConcurrentOrderedDict<string, int>();
        d.Add("a", 1);
        d.Add("b", 2);
        d.Add("c", 3);

        d.IndexOf("b").Should().Be(1);
        d.IndexOf("absent").Should().Be(-1);
        d.TryGetIndex("c", out var ci).Should().BeTrue();
        ci.Should().Be(2);
        d.TryGetIndex("absent", out var ai).Should().BeFalse();
        ai.Should().Be(-1);
        d.GetKeyAt(0).Should().Be("a");
    }

    /// <summary>Removing by key preserves order, shrinks the count, and re-indexes the entries after the hole.</summary>
    [TestMethod]
    public void Remove_ByKey_ShiftsAndReindexes()
    {
        var d = new ConcurrentOrderedDict<string, int>();
        d.Add("a", 1);
        d.Add("b", 2);
        d.Add("c", 3);

        d.TryRemove("b", out var removed).Should().BeTrue();
        removed.Should().Be(2);

        d.Count.Should().Be(2);
        d.Keys.Should().Equal("a", "c");
        d.Should().Equal(1, 3);
        d[1].Should().Be(3);
        d.IndexOf("c").Should().Be(1);
    }

    /// <summary>Removing by position behaves like <see cref="System.Collections.Generic.List{T}.RemoveAt" />: it shifts and re-indexes.</summary>
    [TestMethod]
    public void RemoveAt_ShiftsAndReindexes()
    {
        var d = new ConcurrentOrderedDict<string, int>();
        d.Add("a", 1);
        d.Add("b", 2);
        d.Add("c", 3);

        d.RemoveAt(0);

        d.Keys.Should().Equal("b", "c");
        d.Should().Equal(2, 3);
        d.IndexOf("b").Should().Be(0);
        d.IndexOf("c").Should().Be(1);
    }

    /// <summary>With an <c>int</c> key the bracket resolves to the index path; key access is available through the methods.</summary>
    [TestMethod]
    public void IntKey_BracketIsIndex_MethodsForKey()
    {
        var d = new ConcurrentOrderedDict<int, string>();
        d.Add(5, "five");
        d.Add(9, "nine");

        d[0].Should().Be("five");      // index path
        d[1].Should().Be("nine");
        d.GetByKey(9).Should().Be("nine");
        d.IndexOf(9).Should().Be(1);

        d.SetByKey(5, "FIVE");
        d.GetByKey(5).Should().Be("FIVE");
        d[0].Should().Be("FIVE");
    }

    /// <summary><c>GetOrAdd</c>, <c>AddOrUpdate</c> and <c>TryUpdate</c> match the concurrent-dictionary semantics while preserving order.</summary>
    [TestMethod]
    public void GetOrAdd_AddOrUpdate_TryUpdate()
    {
        var d = new ConcurrentOrderedDict<string, int>();

        d.GetOrAdd("a", 1).Should().Be(1);
        d.GetOrAdd("a", 99).Should().Be(1);
        d.GetOrAdd("b", _ => 2).Should().Be(2);

        d.AddOrUpdate("a", 0, (_, v) => v + 10).Should().Be(11);
        d.AddOrUpdate("c", 3, (_, v) => v).Should().Be(3);

        d.TryUpdate("a", 100, 11).Should().BeTrue();
        d["a"].Should().Be(100);
        d.TryUpdate("a", 0, 11).Should().BeFalse();

        d.Keys.Should().Equal("a", "b", "c");
    }

    /// <summary><c>Clear</c> empties both paths; <c>Count</c>/<c>IsEmpty</c> track it.</summary>
    [TestMethod]
    public void Clear_Count_IsEmpty()
    {
        var d = new ConcurrentOrderedDict<string, int>();
        d.IsEmpty.Should().BeTrue();
        d.Add("a", 1);
        d.Add("b", 2);
        d.Count.Should().Be(2);
        d.IsEmpty.Should().BeFalse();

        d.Clear();

        d.Count.Should().Be(0);
        d.IsEmpty.Should().BeTrue();
        d.ToArray().Should().BeEmpty();
        d.Keys.Should().BeEmpty();
    }

    /// <summary>A supplied comparer governs key equality and hashing (here, case-insensitive keys collapse).</summary>
    [TestMethod]
    public void CustomComparer_IsHonored()
    {
        var d = new ConcurrentOrderedDict<string, int>(StringComparer.OrdinalIgnoreCase);
        d.Add("Key", 1);

        d.ContainsKey("KEY").Should().BeTrue();
        d.TryAdd("key", 2).Should().BeFalse();
        d["KEY"] = 5;
        d["key"].Should().Be(5);
        d.Count.Should().Be(1);
        d.Comparer.Should().BeSameAs(StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>A null key is rejected on every key-taking member.</summary>
    [TestMethod]
    public void NullKey_Throws()
    {
        var d = new ConcurrentOrderedDict<string, int>();
        ((Action)(() => d.Add(null!, 1))).Should().Throw<ArgumentNullException>();
        ((Action)(() => d.TryAdd(null!, 1))).Should().Throw<ArgumentNullException>();
        ((Action)(() => _ = d[null!])).Should().Throw<ArgumentNullException>();
        ((Action)(() => d[null!] = 1)).Should().Throw<ArgumentNullException>();
        ((Action)(() => d.IndexOf(null!))).Should().Throw<ArgumentNullException>();
    }

    /// <summary>Positional members reject out-of-range indices.</summary>
    [TestMethod]
    public void IndexOutOfRange_Throws()
    {
        var d = new ConcurrentOrderedDict<string, int>();
        d.Add("a", 1);

        ((Action)(() => _ = d[1])).Should().Throw<ArgumentOutOfRangeException>();
        ((Action)(() => _ = d[-1])).Should().Throw<ArgumentOutOfRangeException>();
        ((Action)(() => d[1] = 0)).Should().Throw<ArgumentOutOfRangeException>();
        ((Action)(() => d.RemoveAt(5))).Should().Throw<ArgumentOutOfRangeException>();
        ((Action)(() => d.GetKeyAt(5))).Should().Throw<ArgumentOutOfRangeException>();

        d.TryGetAt(5, out _).Should().BeFalse();
        d.TryGetAt(0, out var v0).Should().BeTrue();
        v0.Should().Be(1);
    }

    /// <summary>A value type that cannot be written atomically still replaces cleanly (exercises the clone-and-publish path).</summary>
    [TestMethod]
    public void NonAtomicValueType_ReplacementWorks()
    {
        var d = new ConcurrentOrderedDict<string, Pair16>();
        d.Add("a", new Pair16 { A = 1, B = 2 });
        d.Add("b", new Pair16 { A = 3, B = 4 });

        d["a"] = new Pair16 { A = 10, B = 20 };   // key-path replace (non-atomic clone path)
        d[1] = new Pair16 { A = 30, B = 40 };     // index-path replace (non-atomic clone path)

        d["a"].A.Should().Be(10);
        d["a"].B.Should().Be(20);
        d[0].A.Should().Be(10);
        d["b"].A.Should().Be(30);
        d[1].B.Should().Be(40);
        d.ToArray().Select(p => p.A).Should().Equal(10, 30);
    }

    /// <summary>An enumerator observes the snapshot present when it began; later additions don't retroactively appear.</summary>
    [TestMethod]
    public void Enumerator_IsSnapshot()
    {
        var d = new ConcurrentOrderedDict<int, int>();
        for (int i = 0; i < 3; i++)
        {
            d.Add(i, i);
        }

        var seen = new List<int>();
        foreach (var v in d)
        {
            seen.Add(v);
            if (v == 0)
            {
                d.Add(99, 99); // must not appear in this enumeration
            }
        }

        seen.Should().Equal(0, 1, 2);
        d.Count.Should().Be(4);
    }

    /// <summary>Under heavy parallel insertion every key lands exactly once and the values form the expected set.</summary>
    [TestMethod]
    public void Concurrent_Adds_AllPresent_And_Bijection()
    {
        const int n = 50_000;
        var d = new ConcurrentOrderedDict<int, int>();

        Parallel.For(0, n, i => d.TryAdd(i, i * 2).Should().BeTrue());

        d.Count.Should().Be(n);
        for (int i = 0; i < n; i++)
        {
            d.TryGetValue(i, out var v).Should().BeTrue();
            v.Should().Be(i * 2);
        }

        var values = d.ToArray();
        values.Length.Should().Be(n);
        values.Distinct().Count().Should().Be(n);
        values.All(v => v % 2 == 0).Should().BeTrue();
    }

    /// <summary>Reads never throw or corrupt while a writer mutates concurrently; the final state is exactly consistent.</summary>
    [TestMethod]
    public void Concurrent_ReadDuringWrite_NoThrow()
    {
        var d = new ConcurrentOrderedDict<int, int>();
        for (int i = 0; i < 1_000; i++)
        {
            d.Add(i, i);
        }

        long sink = 0;
        var writer = Task.Run(() =>
        {
            for (int i = 1_000; i < 5_000; i++)
            {
                d.TryAdd(i, i);
            }
        });

        for (int round = 0; round < 200; round++)
        {
            foreach (var v in d)
            {
                sink += v;
            }

            _ = d.ToArray();
            _ = d.Count;
        }

        writer.Wait();

        d.Count.Should().Be(5_000);
        for (int i = 0; i < 5_000; i++)
        {
            d.ContainsKey(i).Should().BeTrue();
        }

        sink.Should().BeGreaterThanOrEqualTo(0);
    }
}

/// <summary>
///     A focused smoke suite for the vendored <see cref="Clone.ConcurrentDictionary{TKey,TValue}" /> clone itself,
///     independent of the ordered wrapper, so a regression in the faithful port is caught directly rather than only
///     through its consumer.
/// </summary>
[TestClass]
public class VendoredConcurrentDictionaryTests
{
    /// <summary>The core key API (add/dedup/get/update/remove) behaves like the framework dictionary.</summary>
    [TestMethod]
    public void CoreKeyApi_Works()
    {
        var cd = new Clone.ConcurrentDictionary<string, int>();
        cd.TryAdd("a", 1).Should().BeTrue();
        cd.TryAdd("a", 2).Should().BeFalse();
        cd["b"] = 2;
        cd["a"] = 10;
        cd["a"].Should().Be(10);
        cd.Count.Should().Be(2);
        cd.ContainsKey("b").Should().BeTrue();

        cd.GetOrAdd("c", 3).Should().Be(3);
        cd.GetOrAdd("c", 99).Should().Be(3);
        cd.AddOrUpdate("a", 0, (_, v) => v + 5).Should().Be(15);

        cd.TryRemove("b", out var bv).Should().BeTrue();
        bv.Should().Be(2);
        cd.ContainsKey("b").Should().BeFalse();
    }

    /// <summary>The striped-lock write path stays correct under heavy parallel insertion (values survive resizes).</summary>
    [TestMethod]
    public void ConcurrentAdds_SurviveResizes()
    {
        const int n = 100_000;
        var cd = new Clone.ConcurrentDictionary<int, int>();

        Parallel.For(0, n, i => cd[i] = i * i);

        cd.Count.Should().Be(n);
        for (int i = 0; i < n; i++)
        {
            cd.TryGetValue(i, out var v).Should().BeTrue();
            v.Should().Be(i * i);
        }
    }
}
