using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using OptunaSharp.Internal;

namespace OptunaSharp.Cli;

/// <summary>
///     PyYAML 6's <c>yaml.safe_dump(value)</c> for the value trees the CLI prints with <c>-f yaml</c>: mappings, sequences
///     and the scalars Optuna's <c>_format_value</c> leaves (<c>None</c>, bool, int, float, str). It ports the parts of
///     PyYAML's <c>SafeRepresenter</c>, <c>Serializer</c>, <c>Resolver</c> and <c>Emitter</c> those trees reach, so the
///     output is PyYAML's byte for byte, and a YAML 1.1 reader reads every value back with its type (QA-61).
/// </summary>
/// <remarks>
///     <para>
///         <b>Why not YamlDotNet.</b> The CLI used YamlDotNet with YAML 1.1 quoting, which wrote <c>2.0</c> as <c>2</c>,
///         <c>1e-07</c> as <c>1E-07</c> (a string to PyYAML: YAML 1.1 floats need a decimal point), <c>1e16</c> as
///         <c>10000000000000000</c> and the formatted datetimes bare (datetimes to PyYAML, strings in Optuna's output),
///         and it quoted keys such as <c>y</c> that PyYAML's resolver leaves plain. Those are type changes, not only byte
///         differences, for anyone parsing the CLI's YAML.
///     </para>
///     <para>
///         <b>What is ported.</b> <c>safe_dump</c>'s defaults: block style for non-empty collections, flow <c>[]</c> /
///         <c>{}</c> for empty ones, sorted mapping keys, indent 2 with indentless sequences under mapping keys, width 80
///         (long plain, single- and double-quoted scalars fold), <c>allow_unicode=False</c> (non-ASCII text is
///         double-quoted with <c>\x</c>, <c>\u</c>, <c>\U</c> escapes), and YAML 1.1's implicit resolvers, which decide
///         whether a string can stay plain or must be quoted to keep it a string. Tags, anchors, aliases, explicit
///         documents and the literal / folded block styles are never produced by <c>safe_dump</c> for these trees and are
///         not ported.
///     </para>
///     <para>
///         <b>Footgun.</b> A value of any other type raises, as PyYAML's <c>RepresenterError</c> does: callers format
///         datetimes and other objects to strings first, exactly as Optuna's <c>_format_value</c> does.
///     </para>
/// </remarks>
internal static class PyYamlSafeDump
{
    /// <summary>
    ///     Dumps <paramref name="value" /> as <c>yaml.safe_dump(value)</c> does, including the trailing line break
    ///     (Optuna strips it).
    /// </summary>
    /// <param name="value">
    ///     A tree of <see cref="IDictionary" /> / <see cref="OrderedDictionary" /> mappings, non-string
    ///     <see cref="IEnumerable" /> sequences (lists and tuples alike, as PyYAML represents both as sequences), and
    ///     <c>null</c>, <see cref="bool" />, integral, <see cref="double" /> / <see cref="float" /> or <see cref="string" />
    ///     scalars.
    /// </param>
    /// <returns>PyYAML's output text.</returns>
    /// <exception cref="ArgumentException">A value of an unsupported type (PyYAML's <c>cannot represent an object</c>).</exception>
    /// <exception cref="NumSharp.TypeError">A mapping mixes string and numeric keys, which Python's <c>sorted</c> refuses.</exception>
    internal static string Dump(object? value)
    {
        var events = new List<Event>();
        Represent(value, events);
        var emitter = new Emitter(events);
        return emitter.Run();
    }

    // ---------------------------------------------------------------- representer + serializer

    /// <summary>The node events PyYAML's serializer hands its emitter (only the kinds safe_dump produces here).</summary>
    private abstract record Event;

    /// <summary>
    ///     A scalar node: its text, its tag in PyYAML's short form, and PyYAML's <c>implicit</c> pair.
    ///     <c>PlainImplicit</c> says the text resolves to its own tag when written plain (a string that would not read
    ///     back as something else); <c>QuotedImplicit</c> says it needs no tag when quoted (true for strings only).
    /// </summary>
    /// <param name="Value">The scalar's text as PyYAML's representer writes it.</param>
    /// <param name="Tag">
    ///     The prepared tag (<c>!!str</c>, <c>!!int</c>, <c>!!float</c>, <c>!!bool</c>, <c>!!null</c>). It is never written
    ///     for these scalars, but <c>check_simple_key</c> counts its length toward the 128-character limit, so a string
    ///     key of 123 characters or more is already a complex <c>?</c> key.
    /// </param>
    /// <param name="PlainImplicit">PyYAML's <c>implicit[0]</c>.</param>
    /// <param name="QuotedImplicit">PyYAML's <c>implicit[1]</c>.</param>
    private sealed record ScalarEvent(string Value, string Tag, bool PlainImplicit, bool QuotedImplicit) : Event;

    /// <summary>The start of a sequence node.</summary>
    private sealed record SequenceStartEvent : Event;

    /// <summary>The end of a sequence node.</summary>
    private sealed record SequenceEndEvent : Event;

    /// <summary>The start of a mapping node.</summary>
    private sealed record MappingStartEvent : Event;

    /// <summary>The end of a mapping node.</summary>
    private sealed record MappingEndEvent : Event;

    /// <summary>
    ///     PyYAML's <c>SafeRepresenter</c> + <c>Serializer</c> for one value: appends the node events of
    ///     <paramref name="value" /> to <paramref name="events" />.
    /// </summary>
    /// <param name="value">The value to represent (see <see cref="Dump" />).</param>
    /// <param name="events">The event list being built.</param>
    /// <exception cref="ArgumentException">A value of an unsupported type.</exception>
    /// <exception cref="NumSharp.TypeError">A mapping mixes string and numeric keys.</exception>
    private static void Represent(object? value, List<Event> events)
    {
        switch (value)
        {
            case null:
                events.Add(new ScalarEvent("null", "!!null", true, false));
                return;
            case bool b:
                events.Add(new ScalarEvent(b ? "true" : "false", "!!bool", true, false));
                return;
            case string s:
                // A string is written plain only if PyYAML's implicit resolvers would read it back as a string.
                events.Add(new ScalarEvent(s, "!!str", ResolvesToString(s), true));
                return;
            case double d:
                events.Add(new ScalarEvent(RepresentFloat(d), "!!float", true, false));
                return;
            case float f:
                events.Add(new ScalarEvent(RepresentFloat(f), "!!float", true, false));
                return;
            case sbyte or byte or short or ushort or int or uint or long or ulong or System.Numerics.BigInteger:
                events.Add(new ScalarEvent(Convert.ToString(value, CultureInfo.InvariantCulture)!, "!!int", true, false));
                return;
            case OrderedDictionary od:
                RepresentMapping(od.Cast<DictionaryEntry>().Select(e => (e.Key, e.Value)), events);
                return;
            case IDictionary dict:
                RepresentMapping(dict.Cast<DictionaryEntry>().Select(e => (e.Key, e.Value)), events);
                return;
            case IEnumerable seq:
                events.Add(new SequenceStartEvent());
                foreach (object? item in seq)
                    Represent(item, events);
                events.Add(new SequenceEndEvent());
                return;
            default:
                throw new ArgumentException($"cannot represent an object: {value}");
        }
    }

    /// <summary>
    ///     <c>represent_mapping</c> with <c>sort_keys=True</c>: the items in Python's <c>sorted(mapping.items())</c> order,
    ///     strings by code point and numbers by value.
    /// </summary>
    /// <param name="items">The mapping's items.</param>
    /// <param name="events">The event list being built.</param>
    /// <exception cref="ArgumentException">A key or value of an unsupported type.</exception>
    /// <exception cref="NumSharp.TypeError">String and numeric keys are mixed.</exception>
    private static void RepresentMapping(IEnumerable<(object Key, object? Value)> items, List<Event> events)
    {
        var list = items.ToList();
        list.Sort((a, b) => ComparePythonKeys(a.Key, b.Key));
        events.Add(new MappingStartEvent());
        foreach (var (key, val) in list)
        {
            Represent(key, events);
            Represent(val, events);
        }
        events.Add(new MappingEndEvent());
    }

    /// <summary>
    ///     Python's <c>&lt;</c> on two mapping keys: strings compare by Unicode code point (not UTF-16 unit, which orders
    ///     astral characters differently), numbers by value; a string against a number raises, as <c>sorted</c> does.
    /// </summary>
    /// <param name="a">The first key.</param>
    /// <param name="b">The second key.</param>
    /// <returns>Negative, zero or positive, as <see cref="IComparer{T}.Compare" />.</returns>
    /// <exception cref="NumSharp.TypeError">One key is a string and the other is not.</exception>
    private static int ComparePythonKeys(object a, object b)
    {
        if (a is string sa && b is string sb)
            return CompareCodePoints(sa, sb);
        if (a is string || b is string)
            throw new NumSharp.TypeError(
                $"'<' not supported between instances of '{PyTypeName(a)}' and '{PyTypeName(b)}'");
        return Convert.ToDouble(a, CultureInfo.InvariantCulture)
            .CompareTo(Convert.ToDouble(b, CultureInfo.InvariantCulture));
    }

    /// <summary>The Python type name of a key, for <see cref="ComparePythonKeys" />'s error message.</summary>
    /// <param name="v">The key.</param>
    /// <returns><c>str</c>, <c>float</c>, <c>bool</c> or <c>int</c>.</returns>
    private static string PyTypeName(object v) => v switch
    {
        string => "str",
        double or float => "float",
        bool => "bool",
        _ => "int",
    };

    /// <summary>Compares two strings code point by code point, as Python compares <c>str</c>.</summary>
    /// <param name="a">The first string.</param>
    /// <param name="b">The second string.</param>
    /// <returns>Negative, zero or positive.</returns>
    private static int CompareCodePoints(string a, string b)
    {
        // Rune enumeration decodes surrogate pairs, so an astral character sorts after U+FFFF as in Python; an unpaired
        // surrogate becomes U+FFFD here, which only matters for strings Python itself could not have produced from JSON.
        using var ea = a.EnumerateRunes().GetEnumerator();
        using var eb = b.EnumerateRunes().GetEnumerator();
        while (true)
        {
            bool ha = ea.MoveNext(), hb = eb.MoveNext();
            if (!ha || !hb)
                return ha.CompareTo(hb);
            int c = ea.Current.Value.CompareTo(eb.Current.Value);
            if (c != 0)
                return c;
        }
    }

    /// <summary>
    ///     <c>SafeRepresenter.represent_float</c>: <c>.nan</c>, <c>.inf</c>, <c>-.inf</c>, otherwise
    ///     <c>repr(data).lower()</c> with <c>.0</c> inserted before an exponent that has no decimal point
    ///     (<c>1e-07</c> → <c>1.0e-07</c>), which YAML 1.1's float resolver requires.
    /// </summary>
    /// <param name="d">The value.</param>
    /// <returns>PyYAML's text for it.</returns>
    private static string RepresentFloat(double d)
    {
        if (double.IsNaN(d))
            return ".nan";
        if (double.IsPositiveInfinity(d))
            return ".inf";
        if (double.IsNegativeInfinity(d))
            return "-.inf";
        string value = PythonRepr.FloatRepr(d).ToLowerInvariant();
        if (!value.Contains('.') && value.Contains('e'))
        {
            int e = value.IndexOf('e');
            value = value[..e] + ".0" + value[e..];
        }
        return value;
    }

    // ---------------------------------------------------------------- resolver (YAML 1.1, PyYAML's Resolver)

    // PyYAML's implicit resolvers, keyed by the first characters they are registered for, in registration order. The
    // patterns are its re.X patterns with the insignificant whitespace removed; `$` matches before a final line break in
    // both Python's re and .NET, so "yes\n" resolves to bool in both.
    private static readonly Regex BoolRe = new(
        @"^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF)$", RegexOptions.CultureInvariant);

    private static readonly Regex FloatRe = new(
        @"^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?" +
        @"|[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$",
        RegexOptions.CultureInvariant);

    private static readonly Regex IntRe = new(
        @"^(?:[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)|[-+]?0x[0-9a-fA-F_]+" +
        @"|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$",
        RegexOptions.CultureInvariant);

    private static readonly Regex MergeRe = new(@"^(?:<<)$", RegexOptions.CultureInvariant);

    private static readonly Regex NullRe = new(@"^(?:~|null|Null|NULL|)$", RegexOptions.CultureInvariant);

    private static readonly Regex TimestampRe = new(
        @"^(?:[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]" +
        @"|[0-9][0-9][0-9][0-9]-[0-9][0-9]?-[0-9][0-9]?(?:[Tt]|[ \t]+)[0-9][0-9]?:[0-9][0-9]:[0-9][0-9](?:\.[0-9]*)?" +
        @"(?:[ \t]*(?:Z|[-+][0-9][0-9]?(?::[0-9][0-9])?))?)$",
        RegexOptions.CultureInvariant);

    private static readonly Regex ValueRe = new(@"^(?:=)$", RegexOptions.CultureInvariant);

    private static readonly Regex YamlRe = new(@"^(?:!|&|\*)$", RegexOptions.CultureInvariant);

    /// <summary>
    ///     Whether PyYAML's <c>Resolver.resolve(ScalarNode, value, (True, False))</c> gives the string tag: no implicit
    ///     resolver registered for the value's first character (or for the empty string) matches it.
    /// </summary>
    /// <param name="value">The string scalar.</param>
    /// <returns><c>true</c> when the string may be written plain and still read back as a string.</returns>
    private static bool ResolvesToString(string value)
    {
        if (value.Length == 0)
            return !NullRe.IsMatch(value);
        // .NET Regex reads UTF-16; the first character decides which resolvers apply, as in PyYAML's dictionary.
        char first = value[0];
        foreach (Regex re in ResolversFor(first))
            if (re.IsMatch(value))
                return false;
        return true;
    }

    /// <summary>The resolvers PyYAML registered for <paramref name="first" />, in registration order.</summary>
    /// <param name="first">A scalar's first character.</param>
    /// <returns>The patterns to try.</returns>
    private static IEnumerable<Regex> ResolversFor(char first)
    {
        if ("yYnNtTfFoO".Contains(first)) yield return BoolRe;
        if ("-+0123456789.".Contains(first)) yield return FloatRe;
        if ("-+0123456789".Contains(first)) yield return IntRe;
        if (first == '<') yield return MergeRe;
        if (first is '~' or 'n' or 'N') yield return NullRe;
        if ("0123456789".Contains(first)) yield return TimestampRe;
        if (first == '=') yield return ValueRe;
        if (first is '!' or '&' or '*') yield return YamlRe;
    }

    // ---------------------------------------------------------------- emitter (PyYAML's Emitter, block style)

    /// <summary>PyYAML's <c>ScalarAnalysis</c>: what a scalar's characters allow.</summary>
    /// <param name="Scalar">The text.</param>
    /// <param name="Empty">The text is empty.</param>
    /// <param name="Multiline">The text contains a line break.</param>
    /// <param name="AllowFlowPlain">Plain style is allowed inside flow collections.</param>
    /// <param name="AllowBlockPlain">Plain style is allowed in block context.</param>
    /// <param name="AllowSingleQuoted">Single-quoted style is allowed.</param>
    /// <param name="AllowDoubleQuoted">Double-quoted style is allowed (always).</param>
    /// <param name="AllowBlock">Literal / folded block style is allowed.</param>
    private sealed record ScalarAnalysis(
        string Scalar, bool Empty, bool Multiline, bool AllowFlowPlain, bool AllowBlockPlain,
        bool AllowSingleQuoted, bool AllowDoubleQuoted, bool AllowBlock);

    /// <summary>
    ///     A port of PyYAML's <c>Emitter</c> state machine for the events <see cref="Represent" /> produces, with
    ///     <c>safe_dump</c>'s settings (indent 2, width 80, line break <c>\n</c>, <c>allow_unicode=False</c>, not
    ///     canonical). Method names follow PyYAML's so the two can be read side by side.
    /// </summary>
    private sealed class Emitter
    {
        private const int BestIndent = 2;
        private const int BestWidth = 80;
        private const string LineBreaks = "\n\u0085\u2028\u2029";
        private const string Whitespace = "\0 \t\r\n\u0085\u2028\u2029";

        private readonly List<Event> _events;
        private readonly StringBuilder _out = new();
        private readonly Stack<Action> _states = new();
        private readonly Stack<int?> _indents = new();
        private int _position;
        private Event? _event;
        private int? _indent;
        private int _flowLevel;
        private bool _rootContext, _sequenceContext, _mappingContext, _simpleKeyContext;
        private int _column;
        private bool _whitespace = true;
        private bool _indention = true;
        // PyYAML's open_ended: a plain scalar at the root leaves the document open, so the stream ends with `...`.
        private bool _openEnded;
        private ScalarAnalysis? _analysis;
        private char? _style;

        /// <summary>Creates an emitter over a complete event list (one document).</summary>
        /// <param name="events">The node events of the value, without stream or document events.</param>
        internal Emitter(List<Event> events) => _events = events;

        /// <summary>
        ///     Emits the stream: an implicit first document (no <c>---</c>), the root node, the document end's line break,
        ///     and, when the root was a plain scalar, the stream end's <c>...</c> line (PyYAML's <c>open_ended</c>).
        /// </summary>
        /// <returns>The emitted text.</returns>
        internal string Run()
        {
            // expect_document_start(first=True) with an implicit document writes nothing; expect_document_root pushes
            // expect_document_end and emits the root node.
            _states.Push(ExpectDocumentEnd);
            _event = Next();
            ExpectNode(root: true);
            while (_states.Count > 0)
            {
                Action state = _states.Pop();
                if (state == ExpectDocumentEnd)
                {
                    ExpectDocumentEnd();
                    // expect_document_start on the StreamEndEvent: an open-ended document is closed explicitly.
                    if (_openEnded)
                    {
                        WriteIndicator("...", true);
                        WriteIndent();
                    }
                    break;
                }
                _event = Next();
                state();
            }
            return _out.ToString();
        }

        /// <summary>Takes the next event, or <c>null</c> once the document's events are exhausted.</summary>
        /// <returns>The next event.</returns>
        private Event? Next() => _position < _events.Count ? _events[_position++] : null;

        /// <summary>The event after the current one, PyYAML's <c>self.events[0]</c>.</summary>
        private Event? Peek => _position < _events.Count ? _events[_position] : null;

        /// <summary><c>expect_document_end</c> for a non-explicit end: a final indent (line break).</summary>
        private void ExpectDocumentEnd() => WriteIndent();

        /// <summary><c>increase_indent</c>.</summary>
        /// <param name="flow">Whether a flow collection or a scalar is starting.</param>
        /// <param name="indentless">Whether a block sequence sits at its mapping key's indent.</param>
        private void IncreaseIndent(bool flow = false, bool indentless = false)
        {
            _indents.Push(_indent);
            if (_indent is null)
                _indent = flow ? BestIndent : 0;
            else if (!indentless)
                _indent += BestIndent;
        }

        /// <summary><c>expect_node</c> for the current event.</summary>
        /// <param name="root">The node is the document root.</param>
        /// <param name="sequence">The node is a sequence item.</param>
        /// <param name="mapping">The node is a mapping key or value.</param>
        /// <param name="simpleKey">The node is a simple (implicit) mapping key.</param>
        /// <exception cref="InvalidOperationException">The current event is not a node (an event list out of order; never produced by <see cref="Represent" />).</exception>
        private void ExpectNode(bool root = false, bool sequence = false, bool mapping = false, bool simpleKey = false)
        {
            _rootContext = root;
            _sequenceContext = sequence;
            _mappingContext = mapping;
            _simpleKeyContext = simpleKey;
            switch (_event)
            {
                case ScalarEvent:
                    ExpectScalar();
                    break;
                case SequenceStartEvent:
                    // safe_dump's default_flow_style=False: block style, except that an empty collection is flow.
                    if (_flowLevel > 0 || CheckEmptySequence())
                        ExpectFlowSequence();
                    else
                        ExpectBlockSequence();
                    break;
                case MappingStartEvent:
                    if (_flowLevel > 0 || CheckEmptyMapping())
                        ExpectFlowMapping();
                    else
                        ExpectBlockMapping();
                    break;
                default:
                    throw new InvalidOperationException($"expected NodeEvent, but got {_event}");
            }
        }

        /// <summary><c>expect_scalar</c>.</summary>
        private void ExpectScalar()
        {
            IncreaseIndent(flow: true);
            ProcessScalar();
            _indent = _indents.Pop();
        }

        /// <summary><c>expect_flow_sequence</c>, reached only by an empty sequence here.</summary>
        private void ExpectFlowSequence()
        {
            WriteIndicator("[", true, whitespace: true);
            _flowLevel++;
            IncreaseIndent(flow: true);
            _states.Push(ExpectFirstFlowSequenceItem);
        }

        /// <summary><c>expect_first_flow_sequence_item</c>: the sequence's end, as it is empty.</summary>
        /// <exception cref="InvalidOperationException">A flow sequence has items (never produced by <see cref="Dump" />).</exception>
        private void ExpectFirstFlowSequenceItem()
        {
            if (_event is not SequenceEndEvent)
                throw new InvalidOperationException("non-empty flow sequences are not produced by safe_dump");
            _indent = _indents.Pop();
            _flowLevel--;
            WriteIndicator("]", false);
        }

        /// <summary><c>expect_flow_mapping</c>, reached only by an empty mapping here.</summary>
        private void ExpectFlowMapping()
        {
            WriteIndicator("{", true, whitespace: true);
            _flowLevel++;
            IncreaseIndent(flow: true);
            _states.Push(ExpectFirstFlowMappingKey);
        }

        /// <summary><c>expect_first_flow_mapping_key</c>: the mapping's end, as it is empty.</summary>
        /// <exception cref="InvalidOperationException">A flow mapping has items (never produced by <see cref="Dump" />).</exception>
        private void ExpectFirstFlowMappingKey()
        {
            if (_event is not MappingEndEvent)
                throw new InvalidOperationException("non-empty flow mappings are not produced by safe_dump");
            _indent = _indents.Pop();
            _flowLevel--;
            WriteIndicator("}", false);
        }

        /// <summary><c>expect_block_sequence</c>: a sequence that is a mapping value stays at the key's indent.</summary>
        private void ExpectBlockSequence()
        {
            bool indentless = _mappingContext && !_indention;
            IncreaseIndent(flow: false, indentless: indentless);
            _states.Push(ExpectFirstBlockSequenceItem);
        }

        /// <summary><c>expect_first_block_sequence_item</c>.</summary>
        private void ExpectFirstBlockSequenceItem() => ExpectBlockSequenceItem(first: true);

        /// <summary><c>expect_block_sequence_item</c>.</summary>
        private void ExpectNextBlockSequenceItem() => ExpectBlockSequenceItem(first: false);

        /// <summary><c>expect_block_sequence_item</c>.</summary>
        /// <param name="first">Whether this is the first item (a sequence end cannot come first).</param>
        private void ExpectBlockSequenceItem(bool first)
        {
            if (!first && _event is SequenceEndEvent)
            {
                _indent = _indents.Pop();
                return;
            }
            WriteIndent();
            WriteIndicator("-", true, indention: true);
            _states.Push(ExpectNextBlockSequenceItem);
            ExpectNode(sequence: true);
        }

        /// <summary><c>expect_block_mapping</c>.</summary>
        private void ExpectBlockMapping()
        {
            IncreaseIndent(flow: false);
            _states.Push(ExpectFirstBlockMappingKey);
        }

        /// <summary><c>expect_first_block_mapping_key</c>.</summary>
        private void ExpectFirstBlockMappingKey() => ExpectBlockMappingKey(first: true);

        /// <summary><c>expect_block_mapping_key</c>.</summary>
        private void ExpectNextBlockMappingKey() => ExpectBlockMappingKey(first: false);

        /// <summary><c>expect_block_mapping_key</c>: a simple <c>key:</c> when possible, else <c>? key</c>.</summary>
        /// <param name="first">Whether this is the first key (a mapping end cannot come first).</param>
        private void ExpectBlockMappingKey(bool first)
        {
            if (!first && _event is MappingEndEvent)
            {
                _indent = _indents.Pop();
                return;
            }
            WriteIndent();
            if (CheckSimpleKey())
            {
                _states.Push(ExpectBlockMappingSimpleValue);
                ExpectNode(mapping: true, simpleKey: true);
            }
            else
            {
                WriteIndicator("?", true, indention: true);
                _states.Push(ExpectBlockMappingValue);
                ExpectNode(mapping: true);
            }
        }

        /// <summary><c>expect_block_mapping_simple_value</c>.</summary>
        private void ExpectBlockMappingSimpleValue()
        {
            WriteIndicator(":", false);
            _states.Push(ExpectNextBlockMappingKey);
            ExpectNode(mapping: true);
        }

        /// <summary><c>expect_block_mapping_value</c>, after a complex (<c>?</c>) key.</summary>
        private void ExpectBlockMappingValue()
        {
            WriteIndent();
            WriteIndicator(":", true, indention: true);
            _states.Push(ExpectNextBlockMappingKey);
            ExpectNode(mapping: true);
        }

        /// <summary><c>check_empty_sequence</c>.</summary>
        /// <returns>Whether the current sequence start is followed by its end.</returns>
        private bool CheckEmptySequence() => _event is SequenceStartEvent && Peek is SequenceEndEvent;

        /// <summary><c>check_empty_mapping</c>.</summary>
        /// <returns>Whether the current mapping start is followed by its end.</returns>
        private bool CheckEmptyMapping() => _event is MappingStartEvent && Peek is MappingEndEvent;

        /// <summary>
        ///     <c>check_simple_key</c>: a key can be written as <c>key:</c> when it is a scalar with no line break whose
        ///     prepared tag and text together are shorter than 128 characters, or an empty collection.
        /// </summary>
        /// <returns>Whether the current key is simple.</returns>
        private bool CheckSimpleKey()
        {
            int length = 0;
            if (_event is ScalarEvent scalar)
            {
                // Every represented node carries a tag, so PyYAML adds its prepared form (`!!str` and so on) even though
                // the tag is never written for these scalars.
                length += scalar.Tag.Length;
                _analysis ??= AnalyzeScalar(scalar.Value);
                // Python's len(): code points, so an astral character counts once toward the 128 limit.
                length += CodePointLength(_analysis.Scalar);
            }
            else if (_event is SequenceStartEvent or MappingStartEvent)
            {
                length += "!!seq".Length;   // `!!seq` and `!!map` are both five characters
            }
            return length < 128
                   && ((_event is ScalarEvent && !_analysis!.Empty && !_analysis.Multiline)
                       || CheckEmptySequence() || CheckEmptyMapping());
        }

        /// <summary><c>choose_scalar_style</c> with no requested style: plain, else single-quoted, else double-quoted.</summary>
        /// <returns><c>null</c> for plain, <c>'</c> or <c>"</c>.</returns>
        private char? ChooseScalarStyle()
        {
            var scalar = (ScalarEvent)_event!;
            _analysis ??= AnalyzeScalar(scalar.Value);
            if (scalar.PlainImplicit
                && !(_simpleKeyContext && (_analysis.Empty || _analysis.Multiline))
                && ((_flowLevel > 0 && _analysis.AllowFlowPlain) || (_flowLevel == 0 && _analysis.AllowBlockPlain)))
                return null;
            if (_analysis.AllowSingleQuoted && !(_simpleKeyContext && _analysis.Multiline))
                return '\'';
            return '"';
        }

        /// <summary>
        ///     <c>process_tag</c> + <c>process_scalar</c>: safe_dump's scalars never need a tag (their style matches
        ///     their implicit flags), so only the scalar is written.
        /// </summary>
        private void ProcessScalar()
        {
            var scalar = (ScalarEvent)_event!;
            _analysis ??= AnalyzeScalar(scalar.Value);
            _style ??= ChooseScalarStyle();
            bool split = !_simpleKeyContext;
            switch (_style)
            {
                case '"':
                    WriteDoubleQuoted(_analysis.Scalar, split);
                    break;
                case '\'':
                    WriteSingleQuoted(_analysis.Scalar, split);
                    break;
                default:
                    WritePlain(_analysis.Scalar, split);
                    break;
            }
            _analysis = null;
            _style = null;
        }

        /// <summary><c>analyze_scalar</c>: which styles the scalar's characters allow.</summary>
        /// <param name="scalar">The text.</param>
        /// <returns>The analysis.</returns>
        private static ScalarAnalysis AnalyzeScalar(string scalar)
        {
            if (scalar.Length == 0)
                return new ScalarAnalysis(scalar, Empty: true, Multiline: false, AllowFlowPlain: false,
                    AllowBlockPlain: true, AllowSingleQuoted: true, AllowDoubleQuoted: true, AllowBlock: false);

            bool blockIndicators = false, flowIndicators = false, lineBreaks = false, specialCharacters = false;
            bool leadingSpace = false, leadingBreak = false, trailingSpace = false, trailingBreak = false;
            bool breakSpace = false, spaceBreak = false;

            if (scalar.StartsWith("---", StringComparison.Ordinal) || scalar.StartsWith("...", StringComparison.Ordinal))
            {
                blockIndicators = true;
                flowIndicators = true;
            }

            // Python iterates code points; a surrogate pair is one character there. Iterate runes so the "followed by
            // whitespace" look-ahead and the unicode-range tests see the same characters.
            int[] cps = scalar.EnumerateRunes().Select(r => r.Value).ToArray();
            bool precededByWhitespace = true;
            bool followedByWhitespace = cps.Length == 1 || IsWhitespace(cps[1]);
            bool previousSpace = false, previousBreak = false;

            for (int index = 0; index < cps.Length; index++)
            {
                int ch = cps[index];
                if (index == 0)
                {
                    if ("#,[]{}&*!|>'\"%@`".Contains((char)ch) && ch < 0x10000)
                    {
                        flowIndicators = true;
                        blockIndicators = true;
                    }
                    if (ch is '?' or ':')
                    {
                        flowIndicators = true;
                        if (followedByWhitespace)
                            blockIndicators = true;
                    }
                    if (ch == '-' && followedByWhitespace)
                    {
                        flowIndicators = true;
                        blockIndicators = true;
                    }
                }
                else
                {
                    if (ch is ',' or '?' or '[' or ']' or '{' or '}')
                        flowIndicators = true;
                    if (ch == ':')
                    {
                        flowIndicators = true;
                        if (followedByWhitespace)
                            blockIndicators = true;
                    }
                    if (ch == '#' && precededByWhitespace)
                    {
                        flowIndicators = true;
                        blockIndicators = true;
                    }
                }

                if (IsLineBreak(ch))
                    lineBreaks = true;
                if (!(ch == '\n' || (ch >= 0x20 && ch <= 0x7E)))
                {
                    if ((ch == 0x85 || (ch >= 0xA0 && ch <= 0xD7FF) || (ch >= 0xE000 && ch <= 0xFFFD)
                         || (ch >= 0x10000 && ch < 0x10FFFF)) && ch != 0xFEFF)
                    {
                        // unicode_characters: with allow_unicode=False they are special too.
                        specialCharacters = true;
                    }
                    else
                    {
                        specialCharacters = true;
                    }
                }

                if (ch == ' ')
                {
                    if (index == 0)
                        leadingSpace = true;
                    if (index == cps.Length - 1)
                        trailingSpace = true;
                    if (previousBreak)
                        breakSpace = true;
                    previousSpace = true;
                    previousBreak = false;
                }
                else if (IsLineBreak(ch))
                {
                    if (index == 0)
                        leadingBreak = true;
                    if (index == cps.Length - 1)
                        trailingBreak = true;
                    if (previousSpace)
                        spaceBreak = true;
                    previousSpace = false;
                    previousBreak = true;
                }
                else
                {
                    previousSpace = false;
                    previousBreak = false;
                }

                precededByWhitespace = IsWhitespace(ch);
                followedByWhitespace = index + 2 >= cps.Length || IsWhitespace(cps[index + 2]);
            }

            bool allowFlowPlain = true, allowBlockPlain = true, allowSingleQuoted = true, allowBlock = true;
            if (leadingSpace || leadingBreak || trailingSpace || trailingBreak)
                allowFlowPlain = allowBlockPlain = false;
            if (trailingSpace)
                allowBlock = false;
            if (breakSpace)
                allowFlowPlain = allowBlockPlain = allowSingleQuoted = false;
            if (spaceBreak || specialCharacters)
                allowFlowPlain = allowBlockPlain = allowSingleQuoted = allowBlock = false;
            if (lineBreaks)
                allowFlowPlain = allowBlockPlain = false;
            if (flowIndicators)
                allowFlowPlain = false;
            if (blockIndicators)
                allowBlockPlain = false;

            return new ScalarAnalysis(scalar, Empty: false, Multiline: lineBreaks, allowFlowPlain, allowBlockPlain,
                allowSingleQuoted, AllowDoubleQuoted: true, allowBlock);
        }

        /// <summary>Whether <paramref name="ch" /> is one of YAML's line breaks (<c>\n</c>, NEL, LS, PS).</summary>
        /// <param name="ch">A code point.</param>
        /// <returns>The test result.</returns>
        private static bool IsLineBreak(int ch) => ch is '\n' or 0x85 or 0x2028 or 0x2029;

        /// <summary>Whether <paramref name="ch" /> is in PyYAML's <c>'\0 \t\r\n\x85\u2028\u2029'</c>.</summary>
        /// <param name="ch">A code point.</param>
        /// <returns>The test result.</returns>
        private static bool IsWhitespace(int ch) => ch < 0x10000 && Whitespace.Contains((char)ch);

        // ---- writers

        /// <summary><c>write_indicator</c>.</summary>
        /// <param name="indicator">The indicator text.</param>
        /// <param name="needWhitespace">Whether a space must separate it from the previous text.</param>
        /// <param name="whitespace">Whether the indicator ends in whitespace.</param>
        /// <param name="indention">Whether the indicator counts as indentation.</param>
        private void WriteIndicator(string indicator, bool needWhitespace, bool whitespace = false, bool indention = false)
        {
            string data = _whitespace || !needWhitespace ? indicator : " " + indicator;
            _whitespace = whitespace;
            _indention = _indention && indention;
            _column += data.Length;
            _openEnded = false;
            _out.Append(data);
        }

        /// <summary><c>write_indent</c>: a line break when needed, then spaces up to the current indent.</summary>
        private void WriteIndent()
        {
            int indent = _indent ?? 0;
            if (!_indention || _column > indent || (_column == indent && !_whitespace))
                WriteLineBreak();
            if (_column < indent)
            {
                _whitespace = true;
                _out.Append(' ', indent - _column);
                _column = indent;
            }
        }

        /// <summary><c>write_line_break</c>.</summary>
        /// <param name="data">The break to write; <c>\n</c> by default.</param>
        private void WriteLineBreak(string? data = null)
        {
            _whitespace = true;
            _indention = true;
            _column = 0;
            _out.Append(data ?? "\n");
        }

        /// <summary>Appends text that contains no line break and advances the column by its code points.</summary>
        /// <param name="data">The text.</param>
        private void Append(string data)
        {
            // PyYAML counts columns in code points (Python string length); a surrogate pair is one.
            _column += CodePointLength(data);
            _out.Append(data);
        }

        /// <summary>The number of code points in <paramref name="s" />.</summary>
        /// <param name="s">A string.</param>
        /// <returns>Its length as Python's <c>len</c> counts it.</returns>
        private static int CodePointLength(string s)
        {
            int n = 0;
            foreach (Rune _ in s.EnumerateRunes())
                n++;
            return n;
        }

        /// <summary><c>write_plain</c>: the text as is, folded at spaces past the width when splitting is allowed.</summary>
        /// <param name="text">The scalar.</param>
        /// <param name="split">Whether long lines may fold (not for simple keys).</param>
        private void WritePlain(string text, bool split)
        {
            if (_rootContext)
                _openEnded = true;
            if (text.Length == 0)
                return;
            if (!_whitespace)
                Append(" ");
            _whitespace = false;
            _indention = false;
            string[] t = ToCodePoints(text);
            bool spaces = false, breaks = false;
            int start = 0, end = 0;
            while (end <= t.Length)
            {
                string? ch = end < t.Length ? t[end] : null;
                if (spaces)
                {
                    if (ch != " ")
                    {
                        if (start + 1 == end && _column > BestWidth && split)
                        {
                            WriteIndent();
                            _whitespace = false;
                            _indention = false;
                        }
                        else
                        {
                            Append(Slice(t, start, end));
                        }
                        start = end;
                    }
                }
                else if (breaks)
                {
                    if (ch is null || !IsLineBreakText(ch))
                    {
                        if (t[start] == "\n")
                            WriteLineBreak();
                        for (int i = start; i < end; i++)
                            WriteLineBreak(t[i] == "\n" ? null : t[i]);
                        WriteIndent();
                        _whitespace = false;
                        _indention = false;
                        start = end;
                    }
                }
                else
                {
                    if (ch is null || ch == " " || IsLineBreakText(ch))
                    {
                        Append(Slice(t, start, end));
                        start = end;
                    }
                }
                if (ch is not null)
                {
                    spaces = ch == " ";
                    breaks = IsLineBreakText(ch);
                }
                end++;
            }
        }

        /// <summary><c>write_single_quoted</c>: <c>'</c> doubled, folded at spaces past the width.</summary>
        /// <param name="text">The scalar.</param>
        /// <param name="split">Whether long lines may fold (not for simple keys).</param>
        private void WriteSingleQuoted(string text, bool split)
        {
            WriteIndicator("'", true);
            string[] t = ToCodePoints(text);
            bool spaces = false, breaks = false;
            int start = 0, end = 0;
            while (end <= t.Length)
            {
                string? ch = end < t.Length ? t[end] : null;
                if (spaces)
                {
                    if (ch is null || ch != " ")
                    {
                        if (start + 1 == end && _column > BestWidth && split && start != 0 && end != t.Length)
                            WriteIndent();
                        else
                            Append(Slice(t, start, end));
                        start = end;
                    }
                }
                else if (breaks)
                {
                    if (ch is null || !IsLineBreakText(ch))
                    {
                        if (t[start] == "\n")
                            WriteLineBreak();
                        for (int i = start; i < end; i++)
                            WriteLineBreak(t[i] == "\n" ? null : t[i]);
                        WriteIndent();
                        start = end;
                    }
                }
                else
                {
                    if (ch is null || ch == " " || IsLineBreakText(ch) || ch == "'")
                    {
                        if (start < end)
                        {
                            Append(Slice(t, start, end));
                            start = end;
                        }
                    }
                }
                if (ch == "'")
                {
                    Append("''");
                    start = end + 1;
                }
                if (ch is not null)
                {
                    spaces = ch == " ";
                    breaks = IsLineBreakText(ch);
                }
                end++;
            }
            WriteIndicator("'", false);
        }

        /// <summary>PyYAML's <c>ESCAPE_REPLACEMENTS</c>.</summary>
        private static readonly Dictionary<int, string> EscapeReplacements = new()
        {
            [0x00] = "0", [0x07] = "a", [0x08] = "b", [0x09] = "t", [0x0A] = "n", [0x0B] = "v", [0x0C] = "f",
            [0x0D] = "r", [0x1B] = "e", ['"'] = "\"", ['\\'] = "\\", [0x85] = "N", [0xA0] = "_", [0x2028] = "L",
            [0x2029] = "P",
        };

        /// <summary>
        ///     <c>write_double_quoted</c>: printable ASCII as is, everything else escaped (<c>allow_unicode=False</c>),
        ///     folded with a trailing <c>\</c> past the width.
        /// </summary>
        /// <param name="text">The scalar.</param>
        /// <param name="split">Whether long lines may fold (not for simple keys).</param>
        private void WriteDoubleQuoted(string text, bool split)
        {
            WriteIndicator("\"", true);
            int[] t = text.EnumerateRunes().Select(r => r.Value).ToArray();
            int start = 0, end = 0;
            while (end <= t.Length)
            {
                int? ch = end < t.Length ? t[end] : null;
                if (ch is null || ch is '"' or '\\' or 0x85 or 0x2028 or 0x2029 or 0xFEFF || !(ch >= 0x20 && ch <= 0x7E))
                {
                    if (start < end)
                    {
                        Append(SliceRunes(t, start, end));
                        start = end;
                    }
                    if (ch is not null)
                    {
                        string data;
                        if (EscapeReplacements.TryGetValue(ch.Value, out string? rep))
                            data = "\\" + rep;
                        else if (ch <= 0xFF)
                            data = "\\x" + ch.Value.ToString("X2", CultureInfo.InvariantCulture);
                        else if (ch <= 0xFFFF)
                            data = "\\u" + ch.Value.ToString("X4", CultureInfo.InvariantCulture);
                        else
                            data = "\\U" + ch.Value.ToString("X8", CultureInfo.InvariantCulture);
                        Append(data);
                        start = end + 1;
                    }
                }
                if (0 < end && end < t.Length - 1 && (ch == ' ' || start >= end)
                    && _column + (end - start) > BestWidth && split)
                {
                    string data = SliceRunes(t, start, end) + "\\";
                    if (start < end)
                        start = end;
                    Append(data);
                    WriteIndent();
                    _whitespace = false;
                    _indention = false;
                    if (t[start] == ' ')
                        Append("\\");
                }
                end++;
            }
            WriteIndicator("\"", false);
        }

        /// <summary>Splits a string into its code points, each as a string (one or two UTF-16 units).</summary>
        /// <param name="s">The text.</param>
        /// <returns>The code points.</returns>
        private static string[] ToCodePoints(string s) => s.EnumerateRunes().Select(r => r.ToString()).ToArray();

        /// <summary>Concatenates code points <c>[start, end)</c>.</summary>
        /// <param name="t">The code points.</param>
        /// <param name="start">The first index.</param>
        /// <param name="end">The index past the last.</param>
        /// <returns>The text.</returns>
        private static string Slice(string[] t, int start, int end) => string.Concat(t.Skip(start).Take(end - start));

        /// <summary>Encodes code points <c>[start, end)</c> as a string.</summary>
        /// <param name="t">The code points.</param>
        /// <param name="start">The first index.</param>
        /// <param name="end">The index past the last.</param>
        /// <returns>The text.</returns>
        private static string SliceRunes(int[] t, int start, int end)
        {
            var sb = new StringBuilder();
            for (int i = start; i < end; i++)
                sb.Append(char.ConvertFromUtf32(t[i]));
            return sb.ToString();
        }

        /// <summary>Whether a one-code-point string is a YAML line break.</summary>
        /// <param name="ch">The code point as a string.</param>
        /// <returns>The test result.</returns>
        private static bool IsLineBreakText(string ch) => ch.Length == 1 && LineBreaks.Contains(ch[0]);
    }
}
