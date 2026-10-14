"""Shared source-preserving sentence boundaries; decisions remain heuristics."""

TERMINALS = frozenset(".!?。！？؟।॥")
CLOSERS = frozenset("\"'”’»)]}")


def sentence_ranges(tokens, spans=None, text=None, abbreviations=()):
    """Return covering half-open token ranges, without inventing word boundaries.

    Corpus-derived abbreviation spellings are hints, not universal rules. Dots
    inside adjacent alphanumerics and punctuation runs do not terminate early.
    Closing punctuation belongs to the preceding sentence. Other script-specific
    endings work without requiring whitespace or capitalization.
    """
    abbreviations = set(abbreviations)
    ranges = []
    start = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        boundary = token in TERMINALS
        if boundary and token == "." and i + 1 < len(tokens):
            before = tokens[i - 1] if i else ""
            after = tokens[i + 1]
            adjacent = (
                (
                    spans is not None
                    and spans[i - 1][1] == spans[i][0]
                    and spans[i][1] == spans[i + 1][0]
                )
                if i
                else False
            )
            if adjacent and before[-1:].isalnum() and after[:1].isalnum():
                boundary = False
            if before.lower() in abbreviations and after[:1].isalpha():
                boundary = False
        if boundary:
            while i + 1 < len(tokens) and (
                tokens[i + 1] in TERMINALS
                or tokens[i + 1] in CLOSERS
                and (spans is None or spans[i][1] == spans[i + 1][0])
            ):
                i += 1
            ranges.append((start, i + 1))
            start = i + 1
        i += 1
    if start < len(tokens):
        ranges.append((start, len(tokens)))
    return ranges


def validate_ranges(ranges, size):
    ranges = list(ranges)
    cursor = 0
    for left, right in ranges:
        if (
            type(left) is not int
            or type(right) is not int
            or left != cursor
            or not left < right <= size
        ):
            raise ValueError("Sentence ranges must cover consecutive tokens exactly")
        cursor = right
    if cursor != size:
        raise ValueError("Sentence ranges must cover all tokens")
    return ranges
