"""Report how often SRN's question tokens use real pretrained GloVe vectors.

SRN consumes GloVe for question words only. The <e> topic-entity marker
deliberately has no GloVe vector and is excluded from OOV rates.
"""

from collections import Counter


SPECIAL_TOKENS = frozenset(("<e>",))


def _statistics(counts, available):
    special = {token: count for token, count in sorted(counts.items())
               if token in SPECIAL_TOKENS}
    words = {token: count for token, count in counts.items()
             if token not in SPECIAL_TOKENS}
    matched = {token for token in words if token in available}
    missing = sorted(
        ((token, count) for token, count in words.items() if token not in available),
        key=lambda item: (-item[1], item[0]),
    )
    total_occurrences = sum(words.values())
    matched_occurrences = sum(words[word] for word in matched)
    return {
        "unique_tokens": len(words),
        "matched_unique": len(matched),
        "unique_coverage": len(matched) / len(words) if words else 0.0,
        "token_occurrences": total_occurrences,
        "matched_occurrences": matched_occurrences,
        "occurrence_coverage": (
            matched_occurrences / total_occurrences if total_occurrences else 0.0
        ),
        "excluded_special_tokens": special,
        "missing_tokens": [
            {"token": token, "occurrences": count} for token, count in missing
        ],
    }


def compute_glove_coverage(splits, word_embeddings):
    """Measure exact SRN token-to-GloVe matches on every loaded QA split.

    Each record has a token list in record[0]. The same lowercase lookup
    is used by Embedder.embed_word during training and evaluation.
    """
    available = set(word_embeddings)
    overall = Counter()
    by_split = {}
    for split, records in splits.items():
        counts = Counter(
            token.lower() for record in records for token in record[0]
        )
        overall.update(counts)
        by_split[split] = _statistics(counts, available)
    return {
        "scope": "SRN-tokenized canonical question words in loaded QA splits",
        "lookup": "exact lowercase token matching, as in Embedder.embed_word",
        "note": "The <e> topic marker is intentionally zero-initialized. "
                "TransE KG relations do not use GloVe.",
        "all": _statistics(overall, available),
        "by_split": by_split,
    }


def format_glove_coverage(report, dimension, missing_limit=10):
    """Render coverage diagnostics without printing all missing words."""
    def coverage_line(label, stats):
        return (
            "{}: unique {}/{} ({:.1%}), occurrences {}/{} ({:.1%})"
        ).format(
            label,
            stats["matched_unique"], stats["unique_tokens"],
            stats["unique_coverage"],
            stats["matched_occurrences"], stats["token_occurrences"],
            stats["occurrence_coverage"],
        )

    lines = ["GloVe vocabulary coverage ({}d, SRN question tokens):".format(dimension)]
    lines.append("  " + coverage_line("all", report["all"]))
    for split, stats in report["by_split"].items():
        lines.append("  " + coverage_line(split, stats))
    excluded = report["all"]["excluded_special_tokens"]
    if excluded:
        lines.append("  Excluded special tokens: {}".format(excluded))
    missing = report["all"]["missing_tokens"]
    if missing:
        sample = ", ".join(
            "{} ({})".format(item["token"], item["occurrences"])
            for item in missing[:missing_limit]
        )
        lines.append(
            "  Top GloVe OOV words ({} total): {}".format(len(missing), sample)
        )
    return "\n".join(lines)
