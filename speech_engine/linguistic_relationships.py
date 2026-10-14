"""Links tokens in a sentence that share a lemma and POS, and notes which features differ."""

from speech_engine.linguistic_resources import normalized_form


FEATURE_NOTES = {
    "Number": "Grammatical number; the available distinctions depend on the language.",
    "Case": "Case marking can contribute to grammatical roles; case alone does not identify a dependency or semantic role.",
    "Tense": "Grammatical tense contrasts; interpreting event time also requires context.",
    "Aspect": "Aspect describes grammatical presentation of an event, separately from tense.",
    "Person": "Grammatical person distinctions can reflect participants or agreement.",
    "Gender": "Grammatical gender or noun classification; not a claim about natural gender.",
    "Mood": "Grammatical mood distinctions; interpretation depends on the construction and language.",
    "Voice": "Voice marking concerns how participants are expressed grammatically.",
    "VerbForm": "Different grammatical verb forms can participate in different constructions.",
    "Degree": "Grammatical comparison distinctions.",
}


def form_relationships(tokens):
    """Link each token to the previous token with the same lemma and POS, in O(n).

    Features that differ between the two are listed; a feature missing on either
    side isn't counted as a difference.
    """
    previous = {}
    links = []
    for index, token in enumerate(tokens):
        if (
            not token.get("candidates")
            or token.get("root_status") not in {"corpus_attested", "predicted"}
            or token.get("pos_status") not in {"corpus_attested", "predicted"}
            or token.get("pos") in {"PUNCT", "SYM", "X"}
        ):
            continue
        key = (normalized_form(token["root"]), token["pos"])
        if key in previous:
            left = previous[key]
            first = tokens[left]
            a, b = first.get("ud_features", {}), token.get("ud_features", {})
            contrasts = []
            for feature in sorted(a.keys() & b.keys()):
                if a[feature] == b[feature]:
                    continue
                contrasts.append(
                    {
                        "feature": feature,
                        "left": a[feature],
                        "right": b[feature],
                        "interpretation": FEATURE_NOTES.get(
                            feature,
                            "Different corpus feature annotations; consult language-specific documentation.",
                        ),
                        "reference": "https://universaldependencies.org/u/feat/"
                        + feature.split("[")[0]
                        + ".html",
                        "overlapping_values": bool(
                            set(a[feature].split(",")) & set(b[feature].split(","))
                        ),
                    }
                )
            predicted = any(
                t.get("root_status") == "predicted"
                or t.get("pos_status") == "predicted"
                or t.get("method") == "corpus_context"
                for t in [first, token]
            )
            links.append(
                {
                    "kind": "shared_lemma_form_comparison",
                    "token_indices": [left, index],
                    "lemma": token["root"],
                    "pos": token["pos"],
                    "contrasts": contrasts,
                    "endpoints": [
                        {
                            "token_index": i,
                            "surface": t["surface"],
                            "span": t.get("span"),
                            "candidates": t["candidates"],
                            "selected_candidate_indices": t.get("selected_candidate_indices", []),
                            "pos_uncertainty": t.get("pos_uncertainty"),
                        }
                        for i, t in [(left, first), (index, token)]
                    ],
                    "status": "predicted" if predicted else "corpus_supported",
                    "confidence": None,
                    "method": "selected_corpus_lemma_pos",
                    "note": "Reused corpus feature annotations are not context-verified or feature-calibrated. Same lemma/POS evidence does not establish the same word sense, dependency, or agreement.",
                }
            )
        previous[key] = index
    return links
