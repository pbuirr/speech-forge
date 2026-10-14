"""Carry generated-input uncertainty across analysis, translation and learning."""

from speech_engine.linguistic_resources import language_code


def input_evidence(value, *, _include_origin=True):
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("method") not in (
        "local_ctc",
        "browser_speech",
        "local_translation",
    ):
        raise ValueError("Input evidence must identify a supported generated-text source")
    result = {
        "method": value["method"],
        "status": "predicted",
        "confidence": None,
        "review_required": True,
        "note": "Source text is model-generated. Downstream analysis does not verify fidelity to the original audio or text.",
    }
    if value.get("language"):
        language_code(value["language"])
        result["language"] = value["language"].lower()
    if value.get("model_sha256") is not None:
        digest = value["model_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in digest)
        ):
            raise ValueError("Model fingerprint must be a SHA256 hexadecimal digest")
        result["model_sha256"] = digest.lower()
    if _include_origin and value.get("original_input_evidence") is not None:
        # Keep one original source, rather than an unbounded recursive history.
        # This is declared provenance, never authentication or verified accuracy.
        result["original_input_evidence"] = input_evidence(
            value["original_input_evidence"], _include_origin=False
        )
    # No client-provided confidence or verified flag can establish correctness.
    return result
