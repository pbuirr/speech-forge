import io
import tarfile

import pytest

from scripts.speech.prepare_voxpopuli_accent_test import RangeTar, eligible_rows, select_rows


def row(identifier, session="20190101-event", speaker="1", accent="en_fr", **extra):
    return {
        "id_": identifier,
        "session_id": session,
        "speaker_id": speaker,
        "accent": accent,
        "split": "test",
        "is_gold_transcript": "True",
        "vad": "[[1, 3]]",
        "normed_text": "i cannot say",
        **extra,
    }


def test_gold_challenge_excludes_bad_boundaries_without_repairing_labels():
    rows = [
        row("valid"),
        row("reversed", vad="[[3, 1]]"),
        row("overlap", vad="[[1, 3], [2, 4]]"),
        row("long", vad="[[1, 30]]"),
        row("lexical-loss", normed_text="café"),
        row("not-gold", is_gold_transcript="False"),
        row("not-test", split="train"),
        row("other-year", session="20200101-event"),
    ]
    from collections import defaultdict

    counts = defaultdict(int)
    assert [r["id_"] for r in eligible_rows(rows, 2019, counts)] == ["valid"]
    assert dict(counts) == {
        "invalid_or_overlapping_boundaries": 2,
        "duration": 1,
        "lexical_normalization": 1,
    }


def test_frozen_selection_is_order_independent_and_ignores_asr_outputs():
    rows = [
        row("a"),
        row("b", speaker="2"),
        row("c", speaker="3"),
        row("d", session="20190201-event", speaker="4", accent="en_de"),
        row("e", session="20190301-event", speaker="5", accent="en_cs"),
    ]
    selected = select_rows(rows)
    changed = [{**r, "decoded": "bad transcript", "wer": "999"} for r in reversed(rows)]
    assert [r["id_"] for r in selected] == [r["id_"] for r in select_rows(changed)]
    assert len({r["speaker_id"] for r in selected}) == len(selected) == 4
    assert sum(r["accent"] == "en_fr" for r in selected) == 2


def test_range_tar_reads_exact_member_and_rejects_ignored_ranges(monkeypatch):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, payload in [
            ("irrelevant.ogg", b"other"),
            ("original/2019/session.ogg", b"expected"),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    data = stream.getvalue()

    class Response(io.BytesIO):
        status = 206

    def serve(request, timeout):
        first, last = map(int, request.get_header("Range").split("=")[1].split("-"))
        response = Response(data[first : last + 1])
        response.headers = {"Content-Range": f"bytes {first}-{last}/{len(data)}", "ETag": "frozen"}
        return response

    monkeypatch.setattr("urllib.request.urlopen", serve)
    source = RangeTar("https://example.test/archive.tar")
    offset, length = source.locate({"original/2019/session.ogg"})["original/2019/session.ogg"]
    assert source.read(offset, length) == b"expected"
    original = serve

    def ignored(request, timeout):
        response = original(request, timeout)
        response.status = 200
        return response

    monkeypatch.setattr("urllib.request.urlopen", ignored)
    with pytest.raises(ValueError, match="exact archive range"):
        source.read(0, 512)
