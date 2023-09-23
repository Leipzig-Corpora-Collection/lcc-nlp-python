from pathlib import Path

import pytest

import lcc.segmentizer

# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def segmentizer(resource_path: Path) -> lcc.segmentizer.LineAwareSegmentizer:
    dn_seg = resource_path / "segmentizer"

    # segmentizer = lcc.segmentizer.LineAwareSegmentizer.create_default(str(dn_seg))
    segmentizer = lcc.segmentizer.LineAwareSegmentizer(
        fn_sentence_boundaries=str(dn_seg / "boundariesFile.txt"),
        fn_pre_boundary_rules=str(dn_seg / "preRules.txt"),
        fn_pre_boundaries_list=str(dn_seg / "preList.txt"),
        fn_post_boundary_rules=str(dn_seg / "postRules.txt"),
        fn_post_boundaries_list=str(dn_seg / "postList.txt"),
        encoding="utf-8",
        is_auto_uppercase_first_letter_pre_list=True,
        use_carriage_return_as_boundary=True,
        use_empty_line_as_boundary=True,
        is_trim_mode=True,
    )

    return segmentizer


# ---------------------------------------------------------------------------


def test_deu(segmentizer: lcc.segmentizer.LineAwareSegmentizer):
    text_input = """Ein Monstrum in jeder Hinsicht. 2.440 Seiten dick, fast 20 Kilo schwer, mehr als 1.500 Rezepturen.
Ein Musterhemd in einem dunklen Blauton gibt es schon. 2500 Shirts sind bestellt.
Dies ist die Überschrift

Die vorige Überschrift stand in einem eigenen Absatz und sollte daher als eigenständiger Satz identifiziert werden. Nun folgt ein weiterer Satz. Und noch einer, damit es ein Absatz wird.

Es geschah im Jahre 2009. Durch seinen 21. Gewinn in der 17. Grand-Slam-Begegnung nacheinander komplettierte der Weltranglisten-Erste als 7. Spieler der Geschichte seine Grand-Slam-Sammlung.

Mister Smith (sein Sohn!) kam herein.

Der König um den es geht ist Louis XVII. Dieser König war ein ganz gemeiner Kerl. Johannes Paul II lebte bis 2005. Dann kam der nächste Papst.
"""

    text_output = """Ein Monstrum in jeder Hinsicht.
2.440 Seiten dick, fast 20 Kilo schwer, mehr als 1.500 Rezepturen.
Ein Musterhemd in einem dunklen Blauton gibt es schon.
2500 Shirts sind bestellt.
Dies ist die Überschrift
Die vorige Überschrift stand in einem eigenen Absatz und sollte daher als eigenständiger Satz identifiziert werden.
Nun folgt ein weiterer Satz.
Und noch einer, damit es ein Absatz wird.
Es geschah im Jahre 2009.
Durch seinen 21. Gewinn in der 17. Grand-Slam-Begegnung nacheinander komplettierte der Weltranglisten-Erste als 7. Spieler der Geschichte seine Grand-Slam-Sammlung.
Mister Smith (sein Sohn!) kam herein.
Der König um den es geht ist Louis XVII.
Dieser König war ein ganz gemeiner Kerl.
Johannes Paul II lebte bis 2005.
Dann kam der nächste Papst.
""".splitlines(
        keepends=False
    )

    sentences = segmentizer.segmentize(text_input)

    assert text_output == sentences


def test_eng(segmentizer: lcc.segmentizer.LineAwareSegmentizer):
    text_input = """This is an example sentence. This is another one.
"""

    text_output = """This is an example sentence.
This is another one.
""".splitlines(
        keepends=False
    )

    sentences = segmentizer.segmentize(text_input)

    assert text_output == sentences


# ---------------------------------------------------------------------------
