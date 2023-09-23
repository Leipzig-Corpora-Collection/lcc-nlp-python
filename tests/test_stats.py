from pathlib import Path

import pytest

import lcc.io
import lcc.stats

# ---------------------------------------------------------------------------


def test_DocStats():
    pass


def test_MedusaStats():
    ms = lcc.stats.MedusaStats(1, 2, 3, None)

    assert "\t1\t2\t3" == ms.to_tsv()
    assert """{"lines": 1, "tokens": 2, "chars": 3, "sources": null}""" == ms.to_json()
    assert (
        """{\n  "lines": 1,\n  "tokens": 2,\n  "chars": 3,\n  "sources": null\n}"""
        == ms.to_json(pretty=True)
    )

    ms = lcc.stats.MedusaStats(1, 2, 3, 0)

    assert "0\t1\t2\t3" == ms.to_tsv()
    assert """{"lines": 1, "tokens": 2, "chars": 3, "sources": 0}""" == ms.to_json()
    assert (
        """{\n  "lines": 1,\n  "tokens": 2,\n  "chars": 3,\n  "sources": 0\n}"""
        == ms.to_json(pretty=True)
    )


def test_compute_docs_stats_heuristic():
    meta1 = lcc.io.DocMetadata("loc", "date")
    meta2 = lcc.io.DocMetadata("loc", "date2")
    doc1 = lcc.io.DocAndMeta(meta1, "abc def.")
    doc2 = lcc.io.DocAndMeta(meta1, b"cde a b d")
    doc3 = lcc.io.DocAndMeta(meta2, "xxx")

    doc_iter = [doc1, doc2, doc3]

    stats = lcc.stats.compute_docs_stats_heuristic(doc_iter)
    assert stats.docs == 3
    assert stats.lines == 3
    assert stats.tokens == 7
    assert stats.chars == 20
    assert stats.sources == 1
    assert stats.dates == 2
    assert stats.source_dates == 2

    stats = lcc.stats.compute_docs_stats_heuristic(doc_iter, hasher_fn=None)
    assert stats.docs == 3
    assert stats.lines == 3
    assert stats.tokens == 7
    assert stats.chars == 20
    assert stats.sources is None
    assert stats.dates is None
    assert stats.source_dates is None


def test_compute_medusa_stats_heuristic():
    meta1 = lcc.io.DocMetadata("loc", "date")
    meta2 = lcc.io.DocMetadata("loc", "date2")
    doc1 = lcc.io.SentenceAndMeta(meta1, "abc def.")
    doc2 = lcc.io.SentencesAndMeta(meta1, ["cde a b d"])
    doc3 = lcc.io.SentenceAndMeta(meta2, "xxx")

    doc_iter = [doc1, doc2, doc3]

    ms = lcc.stats.compute_sentences_stats_heuristic(doc_iter, count_sources=True)
    assert ms.sources == 2
    assert ms.lines == 3
    assert ms.tokens == 7
    assert ms.chars == 20

    ms = lcc.stats.compute_sentences_stats_heuristic(doc_iter, count_sources=False)
    assert ms.sources is None
    assert ms.lines == 3
    assert ms.tokens == 7
    assert ms.chars == 20


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
