from pathlib import Path
from uuid import uuid4

import pytest

import lcc.cooccurrence.medusa


# ---------------------------------------------------------------------------
# word numbers


def test_load_known_word_numbers(resource_path: Path, tmp_path: Path):
    fn_known_wns = resource_path / "tokenizer" / "100-wn-all.txt"

    next_wn, word2nr = lcc.cooccurrence.medusa.load_known_word_numbers(fn_known_wns)
    assert next_wn == 101
    assert set(range(1, 101)) ^ set(word2nr.values()) == {49, 81}

    next_wn, word2nr = lcc.cooccurrence.medusa.load_known_word_numbers(
        tmp_path / str(uuid4())
    )
    assert next_wn == 1
    assert not word2nr


# ---------------------------------------------------------------------------
