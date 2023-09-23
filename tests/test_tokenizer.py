from pathlib import Path

import pytest

import lcc.tokenizer

# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def tokenizer(resource_path: Path) -> lcc.tokenizer.CharacterBasedWordTokenizerImproved:
    dn_tok = resource_path / "tokenizer"

    tokenizer = lcc.tokenizer.CharacterBasedWordTokenizerImproved(
        strAbbrevListFile=str(dn_tok / "default.abbrev"),
        TOKENISATION_CHARACTERS_FILE_NAME=str(dn_tok / "100-wn-all.txt"),
        fixedTokensFile=str(dn_tok / "fixed_tokens.txt"),
        strCharacterActionsFile=str(dn_tok / "tokenization_character_actions.txt"),
    )

    return tokenizer


# ---------------------------------------------------------------------------


def test_simple(tokenizer: lcc.tokenizer.CharacterBasedWordTokenizerImproved):
    assert "Dies ist ein Test ." == tokenizer.execute("Dies ist ein Test.")


def test_simple_2(tokenizer: lcc.tokenizer.CharacterBasedWordTokenizerImproved):
    sentences = [
        """The car is red.""",
        """I arrived in Salt River and parked near its signature traffic circle.""",
        """The 3.5" disk drive is now broken.""",
        """2012 erzielte Endeavour Silver einen Gewinn von 42 Mio. $.""",
        """101 der 630 Verkehrstoten waren ausländische Staatsangehörige, das sind 16 %.""",
        """Auf Dinge, die schön sind, aber nicht systemrelevant, wird man lange verzichten.«""",
        """Auf Dinge, die schön sind, aber nicht systemrelevant, wird man lange verzichten.»""",
    ]
    sentences_tok = [
        """The car is red .""",
        """I arrived in Salt River and parked near its signature traffic circle .""",
        """The 3.5 " disk drive is now broken .""",
        """2012 erzielte Endeavour Silver einen Gewinn von 42 Mio. $ .""",
        """101 der 630 Verkehrstoten waren ausländische Staatsangehörige , das sind 16 % .""",
        """Auf Dinge , die schön sind , aber nicht systemrelevant , wird man lange verzichten .«""",
        """Auf Dinge , die schön sind , aber nicht systemrelevant , wird man lange verzichten .»""",
    ]

    for exno, (sentence, sentence_tok) in enumerate(zip(sentences, sentences_tok)):
        assert sentence_tok == tokenizer.execute(sentence), f"example: {exno}"


# ---------------------------------------------------------------------------


def test_sentence_alignment():
    sentence = """Boolean values can be specified as on, off, true, false, 1, or 0 and are case-insensitive."""
    sentence_tok = """Boolean values can be specified as on , off , true , false , 1 , or 0 and are case-insensitive ."""

    aligned_sentence = lcc.tokenizer.AlignedSentence(
        raw=sentence, tokenized=sentence_tok
    )
    # fmt: off
    assert aligned_sentence.tokens() == ["Boolean", "values", "can", "be", "specified", "as", "on", ",", "off", ",", "true", ",", "false", ",", "1", ",", "or", "0", "and", "are", "case-insensitive", "."]
    assert aligned_sentence.tokens_glue() == [False, False, False, False, False, False, True, False, True, False, True, False, True, False, True, False, False, False, False, False, True]
    # fmt: on


def test_sentence_alignment_real(
    tokenizer: lcc.tokenizer.CharacterBasedWordTokenizerImproved,
):
    sentences = [
        """The car is red.""",
        """I arrived in Salt River and parked near its signature traffic circle.""",
        """The 3.5" disk drive is now broken.""",
        """2012 erzielte Endeavour Silver einen Gewinn von 42 Mio. $.""",
        """101 der 630 Verkehrstoten waren ausländische Staatsangehörige, das sind 16 %.""",
        """Auf Dinge, die schön sind, aber nicht systemrelevant, wird man lange verzichten.«""",
        """Auf Dinge, die schön sind, aber nicht systemrelevant, wird man lange verzichten.»""",
    ]

    for sentence in sentences:
        sentence_tok = tokenizer.execute(sentence)
        aligned_sentence = lcc.tokenizer.AlignedSentence(
            raw=sentence, tokenized=sentence_tok
        )
        # now compute mappings
        aligned_sentence.tokens_glue()


@pytest.mark.skip(
    reason="testing, this case should not really happen with tokenizer in/output"
)
def test_sentence_alignment_subs(
    resource_path: Path, tokenizer: lcc.tokenizer.CharacterBasedWordTokenizerImproved
):  # pragma: no cover
    import lcc.cleaner

    replacer = lcc.cleaner.StringReplacements(
        str(resource_path / "cleaner" / "StringReplacements.list")
    )

    sentences = [
        """The car is red.&para;""",
        'value["temp"]="The car is red ";',
    ]

    for sentence in sentences:
        sentence_repl = replacer.replace(sentence)
        sentence_tok = tokenizer.execute(sentence_repl)
        print("raw", sentence)
        print("rep", sentence_repl)
        print("tok", sentence_tok)

        aligned_sentence = lcc.tokenizer.AlignedSentence(
            raw=sentence, tokenized=sentence_tok
        )
        print(aligned_sentence.alignment_indices())


# ---------------------------------------------------------------------------
