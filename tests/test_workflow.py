import typing
from pathlib import Path
from typing import List
from typing import Optional
from uuid import uuid4

import pytest

import lcc.cleaner
import lcc.io
import lcc.segmentizer
import lcc.tokenizer
import lcc.workflow
from lcc.io import FileFormats

if typing.TYPE_CHECKING:  # pragma: no cover
    from pytest_mock import MockerFixture

# ---------------------------------------------------------------------------
# helpers


def test__validate_file_params(tmp_path: Path):
    fn_input_medusa = tmp_path / "sentence.source"
    fn_input_medusa.write_bytes(b"")

    # invalid filenames

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow._validate_file_params(None, None)
    exc_info.match("Argument 'fn_input' is required!")

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow._validate_file_params(str(uuid4()), None)
    exc_info.match("Argument 'fn_output' is required!")

    with pytest.raises(FileNotFoundError) as exc_info:
        lcc.workflow._validate_file_params(str(uuid4()), str(uuid4()))
    exc_info.match(r"Input file '[\da-f-]+' does not exist!")

    # valid call (input exists, output seems to be a filename (string))

    lcc.workflow._validate_file_params(str(fn_input_medusa), str(uuid4()))

    # stdin / stdout check
    with pytest.raises(FileNotFoundError) as exc_info:
        lcc.workflow._validate_file_params("-", str(uuid4()), allow_stdinout=False)
    exc_info.match(r"Input file '-' does not exist!")

    lcc.workflow._validate_file_params(str(fn_input_medusa), "-", allow_stdinout=False)

    lcc.workflow._validate_file_params("-", str(uuid4()), allow_stdinout=True)

    lcc.workflow._validate_file_params("-", "-", allow_stdinout=True)


# ---------------------------------------------------------------------------
# sentence segmentation


@pytest.fixture(scope="module")
def mock_line_segmentizer() -> lcc.segmentizer.AbstractSegmentizer:
    class LineSegmentizer(lcc.segmentizer.AbstractSegmentizer):
        def init(self):
            pass

        def segmentize(self, text: str) -> List[str]:
            if not text:
                return []  # pragma: no cover
            return text.splitlines(keepends=False)

    return LineSegmentizer()


SOURCE_CONTENT1 = """<source><location>http://www.sueddeutsche.de/politik/562/464164/text/</location><date>2023-10-01</date></source>
Welt-Finanzgipfel Regierungschefs suchen Weg aus der Krise
<source><location>http://www.sueddeutsche.de/politik/562/464164/text2/</location><date>2019-09-01</date></source>
Welt-Finanzgipfel Regierungschefs suchen Weg aus der Krise.
Das Auto ist groß.
""".encode()

MEDUSA_CONTENT1_LINES = [
    (
        "Welt-Finanzgipfel Regierungschefs suchen Weg aus der Krise",
        "2023-10-01",
        "http://www.sueddeutsche.de/politik/562/464164/text/",
    ),
    (
        "Welt-Finanzgipfel Regierungschefs suchen Weg aus der Krise.",
        "2019-09-01",
        "http://www.sueddeutsche.de/politik/562/464164/text2/",
    ),
    (
        "Das Auto ist groß.",
        "2019-09-01",
        "http://www.sueddeutsche.de/politik/562/464164/text2/",
    ),
]
MEDUSA_CONTENT1 = (
    "\n".join("\t".join(line) for line in MEDUSA_CONTENT1_LINES) + "\n"
).encode()

WARC_CONTENT2 = (
    b"WARC/1.1\r\nWARC-Date: 2021-07-12T00:00:00Z\r\nWARC-Record-ID: <urn:uuid:42464efe-a018-4dde-bef4-1a252d956b05>\r\nWARC-Type: conversion\r\nWARC-Target-URI: http://www.fortea.us/\r\nWARC-Payload-Digest: sha1:4YR6PDRX5SEXCZDSYD7IHDPZ5DX3MUX4\r\nWARC-Block-Digest: sha1:4YR6PDRX5SEXCZDSYD7IHDPZ5DX3MUX4\r\nContent-Type: text/plain\r\nContent-Length: 430\r\n\r\n"
    b"Posesi\xc3\xb3n y exorcismo | Specialized catholic web about possession and exorcismPosesi\xc3\xb3n y exorcismo\nUna p\xc3\xa1gina cat\xc3\xb3lica acerca del exorcismo\nQu\xc3\xa9 es la influencia demoniaca\nQu\xc3\xa9 es la infestaci\xc3\xb3n de una casa\nQu\xc3\xa9 son los fantasmas\nA qui\xc3\xa9n dirigirse en alguno de estos casos\nQu\xc3\xa9 hacer si no hay exorcista\nQu\xc3\xa9 es lo que no se debe hacer\nLibros cat\xc3\xb3licos sobre el tema demoniaco\nForo sobre exorcismo\nQui\xc3\xa9n dirige esta p\xc3\xa1gina\r\n\r\n"
    b"WARC/1.1\r\nWARC-Date: 2021-07-12T00:00:00Z\r\nWARC-Record-ID: <urn:uuid:bd01699d-f8ed-4976-8caa-b3e9e3b60a09>\r\nWARC-Type: conversion\r\nWARC-Target-URI: http://www.fortea.us/forteaus/page/menu1.htm\r\nWARC-Payload-Digest: sha1:D4YFGQQGXO5DWJNR6GQAKYWR4XV7R2XF\r\nWARC-Block-Digest: sha1:D4YFGQQGXO5DWJNR6GQAKYWR4XV7R2XF\r\nContent-Type: text/plain\r\nContent-Length: 357\r\n\r\n"
    b"Posesi\xc3\xb3n y exorcismo | Specialized catholic web about possession and exorcismPosesi\xc3\xb3n y exorcismo\nUna p\xc3\xa1gina cat\xc3\xb3lica acerca del exorcismo\nLa posesi\xc3\xb3n demoniaca es la acci\xc3\xb3n extraordinaria de un esp\xc3\xadritu maligno sobre un cuerpo humano, hasta el grado de que en los exorcismos ese esp\xc3\xadritu puede mover ese cuerpo a voluntad y hablar a trav\xc3\xa9s de \xc3\xa9l.\r\n\r\n"
)


def test_sentence_segment(
    tmp_path: Path, mock_line_segmentizer: lcc.segmentizer.AbstractSegmentizer
):
    fn_input = tmp_path / "input.source"
    fn_input.write_bytes(SOURCE_CONTENT1)

    fn_output_source = tmp_path / "output.source"
    lcc.workflow.sentence_segment(
        str(fn_input), str(fn_output_source), segmentizer=mock_line_segmentizer
    )
    assert fn_output_source.read_bytes() == SOURCE_CONTENT1

    fn_output_medusa = tmp_path / "output.medusa"
    lcc.workflow.sentence_segment(
        str(fn_input), str(fn_output_medusa), segmentizer=mock_line_segmentizer
    )
    assert fn_output_medusa.read_bytes() == MEDUSA_CONTENT1

    # overwrite file format (parameter > extension)
    fn_output_medusa_s = tmp_path / "output2.source"
    lcc.workflow.sentence_segment(
        str(fn_input),
        str(fn_output_medusa_s),
        fmt_output=FileFormats.MEDUSA,
        segmentizer=mock_line_segmentizer,
    )
    assert fn_output_medusa_s.read_bytes() == MEDUSA_CONTENT1

    # invalid filenames

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.sentence_segment(None, None, segmentizer=mock_line_segmentizer)
    exc_info.match("Argument 'fn_input' is required!")

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.sentence_segment(
            str(uuid4()), None, segmentizer=mock_line_segmentizer
        )
    exc_info.match("Argument 'fn_output' is required!")

    with pytest.raises(FileNotFoundError) as exc_info:
        lcc.workflow.sentence_segment(
            str(uuid4()), str(uuid4()), segmentizer=mock_line_segmentizer
        )
    exc_info.match(r"Input file '[\da-f-]+' does not exist!")

    # no segmentizer

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.sentence_segment(str(fn_input), str(fn_output_source))
    exc_info.match(
        "Value for 'dn_segmentizer_resources' is required if 'segmentizer' is None!"
    )

    # TODO: test default segmentizer creation

    # invalid file formats

    fn_input_empty_warc = tmp_path / f"{uuid4()!s}.warc"
    fn_input_empty_warc.write_text("")
    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.sentence_segment(
            str(fn_input_empty_warc), str(uuid4()), segmentizer=mock_line_segmentizer
        )
    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.sentence_segment(
            str(fn_input), f"{uuid4()!s}.xyz", segmentizer=mock_line_segmentizer
        )


@pytest.mark.skip(reason="todo, + skipif warcio installed")
def test_sentence_segment_warc():
    pass


# ---------------------------------------------------------------------------
# sentence cleaning


def test_clean_sentences(tmp_path: Path):
    class AllOkSentenceCleaner(lcc.cleaner.SentenceCleaner):
        def __init__(self) -> None:
            pass

        def filter_sentence(
            self, sentence: str, do_replacements: bool = True
        ) -> Optional[str]:
            return sentence

    class AllBadSentenceCleaner(lcc.cleaner.SentenceCleaner):
        def __init__(self) -> None:
            pass

        def filter_sentence(
            self, sentence: str, do_replacements: bool = True
        ) -> Optional[str]:
            return None

    mock_cleaner: lcc.cleaner.SentenceCleaner = AllOkSentenceCleaner()

    fn_input_source = tmp_path / "input.source"
    fn_input_source.write_bytes(SOURCE_CONTENT1)
    fn_input_medusa = tmp_path / "input.medusa"
    fn_input_medusa.write_bytes(MEDUSA_CONTENT1)

    fn_output_source = tmp_path / "output.source"
    lcc.workflow.clean_sentences(
        str(fn_input_source), str(fn_output_source), cleaner=mock_cleaner
    )
    assert fn_output_source.read_bytes() == SOURCE_CONTENT1

    fn_output_medusa = tmp_path / "output.medusa"
    lcc.workflow.clean_sentences(
        str(fn_input_medusa), str(fn_output_medusa), cleaner=mock_cleaner
    )
    assert fn_output_medusa.read_bytes() == MEDUSA_CONTENT1

    lcc.workflow.clean_sentences(
        str(fn_input_source), str(fn_output_medusa), cleaner=mock_cleaner
    )
    assert fn_output_medusa.read_bytes() == MEDUSA_CONTENT1

    # invalid format combination
    with pytest.raises(ValueError):
        lcc.workflow.clean_sentences(
            str(fn_input_medusa), str(fn_output_source), cleaner=mock_cleaner
        )

    # invalid filenames

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.clean_sentences(None, None, cleaner=mock_cleaner)
    exc_info.match("Argument 'fn_input' is required!")

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.clean_sentences(str(uuid4()), None, cleaner=mock_cleaner)
    exc_info.match("Argument 'fn_output' is required!")

    with pytest.raises(FileNotFoundError) as exc_info:
        lcc.workflow.clean_sentences(str(uuid4()), str(uuid4()), cleaner=mock_cleaner)
    exc_info.match(r"Input file '[\da-f-]+' does not exist!")

    # no cleaner

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.clean_sentences(
            str(fn_input_source), str(fn_output_source), cleaner=None
        )
    exc_info.match("Argument 'cleaner' is required!")

    # invalid file formats

    fn_input_empty_warc = tmp_path / f"{uuid4()!s}.warc"
    fn_input_empty_warc.write_text("")
    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.clean_sentences(
            str(fn_input_empty_warc), str(uuid4()), cleaner=mock_cleaner
        )
    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.clean_sentences(
            str(fn_input_source), f"{uuid4()!s}.xyz", cleaner=mock_cleaner
        )

    # empty file (?) if all cleaned away

    mock_cleaner = AllBadSentenceCleaner()
    fn_output_empty = tmp_path / f"{uuid4()!s}.source"
    lcc.workflow.clean_sentences(
        str(fn_input_source), str(fn_output_empty), cleaner=mock_cleaner
    )
    assert all(
        lcc.io.SourceDocMetadata.is_header(line)
        for line in fn_output_empty.read_text().splitlines(keepends=False)
        if line
    )

    fn_output_empty = tmp_path / f"{uuid4()!s}.medusa"
    lcc.workflow.clean_sentences(
        str(fn_input_source),
        str(fn_output_empty),
        cleaner=mock_cleaner,
    )
    assert fn_output_empty.stat().st_size == 0
    lcc.workflow.clean_sentences(
        str(fn_input_medusa),
        str(fn_output_empty),
        cleaner=mock_cleaner,
    )
    assert fn_output_empty.stat().st_size == 0


@pytest.mark.skip(reason="todo, + skipif warcio installed")
def test_clean_sentences_warc():
    pass


# ---------------------------------------------------------------------------
# tokenization


MEDUSA_CONTENT1_DUMMY_TOKENIZED = (
    "\n".join("\t".join((line[0] + "abc", *line[1:])) for line in MEDUSA_CONTENT1_LINES)
    + "\n"
).encode()


@pytest.fixture(scope="module")
def mock_sentence_tokenizer() -> lcc.tokenizer.AbstractWordTokenizer:
    class NoOpTokenizer(lcc.tokenizer.AbstractWordTokenizer):
        def init(self):
            pass

        def execute(self, line: str) -> str:
            return line + "abc"

    return NoOpTokenizer()


def test_tokenize_sentence(
    tmp_path: Path, mock_sentence_tokenizer: lcc.tokenizer.AbstractWordTokenizer
):
    fn_input = tmp_path / "input.medusa"
    fn_input.write_bytes(MEDUSA_CONTENT1)

    fn_output = tmp_path / "output.medusa"
    lcc.workflow.tokenize_sentence(
        str(fn_input), str(fn_output), tokenizer=mock_sentence_tokenizer
    )
    assert fn_output.read_bytes() == MEDUSA_CONTENT1_DUMMY_TOKENIZED

    # invalid filenames

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.tokenize_sentence(None, None, tokenizer=mock_sentence_tokenizer)
    exc_info.match("Argument 'fn_input' is required!")

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.tokenize_sentence(
            str(uuid4()), None, tokenizer=mock_sentence_tokenizer
        )
    exc_info.match("Argument 'fn_output' is required!")

    with pytest.raises(FileNotFoundError) as exc_info:
        lcc.workflow.tokenize_sentence(
            str(uuid4()), str(uuid4()), tokenizer=mock_sentence_tokenizer
        )
    exc_info.match(r"Input file '[\da-f-]+' does not exist!")

    # no tokenizer

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.tokenize_sentence(str(fn_input), str(fn_output))
    exc_info.match(
        "Value for 'dn_tokenizer_resources' is required if 'tokenizer' is None!"
    )

    # TODO: test default tokenizer creation

    # invalid file formats

    fn_input_empty_warc = tmp_path / f"{uuid4()!s}.warc"
    fn_input_empty_warc.write_text("")
    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.tokenize_sentence(
            str(fn_input_empty_warc), str(uuid4()), tokenizer=mock_sentence_tokenizer
        )
    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.tokenize_sentence(
            str(fn_input), f"{uuid4()!s}.warc", tokenizer=mock_sentence_tokenizer
        )


# ---------------------------------------------------------------------------
# format conversion


def test_convert_source_to_medusa(tmp_path: Path):
    fn_input_source = tmp_path / "input.source"
    fn_input_source.write_bytes(SOURCE_CONTENT1)
    fn_output_medusa = tmp_path / "output.medusa"

    lcc.workflow.convert_source_to_medusa(str(fn_input_source), str(fn_output_medusa))
    assert fn_output_medusa.read_bytes() == MEDUSA_CONTENT1

    # invalid filenames

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.convert_source_to_medusa(None, None)
    exc_info.match("Argument 'fn_input' is required!")

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.convert_source_to_medusa(str(uuid4()), None)
    exc_info.match("Argument 'fn_output' is required!")

    with pytest.raises(FileNotFoundError) as exc_info:
        lcc.workflow.convert_source_to_medusa(str(uuid4()), str(uuid4()))
    exc_info.match(r"Input file '[\da-f-]+' does not exist!")


SOURCE_CONTENT1_JSONL = (
    b'{"location": "http://www.sueddeutsche.de/politik/562/464164/text/", "date": "2023-10-01", "content": "Welt-Finanzgipfel Regierungschefs suchen Weg aus der Krise\\n"}\n'
    b'{"location": "http://www.sueddeutsche.de/politik/562/464164/text2/", "date": "2019-09-01", "content": "Welt-Finanzgipfel Regierungschefs suchen Weg aus der Krise.\\nDas Auto ist gro\\u00df.\\n"}\n'
)


def test_convert_source_to_jsonl(tmp_path: Path):
    fn_input_source = tmp_path / "docs.source"
    fn_input_source.write_bytes(SOURCE_CONTENT1)
    fn_output_jsonl = tmp_path / "docs.jsonl"

    lcc.workflow.convert_source_to_jsonl(str(fn_input_source), str(fn_output_jsonl))
    assert fn_output_jsonl.read_bytes() == SOURCE_CONTENT1_JSONL

    # invalid filenames

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.convert_source_to_jsonl(None, None)
    exc_info.match("Argument 'fn_input' is required!")

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.convert_source_to_jsonl(str(uuid4()), None)
    exc_info.match("Argument 'fn_output' is required!")

    with pytest.raises(FileNotFoundError) as exc_info:
        lcc.workflow.convert_source_to_jsonl(str(uuid4()), str(uuid4()))
    exc_info.match(r"Input file '[\da-f-]+' does not exist!")


MEDUSA_CONTENT1_JSONL = (
    b'{"location": "http://www.sueddeutsche.de/politik/562/464164/text/", "date": "2023-10-01", "sentence": "Welt-Finanzgipfel Regierungschefs suchen Weg aus der Krise"}\n'
    b'{"location": "http://www.sueddeutsche.de/politik/562/464164/text2/", "date": "2019-09-01", "sentence": "Welt-Finanzgipfel Regierungschefs suchen Weg aus der Krise."}\n'
    b'{"location": "http://www.sueddeutsche.de/politik/562/464164/text2/", "date": "2019-09-01", "sentence": "Das Auto ist gro\\u00df."}\n'
)


def test_convert_medusa_to_jsonl(tmp_path: Path):
    fn_input_medusa = tmp_path / "sentence.source"
    fn_input_medusa.write_bytes(MEDUSA_CONTENT1)
    fn_output_jsonl = tmp_path / "sentence.jsonl"

    lcc.workflow.convert_medusa_to_jsonl(str(fn_input_medusa), str(fn_output_jsonl))
    assert fn_output_jsonl.read_bytes() == MEDUSA_CONTENT1_JSONL

    # invalid filenames

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.convert_medusa_to_jsonl(None, None)
    exc_info.match("Argument 'fn_input' is required!")

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.convert_medusa_to_jsonl(str(uuid4()), None)
    exc_info.match("Argument 'fn_output' is required!")

    with pytest.raises(FileNotFoundError) as exc_info:
        lcc.workflow.convert_medusa_to_jsonl(str(uuid4()), str(uuid4()))
    exc_info.match(r"Input file '[\da-f-]+' does not exist!")


# TODO: test warc conversion
@pytest.mark.skip(reason="todo, + skipif warcio installed")
def test_convert_source_to_warc():
    pass


@pytest.mark.skip(reason="todo, + skipif warcio installed")
def test_convert_warc_to_source():
    pass


@pytest.mark.skip(reason="todo, + skipif warcio installed")
def test_convert_warc_to_jsonl():
    pass


# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="not yet implemented")
def test_split_source_file(tmp_path: Path):
    pass


@pytest.mark.xfail(reason="not yet implemented")
def test_merge_source_files(tmp_path: Path):
    pass


@pytest.mark.xfail(reason="not yet implemented")
def test_slice_source_file(tmp_path: Path):
    pass


@pytest.mark.xfail(reason="not yet implemented")
def test__validate_slice_params():
    pass


# ---------------------------------------------------------------------------


def test_compute_docs_stats_heuristic(tmp_path: Path, mocker: "MockerFixture"):
    fn_source = tmp_path / "input.source"
    fn_source.write_bytes(SOURCE_CONTENT1)

    mock_stats = mocker.patch("lcc.workflow.lcc.stats.compute_docs_stats_heuristic")

    mock_stats.return_value = 2
    assert 2 == lcc.workflow.compute_docs_stats_heuristic(str(fn_source))
    assert mock_stats.call_count == 1

    mock_stats.reset_mock()

    # invalid file formats

    fn_input_invalid = tmp_path / f"{uuid4()!s}.xyz"
    fn_input_invalid.write_text("")
    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.compute_docs_stats_heuristic(str(fn_input_invalid))
    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.compute_docs_stats_heuristic(
            str(fn_input_invalid), fmt_input=FileFormats.MEDUSA
        )
    with pytest.raises(ValueError) as exc_info:
        lcc.workflow.compute_docs_stats_heuristic("-")

    assert mock_stats.call_count == 0

    mock_stats.return_value = 1
    assert 1 == lcc.workflow.compute_docs_stats_heuristic(
        str(fn_input_invalid), fmt_input=FileFormats.SOURCE
    )


def test_compute_medusa_stats_heuristic(tmp_path: Path):
    fn_input = tmp_path / "input.medusa"
    fn_input.write_bytes(
        b"abc def.\tdate\tloc\n" b"cde a b d\tdate\tloc\n" b"xxx\tdate2\tloc\n"
    )

    ms = lcc.workflow.compute_medusa_stats_heuristic(fn_input, count_sources=True)
    assert ms.sources == 2
    assert ms.lines == 3
    assert ms.tokens == 7
    assert ms.chars == 20

    ms = lcc.workflow.compute_medusa_stats_heuristic(fn_input, count_sources=False)
    assert ms.sources is None
    assert ms.lines == 3
    assert ms.tokens == 7
    assert ms.chars == 20


# ---------------------------------------------------------------------------
