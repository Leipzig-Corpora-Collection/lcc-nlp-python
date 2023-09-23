import io
import itertools
import typing
from pathlib import Path
from typing import Literal
from typing import Optional

import pytest

import lcc.io
from lcc.io import FileFormats

if typing.TYPE_CHECKING:  # pragma: no cover
    from pytest import CaptureFixture
    from pytest import LogCaptureFixture
    from pytest import MonkeyPatch
    from pytest_mock import MockerFixture

# ---------------------------------------------------------------------------


def test__find_files(tmp_path: Path):
    # setup tree
    (tmp_path / "root.py").write_bytes(b"")
    (tmp_path / "root.txt").write_bytes(b"")
    (tmp_path / "folder1").mkdir()
    (tmp_path / "folder1" / "file1.py").write_bytes(b"")
    (tmp_path / "folder1" / "file2.txt").write_bytes(b"")
    (tmp_path / "folder2").mkdir()
    (tmp_path / "folder2" / "folder21").mkdir()
    (tmp_path / "folder2" / "folder21" / "file3.txt").write_bytes(b"")

    assert set(lcc.io._find_files(tmp_path)) == {
        tmp_path / "root.py",
        tmp_path / "root.txt",
    }
    assert len(lcc.io._find_files(tmp_path, recursive=True)) == 5
    assert set(lcc.io._find_files(tmp_path, "*.py", recursive=True)) == {
        tmp_path / "root.py",
        tmp_path / "folder1" / "file1.py",
    }
    assert len(lcc.io._find_files(tmp_path, "*.txt", recursive=True)) == 3
    assert lcc.io._find_files(tmp_path, "*.txt") == [tmp_path / "root.txt"]
    assert lcc.io._find_files(tmp_path / "invalid") == []

    # error, use recursive keyword instead
    assert lcc.io._find_files(tmp_path, "**/*.txt") == []
    # error, we do not want to allow parent access
    assert lcc.io._find_files(tmp_path, "../*.txt") == []
    assert lcc.io._find_files(tmp_path, "*/..") == []


def test__generate_filenames():
    # start/end
    assert list(itertools.islice(lcc.io._generate_filenames("test"), 3)) == [
        "test_001",
        "test_002",
        "test_003",
    ]
    assert list(lcc.io._generate_filenames("abc.txt", stop=-1)) == ["abc_001.txt"]
    assert list(lcc.io._generate_filenames("abc.txt", start=1, stop=-1)) == [
        "abc_001.txt"
    ]
    assert list(lcc.io._generate_filenames("abc.txt", start=1, stop=2)) == [
        "abc_001.txt",
        "abc_002.txt",
    ]
    assert list(lcc.io._generate_filenames("abc.txt", start=100, stop=2)) == [
        "abc_100.txt"
    ]

    # padlen
    assert list(lcc.io._generate_filenames("abc.txt", start=9, stop=10, padlen=-1)) == [
        "abc_9.txt",
        "abc_10.txt",
    ]
    assert list(lcc.io._generate_filenames("abc.txt", stop=1, padlen=None)) == [
        "abc_1.txt"
    ]
    assert list(
        lcc.io._generate_filenames("x", start=9, stop=10, padlen=10, sep="+")
    ) == [
        "x+0000000009",
        "x+0000000010",
    ]
    assert list(lcc.io._generate_filenames("x...t", start=9, stop=1, sep="+")) == [
        "x..+009.t"
    ]

    # sep
    assert list(lcc.io._generate_filenames("T", stop=1, sep="")) == ["T001"]
    assert list(lcc.io._generate_filenames("T", stop=1, sep=None)) == ["T001"]

    # where
    assert list(
        lcc.io._generate_filenames("x...t", stop=1, sep="=", where="prefix")
    ) == ["001=x...t"]
    assert list(
        lcc.io._generate_filenames("x...t", stop=1, sep="=", where="suffix")
    ) == ["x...t=001"]
    assert list(
        lcc.io._generate_filenames("x...t", stop=1, sep="=", where="infix")
    ) == ["x..=001.t"]
    assert list(lcc.io._generate_filenames("x", stop=1, sep="=", where="infix")) == [
        "x=001"
    ]
    assert list(lcc.io._generate_filenames("x", stop=1, sep="=", where="suffix")) == [
        "x=001"
    ]
    assert list(lcc.io._generate_filenames("x", stop=1, sep="=", where="prefix")) == [
        "001=x"
    ]

    # errors
    # not using the generator does not seem to raise an error!
    lcc.io._generate_filenames(None)
    # errors
    with pytest.raises(ValueError):
        next(lcc.io._generate_filenames(None))
    with pytest.raises(ValueError):
        next(lcc.io._generate_filenames("name", where=None))


def test__open_stream(tmp_path: Path, capsys: "CaptureFixture"):
    fn_in = tmp_path / "input.file"
    fn_out = tmp_path / "output.file"

    fn_in.write_bytes(b"this is an example\nnext line")

    with lcc.io._open_stream(fn_in, "r") as fp:
        lines = fp.readlines()
        assert lines == ["this is an example\n", "next line"]

    with lcc.io._open_stream(fn_in, "rb") as fp:
        content = fp.read()
        assert content == b"this is an example\nnext line"

    with lcc.io._open_stream(fn_out, "wb") as fp:
        fp.write(b"this is an examp")
        fp.write(b"le\nnext line")
    assert fn_out.read_bytes() == b"this is an examp" b"le\nnext line"

    with lcc.io._open_stream(fn_out, "w") as fp:
        fp.write("this is another nexamp")
        fp.write("le\nlast line\n")
    assert fn_out.read_bytes() == b"this is another nexamp" b"le\nlast line\n"


def test__open_stream_stdin(set_stdin_content):
    set_stdin_content(b"some random junk\non many\n lines")
    with lcc.io._open_stream("-", "rb") as fp:
        content = fp.read()
        assert content == b"some random junk\non many\n lines"

    set_stdin_content(b"some random junk\non many\n lines")
    with lcc.io._open_stream("-", "rb") as fp:
        lines = fp.readlines()
        assert lines == [b"some random junk\n", b"on many\n", b" lines"]

    set_stdin_content(b"some random junk\non many\n lines")
    with lcc.io._open_stream("-", "r") as fp:
        lines = fp.readlines()
        assert lines == ["some random junk\n", "on many\n", " lines"]


def test__open_stream_stdout(capsysbinary: "CaptureFixture"):
    with lcc.io._open_stream("-", "w") as fp:
        fp.write("abc")
        fp.write("\nfdjklsafdsa\nak")
    captured = capsysbinary.readouterr()
    assert captured.out == b"abc\nfdjklsafdsa\nak"

    with lcc.io._open_stream("-", "wb") as fp:
        fp.write(b"abc")
        fp.write(b"\nfdjklsafdsa\nak2")
    captured = capsysbinary.readouterr()
    assert captured.out == b"abc\nfdjklsafdsa\nak2"


# ---------------------------------------------------------------------------
# file formats


def test_FileFormats_detect_format_by_ext():
    # invalid input
    assert FileFormats.detect_format_by_ext(None) is None
    assert FileFormats.detect_format_by_ext("") is None
    # no extension
    assert FileFormats.detect_format_by_ext("no_extension") is None

    # unknown
    assert FileFormats.detect_format_by_ext("unknown.extension") is None

    # known extensions
    assert FileFormats.detect_format_by_ext("file.source") is FileFormats.SOURCE
    assert FileFormats.detect_format_by_ext("file.medusa") is FileFormats.MEDUSA

    assert FileFormats.detect_format_by_ext("file.warc") is FileFormats.WARC
    assert FileFormats.detect_format_by_ext("file.arc") is FileFormats.WARC
    assert FileFormats.detect_format_by_ext("file.wet") is FileFormats.WARC

    assert FileFormats.detect_format_by_ext("file.json") is None
    assert FileFormats.detect_format_by_ext("file.jsonl") is FileFormats.JSONL
    assert FileFormats.detect_format_by_ext("file.jsonlines") is FileFormats.JSONL


def test__extract_file_formats():
    def process1(fmt: Literal[FileFormats.SOURCE]):
        pass

    def process1o(fmt: Optional[Literal[FileFormats.SOURCE]]):
        pass

    def process2o(
        fn: str,
        fmt: Optional[
            Literal[FileFormats.SOURCE, FileFormats.MEDUSA, FileFormats.JSONL]
        ],
        some_other_param: Optional[int] = 2,
    ):
        pass

    p1types = lcc.io._extract_file_formats(process1, "fmt")
    assert p1types == (FileFormats.SOURCE,)

    p1otypes = lcc.io._extract_file_formats(process1o, "fmt")
    assert p1otypes == (FileFormats.SOURCE,)

    p2otypes = lcc.io._extract_file_formats(process2o, "fmt")
    assert p2otypes == (FileFormats.SOURCE, FileFormats.MEDUSA, FileFormats.JSONL)

    # unknown parameter
    p1types2 = lcc.io._extract_file_formats(process1, "invalid")
    assert p1types2 == tuple()

    # not a function
    p1types3 = lcc.io._extract_file_formats(None, "invalid")
    assert p1types3 == tuple()


def test__validate_file_format():
    # non-optional file format parameter

    def process1(fmt: Literal[FileFormats.SOURCE]):
        pass

    assert (
        lcc.workflow._validate_file_format(
            "file.source", FileFormats.SOURCE, process1, "fmt", "file"
        )
        is FileFormats.SOURCE
    )

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow._validate_file_format("file.source", None, process1, "fmt", "file")
    exc_info.match("Argument 'fmt' is required to be not None!")

    # optional file format parameter

    def process1o(fmt: Optional[Literal[FileFormats.SOURCE]]):
        pass

    assert (
        lcc.workflow._validate_file_format(
            "file.source", None, process1o, "fmt", "file"
        )
        is FileFormats.SOURCE
    )

    # multiple file formats

    def process2o(
        fn: str,
        fmt: Optional[
            Literal[FileFormats.SOURCE, FileFormats.MEDUSA, FileFormats.JSONL]
        ],
        some_other_param: Optional[int] = 2,
    ):
        pass

    assert (
        lcc.workflow._validate_file_format(
            "file.source", None, process2o, "fmt", "file"
        )
        is FileFormats.SOURCE
    )
    assert (
        lcc.workflow._validate_file_format(
            "file.medusa", None, process2o, "fmt", "file"
        )
        is FileFormats.MEDUSA
    )
    assert (
        lcc.workflow._validate_file_format(
            "file.warc", FileFormats.JSONL, process2o, "fmt", "file"
        )
        is FileFormats.JSONL
    )

    msg = r"Unsupported file file format 'warc'. Choose one of \(<FileFormats.SOURCE: 'source'>, <FileFormats.MEDUSA: 'medusa'>, <FileFormats.JSONL: 'jsonl'>\)."
    with pytest.raises(ValueError) as exc_info:
        lcc.workflow._validate_file_format("file.warc", None, process2o, "fmt", "file")
    exc_info.match(msg)

    # no file format extension

    with pytest.raises(ValueError) as exc_info:
        lcc.workflow._validate_file_format("file", None, process1, "fmt", "file")
    exc_info.match("File file format couldn't be detected!")
    with pytest.raises(ValueError) as exc_info:
        lcc.workflow._validate_file_format("file", None, process1o, "fmt", "file")
    exc_info.match("File file format couldn't be detected!")

    # wrong file format extension (by parameter)

    msg = r"Unsupported file file format 'warc'. Choose one of \(<FileFormats.SOURCE: 'source'>,\)."
    with pytest.raises(ValueError) as exc_info:
        lcc.workflow._validate_file_format(
            "file", FileFormats.WARC, process1, "fmt", "file"
        )
    exc_info.match(msg)
    with pytest.raises(ValueError) as exc_info:
        lcc.workflow._validate_file_format(
            "file.source", FileFormats.WARC, process1, "fmt", "file"
        )
    exc_info.match(msg)
    with pytest.raises(ValueError) as exc_info:
        lcc.workflow._validate_file_format(
            "file.warc", FileFormats.WARC, process1, "fmt", "file"
        )
    exc_info.match(msg)
    with pytest.raises(ValueError) as exc_info:
        lcc.workflow._validate_file_format(
            "file", FileFormats.WARC, process1o, "fmt", "file"
        )
    exc_info.match(msg)

    # wrong file format extension (by extension)

    msg = r"Unsupported file file format 'warc'. Choose one of \(<FileFormats.SOURCE: 'source'>,\)."
    with pytest.raises(ValueError) as exc_info:
        lcc.workflow._validate_file_format("file.warc", None, process1o, "fmt", "file")
    exc_info.match(msg)
    msg = r"Unsupported file file format 'jsonl'. Choose one of \(<FileFormats.SOURCE: 'source'>,\)."
    with pytest.raises(ValueError) as exc_info:
        lcc.workflow._validate_file_format(
            "file.jsonlines", None, process1o, "fmt", "file"
        )
    exc_info.match(msg)


# ---------------------------------------------------------------------------
# source


SOURCE_CONTENT_FAKE = b"""<source><location>NAME</location><date>DATE</date></source>
CONTENT
"""

# from: de.uni_leipzig.asv.tools.sentencecleaner/-/blob/master/testdata/inputfile_raw
SOURCE_CONTENT_1 = """<source><location>http://www.sueddeutsche.de/politik/562/464164/text/</location></source>
Welt-Finanzgipfel Regierungschefs suchen Weg aus der Krise
<source><location>http://www.sueddeutsche.de/politik/562/464164/text2/</location></source>
Welt-Finanzgipfel Regierungschefs suchen Weg aus der Krise.
Das Auto ist groß.
Das Auto ist groß...
(Oh ja!)
Das ist ein Satz mit	Tab.
Das ist ein Satz mit  zwei Leerzeichen!
Das is,t,,, ei,n Sa,tz mit ,vi,,,,elen Kommata.
""".encode()


def test_is_source_header():
    # no headers
    assert not lcc.io.SourceDocMetadata.is_header("")
    assert not lcc.io.SourceDocMetadata.is_header("This ...")
    assert not lcc.io.SourceDocMetadata.is_header("<source><")
    assert not lcc.io.SourceDocMetadata.is_header(b"<source><")
    assert not lcc.io.SourceDocMetadata.is_header("...></source>\n")
    assert not lcc.io.SourceDocMetadata.is_header("<source</source>")
    # needs to be xml
    assert not lcc.io.SourceDocMetadata.is_header("<source>...</source>\n")
    # can not be empty
    assert not lcc.io.SourceDocMetadata.is_header("<source></source>\n")
    # fake but would usually not be valid
    assert lcc.io.SourceDocMetadata.is_header("<source><x></x></source>\n")
    assert lcc.io.SourceDocMetadata.is_header("<source><x></x></source>")
    assert lcc.io.SourceDocMetadata.is_header(b"<source><x></x></source>")


def test_parse_source_header():
    line = """<source><location>http://www.example.de/</location><date>2021-07-12</date><encoding>UTF-8</encoding><languages ls-version="2.0.0"><language confidence="1.00">spa</language></languages></source>"""
    meta = lcc.io.SourceDocMetadata.from_str(line)
    assert meta.location == "http://www.example.de/"
    assert meta.date == "2021-07-12"
    assert meta.extra == [
        "<encoding>UTF-8</encoding>",
        """<languages ls-version="2.0.0"><language confidence="1.00">spa</language></languages>""",
    ]

    line_b = b"""<source><location>http://www.example.de/</location><date>2021-07-12</date><encoding>UTF-8</encoding><languages ls-version="2.0.0"><language confidence="1.00">spa</language></languages></source>\n"""
    meta_b = lcc.io.SourceDocMetadata.from_str(line_b)
    assert meta.location == meta_b.location
    assert meta.date == meta_b.date
    assert meta.extra == meta_b.extra

    line = """<source><location>http://www.example.de/</location><date>2021-07-12</date></source>"""
    meta = lcc.io.SourceDocMetadata.from_str(line)
    assert meta.location == "http://www.example.de/"
    assert meta.date == "2021-07-12"
    assert meta.extra == []

    line = """<source><location>http://www.example.de/</location></source>"""
    meta = lcc.io.SourceDocMetadata.from_str(line)
    assert meta.location == "http://www.example.de/"
    assert meta.date is None
    assert not meta.extra

    line = """<source><date>2021-07-12</date><x-tra>value</x-tra></source>"""
    meta = lcc.io.SourceDocMetadata.from_str(line)
    assert meta.location is None
    assert meta.date == "2021-07-12"
    assert meta.extra == ["<x-tra>value</x-tra>"]

    # encoding and languages are known fields
    line = """<source><date>2021-07-12</date><encoding>UTF-8</encoding><x-tra>value</x-tra><languages ls-version="2.0.0"><language confidence="1.00">deu?</language></languages><y-tra>value</y-tra></source>"""
    meta = lcc.io.SourceDocMetadata.from_str(line)
    assert meta.location is None
    assert meta.date == "2021-07-12"
    assert meta.extra == [
        "<encoding>UTF-8</encoding>",
        """<languages ls-version="2.0.0"><language confidence="1.00">deu?</language></languages>""",
        "<x-tra>value</x-tra><y-tra>value</y-tra>",
    ]
    line = """<source><date>2021-07-12</date><encoding>UTF-8</encoding><x-tra>value</x-tra><languages><language>xxx</language></languages><y-tra>value</y-tra></source>"""
    meta = lcc.io.SourceDocMetadata.from_str(line)
    assert meta.location is None
    assert meta.date == "2021-07-12"
    assert meta.extra == [
        "<encoding>UTF-8</encoding>",
        """<languages><language>xxx</language></languages>""",
        "<x-tra>value</x-tra><y-tra>value</y-tra>",
    ]

    # TODO: do we require content or is empty allowed?
    # NOTE: in source converter, missing will be set to empty!
    line = """<source><location></location><date></date></source>"""
    meta = lcc.io.SourceDocMetadata.from_str(line)
    assert meta.location == ""
    assert meta.date == ""
    assert not meta.extra

    # missing location and date is invalid (tags should exist)
    line = """<source><x-tra>value</x-tra></source>"""
    meta = lcc.io.SourceDocMetadata.from_str(line)
    assert meta is None

    line = """<source></source>\n"""
    meta = lcc.io.SourceDocMetadata.from_str(line)
    assert meta is None


def test_gen_source_header():
    meta = lcc.io.SourceDocMetadata("https://url.de", "2023-10-04")
    assert (
        meta.to_str()
        == """<source><location>https://url.de</location><date>2023-10-04</date></source>\n"""
    )

    meta.extra.extend(["MORE", "<<CONTENT>>"])
    assert (
        meta.to_str()
        == """<source><location>https://url.de</location><date>2023-10-04</date>MORE<<CONTENT>></source>\n"""
    )


def test_parse_source_docs_iter(tmp_path: Path):
    fn_source = tmp_path / "fake.source"
    fn_source.write_bytes(SOURCE_CONTENT_FAKE)
    source_docs = list(lcc.io.parse_source_docs_iter(str(fn_source), add_content=True))
    assert 1 == len(source_docs)

    source_doc = source_docs[0]
    assert source_doc.meta
    assert source_doc.content
    assert source_doc.meta.location == "NAME"
    assert source_doc.meta.date == "DATE"
    assert source_doc.content == b"CONTENT\n"

    fn_source = tmp_path / "test1.source"
    fn_source.write_bytes(SOURCE_CONTENT_1)
    assert fn_source.stat().st_size == 465
    source_docs = list(lcc.io.parse_source_docs_iter(str(fn_source), add_content=True))
    assert 2 == len(source_docs)

    source_doc = source_docs[0]
    assert source_doc.offset == 0
    assert source_doc.length == 149
    assert (
        source_doc.meta.location
        == "http://www.sueddeutsche.de/politik/562/464164/text/"
    )
    assert source_doc.meta.date is None
    assert (
        source_doc.content
        == b"Welt-Finanzgipfel Regierungschefs suchen Weg aus der Krise\n"
    )

    source_doc = source_docs[1]
    assert source_doc.offset == 149
    assert source_doc.length == 316  # 465 - 149
    assert (
        source_doc.meta.location
        == "http://www.sueddeutsche.de/politik/562/464164/text2/"
    )
    assert source_doc.meta.date is None
    assert isinstance(source_doc.content, (bytes, bytearray))
    assert source_doc.content.decode().startswith(
        "Welt-Finanzgipfel Regierungschefs suchen Weg aus der Krise.\nDas Auto ist groß.\n"
    )


def test_parse_source_docs_iter_stdin(set_stdin_content):
    # test for stdin input (non-seekable)
    set_stdin_content(SOURCE_CONTENT_FAKE)
    source_docs = list(lcc.io.parse_source_docs_iter("-", add_content=True))
    assert 1 == len(source_docs)

    source_doc = source_docs[0]
    assert source_doc.meta
    assert source_doc.content
    assert source_doc.meta.location == "NAME"
    assert source_doc.meta.date == "DATE"
    assert source_doc.content == b"CONTENT\n"

    set_stdin_content(SOURCE_CONTENT_1)
    source_docs = list(lcc.io.parse_source_docs_iter("-", add_content=True))
    assert 2 == len(source_docs)

    source_doc = source_docs[0]
    assert source_doc.offset == 0
    assert source_doc.length == 149
    assert (
        source_doc.meta.location
        == "http://www.sueddeutsche.de/politik/562/464164/text/"
    )
    assert source_doc.meta.date is None
    assert (
        source_doc.content
        == b"Welt-Finanzgipfel Regierungschefs suchen Weg aus der Krise\n"
    )

    source_doc = source_docs[1]
    assert source_doc.offset == 149
    assert source_doc.length == 316  # 465 - 149
    assert (
        source_doc.meta.location
        == "http://www.sueddeutsche.de/politik/562/464164/text2/"
    )
    assert source_doc.meta.date is None
    assert isinstance(source_doc.content, (bytes, bytearray))
    assert source_doc.content.decode().startswith(
        "Welt-Finanzgipfel Regierungschefs suchen Weg aus der Krise.\nDas Auto ist groß.\n"
    )


def test_write_source_docs_iter(tmp_path: Path, mocker: "MockerFixture"):
    fn_source = tmp_path / "out.source"

    meta = lcc.io.SourceDocMetadata("loc", "date", ["EXTRA"])
    doc = lcc.io.DocAndMeta(meta, b"a lot of\ncontent")
    meta2 = lcc.io.SourceDocMetadata("loc2", None)
    doc2 = lcc.io.DocAndMeta(meta2, "moar content\n\n\n")

    spy_writer = mocker.spy(lcc.io, "_write_source_doc")

    lcc.io.write_source_docs_iter(str(fn_source), [doc, doc2])
    assert spy_writer.call_count == 2
    spy_writer.reset_mock()

    content = fn_source.read_bytes()
    assert (
        content == b"<source><location>loc</location><date>date</date>EXTRA</source>\n"
        b"a lot of\ncontent\n"
        b"<source><location>loc2</location><date></date></source>\n"
        b"moar content\n\n\n"
    )

    # handle base docmeta class

    meta = lcc.io.DocMetadata("loc", "date")
    doc = lcc.io.DocAndMeta(meta, b"a lot of\ncontent")
    meta2 = lcc.io.DocMetadata("loc2", None)
    doc2 = lcc.io.DocAndMeta(meta2, "moar content\n\n\n")

    lcc.io.write_source_docs_iter(str(fn_source), [doc, doc2])
    assert spy_writer.call_count == 2

    content = fn_source.read_bytes()
    assert (
        content == b"<source><location>loc</location><date>date</date></source>\n"
        b"a lot of\ncontent\n"
        b"<source><location>loc2</location><date></date></source>\n"
        b"moar content\n\n\n"
    )


def test_write_source_docs_split_iter(tmp_path: Path, mocker: "MockerFixture"):
    fn_source = tmp_path / "out.source"

    meta = lcc.io.SourceDocMetadata("loc", "date", ["EXTRA"])
    doc = lcc.io.DocAndMeta(meta, b"a lot of\ncontent")
    meta2 = lcc.io.DocMetadata("loc2", None)
    doc2 = lcc.io.DocAndMeta(meta2, "moar content\n\n\n")
    meta3 = lcc.io.SourceDocMetadata("loc3", None)
    doc3 = lcc.io.DocAndMeta(meta3, "moar content \nagain\n\n")

    # check call without limits is redirected to base method
    mock_write = mocker.patch("lcc.io.write_source_docs_iter")
    fns = lcc.io.write_source_docs_split_iter(str(fn_source), [doc, doc2, doc3])
    assert fns == [str(fn_source)]
    mock_write.assert_called_once_with(
        str(fn_source), [doc, doc2, doc3], encoding=lcc.io.ENCODING
    )
    # mocked, should have written nothing
    assert not fn_source.exists()
    mock_write.reset_mock()

    # never to reach limits, should write but using own filenames
    fns = lcc.io.write_source_docs_split_iter(
        str(fn_source), [doc, doc2, doc3], maxbytes=2**20, maxdocs=1000
    )
    fn_source_1 = Path(next(lcc.io._generate_filenames(fn_source)))
    assert len(fns) == 1
    assert fns == [str(fn_source_1)]
    assert fn_source != fn_source_1
    assert not fn_source.exists()
    assert fn_source_1.exists()

    content = fn_source_1.read_bytes()
    assert (
        content == b"<source><location>loc</location><date>date</date>EXTRA</source>\n"
        b"a lot of\ncontent\n"
        b"<source><location>loc2</location><date></date></source>\n"
        b"moar content\n\n\n"
        b"<source><location>loc3</location><date></date></source>\n"
        b"moar content \nagain\n\n"
    )

    # set limits (n_docs=2)
    fn_source = tmp_path / "out-n2n.source"
    fn_iter = lcc.io._generate_filenames(fn_source)
    fn_source_1 = Path(next(fn_iter))
    fn_source_2 = Path(next(fn_iter))
    fns = lcc.io.write_source_docs_split_iter(
        str(fn_source), [doc, doc2, doc3], maxbytes=2**20, maxdocs=2
    )
    assert fns == [str(fn_source_1), str(fn_source_2)]
    assert not fn_source.exists()
    assert fn_source_1.exists()
    assert fn_source_2.exists()

    content1 = fn_source_1.read_bytes()
    assert (
        content1 == b"<source><location>loc</location><date>date</date>EXTRA</source>\n"
        b"a lot of\ncontent\n"
        b"<source><location>loc2</location><date></date></source>\n"
        b"moar content\n\n\n"
    )
    content2 = fn_source_2.read_bytes()
    assert (
        content2 == b"<source><location>loc3</location><date></date></source>\n"
        b"moar content \nagain\n\n"
    )

    # set limits (n_bytes=64+17+56+15)
    fn_source = tmp_path / "out-b.source"
    fn_iter = lcc.io._generate_filenames(fn_source)
    fn_source_1 = Path(next(fn_iter))
    fn_source_2 = Path(next(fn_iter))
    lcc.io.write_source_docs_split_iter(
        str(fn_source), [doc, doc2, doc3], maxbytes=64 + 17 + 56 + 15, maxdocs=10
    )
    assert not fn_source.exists()
    assert fn_source_1.exists()
    assert fn_source_2.exists()

    content1 = fn_source_1.read_bytes()
    assert (
        content1 == b"<source><location>loc</location><date>date</date>EXTRA</source>\n"
        b"a lot of\ncontent\n"
        b"<source><location>loc2</location><date></date></source>\n"
        b"moar content\n\n\n"
    )
    content2 = fn_source_2.read_bytes()
    assert (
        content2 == b"<source><location>loc3</location><date></date></source>\n"
        b"moar content \nagain\n\n"
    )

    # set limits (n_bytes=64+17+56+15 -1) - [1st doc] [2nd doc (too large for first slice) + 3rd doc]
    fn_source = tmp_path / "out-bd.source"
    fn_iter = lcc.io._generate_filenames(fn_source)
    fn_source_1 = Path(next(fn_iter))
    fn_source_2 = Path(next(fn_iter))
    lcc.io.write_source_docs_split_iter(
        str(fn_source), [doc, doc2, doc3], maxbytes=64 + 17 + 56 + 15 - 1, maxdocs=2
    )
    assert not fn_source.exists()
    assert fn_source_1.exists()
    assert fn_source_2.exists()

    content1 = fn_source_1.read_bytes()
    assert (
        content1 == b"<source><location>loc</location><date>date</date>EXTRA</source>\n"
        b"a lot of\ncontent\n"
    )
    content2 = fn_source_2.read_bytes()
    assert (
        content2 == b"<source><location>loc2</location><date></date></source>\n"
        b"moar content\n\n\n"
        b"<source><location>loc3</location><date></date></source>\n"
        b"moar content \nagain\n\n"
    )

    # set limits: minimum (writes at least 1 document in each file, otherwise endless loop)
    fn_source = tmp_path / "out-1m.source"
    fn_iter = lcc.io._generate_filenames(fn_source)
    fn_source_1 = Path(next(fn_iter))
    fn_source_2 = Path(next(fn_iter))
    fn_source_3 = Path(next(fn_iter))
    fn_source_4 = Path(next(fn_iter))
    lcc.io.write_source_docs_split_iter(
        str(fn_source), [doc, doc2, doc3], maxbytes=1, maxdocs=0
    )
    assert not fn_source.exists()
    assert fn_source_1.exists()
    assert fn_source_2.exists()
    assert fn_source_3.exists()
    assert not fn_source_4.exists()

    content1 = fn_source_1.read_bytes()
    assert (
        content1 == b"<source><location>loc</location><date>date</date>EXTRA</source>\n"
        b"a lot of\ncontent\n"
    )
    content2 = fn_source_2.read_bytes()
    assert (
        content2 == b"<source><location>loc2</location><date></date></source>\n"
        b"moar content\n\n\n"
    )
    content3 = fn_source_3.read_bytes()
    assert (
        content3 == b"<source><location>loc3</location><date></date></source>\n"
        b"moar content \nagain\n\n"
    )
    # NOTE: we could check the logs output, it should have warnung us

    # set limits: minimum (writes at least 1 document in each file, no size limit)
    fn_source = tmp_path / "out-1m2.source"
    fn_iter = lcc.io._generate_filenames(fn_source)
    fn_source_1 = Path(next(fn_iter))
    fn_source_2 = Path(next(fn_iter))
    fn_source_3 = Path(next(fn_iter))
    fn_source_4 = Path(next(fn_iter))
    fns = lcc.io.write_source_docs_split_iter(
        str(fn_source), [doc, doc2, doc3], maxbytes=None, maxdocs=1
    )
    assert fns == [str(fn_source_1), str(fn_source_2), str(fn_source_3)]
    assert not fn_source.exists()
    assert fn_source_1.exists()
    assert fn_source_2.exists()
    assert fn_source_3.exists()
    assert not fn_source_4.exists()

    content1 = fn_source_1.read_bytes()
    assert (
        content1 == b"<source><location>loc</location><date>date</date>EXTRA</source>\n"
        b"a lot of\ncontent\n"
    )
    content2 = fn_source_2.read_bytes()
    assert (
        content2 == b"<source><location>loc2</location><date></date></source>\n"
        b"moar content\n\n\n"
    )
    content3 = fn_source_3.read_bytes()
    assert (
        content3 == b"<source><location>loc3</location><date></date></source>\n"
        b"moar content \nagain\n\n"
    )

    mock_write.assert_not_called()


# ---------------------------------------------------------------------------
# medusa


def test__convert_from_source_to_medusa():
    meta = lcc.io.DocMetadata("loc", "date")
    doc_in = lcc.io.DocAndMeta(meta=meta, content="abc\nde\tf\n")

    sents_out = list(lcc.io._convert_from_source_to_medusa([doc_in]))
    assert len(sents_out) == 1
    assert sents_out[0].meta == meta
    assert sents_out[0].sentences == ["abc", "de f"]


def test_write_sentences_to_medusa_format_iter(tmp_path: Path):
    fn_out = tmp_path / "out.medusa"

    meta = lcc.io.DocMetadata("loc", "date")
    sents_out = lcc.io.SentencesAndMeta(meta, ["abc", "cde"])

    lcc.io.write_sentences_to_medusa_format_iter(str(fn_out), [sents_out])

    content = fn_out.read_bytes()
    assert content == b"abc\tdate\tloc\ncde\tdate\tloc\n"


# ---------------------------------------------------------------------------
# jsonl


def test_parse_document_jsonl(tmp_path: Path):
    fn_in = tmp_path / "docs.jsonl"
    fn_in.write_bytes(b'{"location": "loc", "date": "date", "content": "abcdef\\n"}\n')

    doc_in = list(lcc.io.parse_document_jsonl(str(fn_in)))
    assert 1 == len(doc_in)

    doc = doc_in[0]
    assert doc.meta
    assert doc.content
    assert doc.meta.location == "loc"
    assert doc.meta.date == "date"
    # content will always be a string when loaded from json
    # does not make sense to encode back to bytes
    assert doc.content == "abcdef\n"


def test_parse_document_jsonl_stdin(set_stdin_content):
    set_stdin_content(b'{"location": "loc", "date": "date", "content": "abcdef\\n"}\n')

    doc_in = list(lcc.io.parse_document_jsonl("-"))
    assert 1 == len(doc_in)

    doc = doc_in[0]
    assert doc.meta
    assert doc.content
    assert doc.meta.location == "loc"
    assert doc.meta.date == "date"
    # content will always be a string when loaded from json
    # does not make sense to encode back to bytes
    assert doc.content == "abcdef\n"


def test_parse_sentence_jsonl(tmp_path: Path, caplog: "LogCaptureFixture"):
    fn_in = tmp_path / "docs.jsonl"
    fn_in.write_bytes(b'{"location": "loc2", "date": "date", "sentence": "abcdef"}\n')

    sentences_in = list(
        lcc.io.parse_sentence_jsonl(str(fn_in), is_single_sentence=True)
    )
    assert len(sentences_in) == 1
    sentence = sentences_in[0]
    assert isinstance(sentence, lcc.io.SentenceAndMeta)
    assert sentence.meta
    assert sentence.meta.location == "loc2"
    assert sentence.meta.date == "date"
    assert sentence.sentence == "abcdef"

    sentences_in = list(
        lcc.io.parse_sentence_jsonl(str(fn_in), is_single_sentence=False)
    )
    assert len(sentences_in) == 1
    sentence = sentences_in[0]
    assert isinstance(sentence, lcc.io.SentencesAndMeta)
    assert sentence.meta
    assert sentence.meta.location == "loc2"
    assert sentence.meta.date == "date"
    assert sentence.sentences == ["abcdef"]

    # multi sentence doc
    fn_in.write_bytes(
        b'{"location": "loc2", "date": "date", "sentences": ["abcdef", "123"]}\n'
    )

    sentences_in = list(
        lcc.io.parse_sentence_jsonl(str(fn_in), is_single_sentence=True)
    )
    assert len(sentences_in) == 2
    sentence = sentences_in[1]
    assert isinstance(sentence, lcc.io.SentenceAndMeta)
    assert sentence.meta
    assert sentence.meta.location == "loc2"
    assert sentence.meta.date == "date"
    assert sentence.sentence == "123"

    sentences_in = list(
        lcc.io.parse_sentence_jsonl(str(fn_in), is_single_sentence=False)
    )
    assert len(sentences_in) == 1
    sentence = sentences_in[0]
    assert isinstance(sentence, lcc.io.SentencesAndMeta)
    assert sentence.meta
    assert sentence.meta.location == "loc2"
    assert sentence.meta.date == "date"
    assert sentence.sentences == ["abcdef", "123"]

    # test invalid doc
    fn_in.write_bytes(b'{"location": "loc", "date": "date", "content": "abcdef\\n"}\n')
    caplog.clear()
    sentences_in = list(lcc.io.parse_sentence_jsonl(str(fn_in)))
    assert len(sentences_in) == 0
    assert len(caplog.records) == 1


def test_write_document_jsonl(tmp_path: Path):
    fn_out = tmp_path / "docs.jsonl"

    meta = lcc.io.DocMetadata("loc", "date")

    # content string
    doc_out = lcc.io.DocAndMeta(meta=meta, content="abcdef\n", offset=42)
    lcc.io.write_document_jsonl(str(fn_out), [doc_out])

    content = fn_out.read_bytes()
    assert content == b'{"location": "loc", "date": "date", "content": "abcdef\\n"}\n'

    # content bytes
    doc_out = lcc.io.DocAndMeta(meta=meta, content=b"xyz123", offset=42)
    lcc.io.write_document_jsonl(str(fn_out), [doc_out])

    content = fn_out.read_bytes()
    assert content == b'{"location": "loc", "date": "date", "content": "xyz123"}\n'

    # extra meta
    meta_source = lcc.io.SourceDocMetadata(
        meta.location, meta.date, extra=["extra", "values"]
    )
    doc_out = lcc.io.DocAndMeta(meta=meta_source, content="xxx\naaa\n")

    lcc.io.write_document_jsonl(str(fn_out), [doc_out])
    content = fn_out.read_bytes()
    assert (
        content == b'{"location": "loc", "date": "date", "content": "xxx\\naaa\\n"}\n'
    )

    # writing needs to be enabled
    lcc.io.write_document_jsonl(str(fn_out), [doc_out], include_extra_meta=True)
    content = fn_out.read_bytes()
    assert (
        content
        == b'{"location": "loc", "date": "date", "content": "xxx\\naaa\\n", "extra_meta": ["extra", "values"]}\n'
    )

    # if no extra meta, skip
    meta_source.extra.clear()
    lcc.io.write_document_jsonl(str(fn_out), [doc_out], include_extra_meta=True)
    content = fn_out.read_bytes()
    assert (
        content == b'{"location": "loc", "date": "date", "content": "xxx\\naaa\\n"}\n'
    )

    # if no extra meta, skip
    doc_out = lcc.io.DocAndMeta(meta=meta, content="xxx\naaa\n")
    lcc.io.write_document_jsonl(str(fn_out), [doc_out], include_extra_meta=True)
    content = fn_out.read_bytes()
    assert (
        content == b'{"location": "loc", "date": "date", "content": "xxx\\naaa\\n"}\n'
    )


def test_write_sentences_jsonl(tmp_path: Path):
    fn_out = tmp_path / "sentences.jsonl"

    meta = lcc.io.DocMetadata("loc", "date")

    # multiple sentences
    sents_out = lcc.io.SentencesAndMeta(meta, ["abc", "cde"])
    lcc.io.write_sentences_jsonl(str(fn_out), [sents_out])

    content = fn_out.read_bytes()
    assert (
        content == b'{"location": "loc", "date": "date", "sentences": ["abc", "cde"]}\n'
    )

    # single sentences
    sent_out = lcc.io.SentenceAndMeta(meta, "xyz")
    lcc.io.write_sentences_jsonl(str(fn_out), [sent_out])

    content = fn_out.read_bytes()
    assert content == b'{"location": "loc", "date": "date", "sentences": ["xyz"]}\n'


def test_write_sentence_jsonl(tmp_path: Path):
    fn_out = tmp_path / "sentence.jsonl"

    meta = lcc.io.DocMetadata("loc", "date")

    # multiple sentences
    sents_out = lcc.io.SentencesAndMeta(meta, ["abc", "cde"])
    lcc.io.write_sentence_jsonl(str(fn_out), [sents_out])

    content = fn_out.read_bytes()
    assert (
        content == b'{"location": "loc", "date": "date", "sentence": "abc"}\n'
        b'{"location": "loc", "date": "date", "sentence": "cde"}\n'
    )

    # single sentences
    sent_out = lcc.io.SentenceAndMeta(meta, "xyz")
    lcc.io.write_sentence_jsonl(str(fn_out), [sent_out])

    content = fn_out.read_bytes()
    assert content == b'{"location": "loc", "date": "date", "sentence": "xyz"}\n'


# ---------------------------------------------------------------------------
# warc

# ---------------------------------------------------------------------------
