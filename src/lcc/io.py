import contextlib
import inspect
import io
import itertools
import json
import logging
import os.path
import re
import sys
import time
import typing
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union
from typing import cast

# ---------------------------------------------------------------------------
# polyfill (3.9 compatibility)


if not hasattr(inspect, "get_annotations"):
    # backport `inspect.get_annotations` from 3.10.12 for <=3.9
    import functools
    import types

    def get_annotations(obj, *, globals=None, locals=None, eval_str=False):
        if isinstance(obj, type):
            # class
            obj_dict = getattr(obj, "__dict__", None)
            if obj_dict and hasattr(obj_dict, "get"):
                ann = obj_dict.get("__annotations__", None)
                if isinstance(ann, types.GetSetDescriptorType):
                    ann = None
            else:
                ann = None

            obj_globals = None
            module_name = getattr(obj, "__module__", None)
            if module_name:
                module = sys.modules.get(module_name, None)
                if module:
                    obj_globals = getattr(module, "__dict__", None)
            obj_locals = dict(vars(obj))
            unwrap = obj
        elif isinstance(obj, types.ModuleType):
            # module
            ann = getattr(obj, "__annotations__", None)
            obj_globals = getattr(obj, "__dict__")  # noqa: B009
            obj_locals = None
            unwrap = None
        elif callable(obj):
            # this includes types.Function, types.BuiltinFunctionType,
            # types.BuiltinMethodType, functools.partial, functools.singledispatch,
            # "class funclike" from Lib/test/test_inspect... on and on it goes.
            ann = getattr(obj, "__annotations__", None)
            obj_globals = getattr(obj, "__globals__", None)
            obj_locals = None
            unwrap = obj
        else:
            raise TypeError(f"{obj!r} is not a module, class, or callable.")

        if ann is None:
            return {}

        if not isinstance(ann, dict):
            raise ValueError(f"{obj!r}.__annotations__ is neither a dict nor None")

        if not ann:
            return {}

        if not eval_str:
            return dict(ann)

        if unwrap is not None:
            while True:
                if hasattr(unwrap, "__wrapped__"):
                    unwrap = unwrap.__wrapped__
                    continue
                if isinstance(unwrap, functools.partial):
                    unwrap = unwrap.func
                    continue
                break
            if hasattr(unwrap, "__globals__"):
                obj_globals = unwrap.__globals__

        if globals is None:
            globals = obj_globals
        if locals is None:
            locals = obj_locals

        return_value = {
            key: value
            if not isinstance(value, str)
            else eval(value, globals, locals)  # noqa: S307
            for key, value in ann.items()
        }
        return return_value

    setattr(inspect, "get_annotations", get_annotations)  # noqa: B010


# ---------------------------------------------------------------------------


LOGGER = logging.getLogger(__name__)


ENCODING = "utf-8"

# TODO: do we require content or is empty allowed?
PATTERN_TAG_LOCATION = re.compile("<location>(.*?)</location>")
PATTERN_TAG_DATE = re.compile("<date>(.*?)</date>")
PATTERN_TAG_ENCODING = re.compile("<encoding>(.*?)</encoding>")
PATTERN_TAG_LANGUAGES = re.compile(
    r'<languages( ls-version="(\d+(\.\d+)*)")?>(.*?)</languages>'
)
# <quelle> --> <source>
#   <name_lang>  --> <location>
#   <datum>      --> <date>
#   <sachgebiet> --> <subject_area>
#   <name>       --> <description>
LEN_SOURCE_TAGS: int = 2 * 6 + 5
LEN_LOCATION_TAGS: int = 2 * 8 + 5
LEN_DATE_TAGS: int = 2 * 4 + 5

MEDUSA_COL_SEP: str = "\t"

DEFAULT_DATE_IF_MISSING: str = time.strftime("%Y-%m-%d")
DEFAULT_LOCATION_IF_MISSING: str = "unknown source"


# ---------------------------------------------------------------------------
# helpers


def _find_files(
    folder: Union[str, PathLike],
    file_pattern: str = "*",
    recursive: bool = False,
    files_only: bool = True,
) -> List[Path]:
    if "../" in file_pattern or "/.." in file_pattern:
        LOGGER.warning("'..' pattern not allowed! pattern=%s", file_pattern)
        return []
    if not recursive and "**" in file_pattern:
        LOGGER.warning(
            "'**' pattern not allowed, use 'recursive' parameter! pattern=%s",
            file_pattern,
        )
        return []
    folder = Path(folder)
    globber = folder.rglob if recursive else folder.glob
    fileiter = globber(file_pattern)
    if files_only:
        fileiter = (file for file in fileiter if file.is_file())
    return list(fileiter)


def _generate_filenames(
    filename: Union[str, PathLike],
    start: int = 1,
    stop: Optional[int] = None,
    sep: str = "_",
    padlen: int = 3,
    where: Literal["prefix", "suffix", "infix"] = "infix",
) -> Iterator[str]:
    # validate params
    if not filename:
        raise ValueError("filename must not be empty/None!")
    if not where or where.lower() not in ["prefix", "suffix", "infix"]:
        raise ValueError("where has invalid value!")

    # sane basics if out of range
    where = where.lower()  # type: ignore[assignment]
    sep = sep or ""
    padlen = max(0, padlen or 0)
    start = max(0, start or 0)
    if stop is not None and stop < start:
        stop = start

    # template
    folder, file = os.path.split(str(filename))
    file_base, file_ext = os.path.splitext(file)
    if where == "prefix":
        name_pattern = f"{{:0{padlen}d}}{sep}{file_base}{file_ext}"
    elif where == "suffix":
        name_pattern = f"{file_base}{file_ext}{sep}{{:0{padlen}d}}"
    else:
        name_pattern = f"{file_base}{sep}{{:0{padlen}d}}{file_ext}"

    # generate patterns
    num_gen: Iterable[int]
    if stop is None:
        num_gen = itertools.count(start)
    else:
        num_gen = range(start, stop + 1)
    for num in num_gen:
        filename = name_pattern.format(num)
        if folder:
            filename = os.path.join(folder, filename)
        yield filename


OpenTextMode = Literal["w", "a", "r"]
OpenBinaryModeWriting = Literal["wb", "bw"]
OpenBinaryModeReading = Literal["rb", "br"]


@typing.overload
def _open_stream(
    fn: Union[str, PathLike], mode: OpenBinaryModeReading, encoding: None = None
) -> io.BufferedReader:
    ...


@typing.overload
def _open_stream(
    fn: Union[str, PathLike], mode: OpenBinaryModeWriting, encoding: None = None
) -> io.BufferedWriter:
    ...


@typing.overload
def _open_stream(
    fn: Union[str, PathLike],
    mode: Literal["w", "r", "a"],
    encoding: Optional[str] = ENCODING,
) -> io.TextIOWrapper:
    ...


def _open_stream(
    fn: Union[str, PathLike],
    mode: Union[OpenTextMode, OpenBinaryModeReading, OpenBinaryModeWriting] = "rb",
    encoding: Optional[str] = ENCODING,
) -> io.IOBase:
    # https://stackoverflow.com/a/61399222/9360161
    # https://stackoverflow.com/a/53088625/9360161
    # https://docs.python.org/3.9/library/contextlib.html#contextlib.nullcontext

    if not any(mode_char in mode for mode_char in ("r", "w")):
        raise ValueError("'mode' requires one of 'r' or 'w'")

    if fn == "-":
        fp: Union[typing.TextIO, typing.BinaryIO]
        if "r" in mode:
            fp = sys.stdin
        elif "w" in mode:
            fp = sys.stdout
        else:
            raise ValueError("Unknown 'mode' for stdin/stdout!")

        if "b" in mode:
            fp = cast(typing.BinaryIO, fp.buffer)

        return cast(io.IOBase, contextlib.nullcontext(fp))

    if "b" in mode:
        encoding = None

    return cast(io.IOBase, open(fn, mode=mode, encoding=encoding))


# ---------------------------------------------------------------------------
# file formats


class FileFormats(str, Enum):
    WARC = "warc"
    JSONL = "jsonl"
    SOURCE = "source"
    MEDUSA = "medusa"

    def __str__(self) -> str:
        return self._value_

    @staticmethod
    def detect_format_by_ext(filename: str) -> Optional["FileFormats"]:
        if not filename:
            return None

        ext = os.path.splitext(filename)[1]
        if not ext:
            return None
        ext = ext[1:]

        if ext.lower() in ("warc", "arc", "wet"):
            return FileFormats.WARC
        if ext.lower() in ("jsonl", "jsonlines"):
            return FileFormats.JSONL
        if ext.lower() == FileFormats.SOURCE.value:
            return FileFormats.SOURCE
        if ext.lower() == FileFormats.MEDUSA.value:
            return FileFormats.MEDUSA

        return None


def _extract_file_formats(function: Callable, param_name: str) -> Tuple[FileFormats]:
    try:
        func_types = inspect.get_annotations(function)
    except TypeError:
        return cast(Tuple[FileFormats], tuple())

    try:
        param_type = func_types[param_name]  # type for single parameter
    except KeyError:
        return cast(Tuple[FileFormats], tuple())

    # check if types were optional, then strip
    if typing.get_origin(param_type) is typing.Union and type(None) in typing.get_args(
        param_type
    ):
        param_type = typing.get_args(param_type)[0]  # strip Optional

    allowed_formats = typing.get_args(param_type)  # Literal values
    if type(allowed_formats) is not tuple or not all(
        type(ttype) is FileFormats for ttype in allowed_formats
    ):
        return cast(Tuple[FileFormats], tuple())

    # optional dependencies
    if not WARC_SUPPORTED and FileFormats.WARC in allowed_formats:
        LOGGER.debug(
            "Dropping %r from allowed file formats for '%s' of %s since optional dependency 'warcio' is not installed!",
            FileFormats.WARC,
            param_name,
            function,
        )
        allowed_formats = tuple(
            fmt for fmt in allowed_formats if fmt != FileFormats.WARC
        )

    return cast(Tuple[FileFormats], allowed_formats)


def _validate_file_format(
    fn: str, fmt: Optional[FileFormats], func: Callable, param: str, type_: str
) -> FileFormats:
    fmt_detected: Optional[FileFormats] = fmt

    if fmt is None:
        fmt_detected = FileFormats.detect_format_by_ext(fn)
        LOGGER.debug("Detected %s file format: %s", type_, fmt_detected)
    if fmt_detected is None:
        raise ValueError(f"{type_.title()} file format couldn't be detected!")

    func_types = inspect.get_annotations(func)
    param_type = func_types[param]  # type for single parameter
    # check if types were optional, then strip
    if typing.get_origin(param_type) is typing.Union and type(None) in typing.get_args(
        param_type
    ):
        param_type = typing.get_args(param_type)[0]  # strip Optional

    # if not optional, then also check if argument value was not optional (None)
    elif fmt is None:
        raise ValueError(f"Argument '{param}' is required to be not None!")

    allowed_formats = typing.get_args(param_type)  # Literal values

    # optional dependencies
    if not WARC_SUPPORTED and FileFormats.WARC in allowed_formats:
        LOGGER.debug(
            "Dropping %r from allowed file formats for '%s' of %s since optional dependency 'warcio' is not installed!",
            FileFormats.WARC,
            param,
            func,
        )
        allowed_formats = tuple(
            fmt for fmt in allowed_formats if fmt != FileFormats.WARC
        )

    if fmt_detected not in allowed_formats:
        raise ValueError(
            f"Unsupported {type_} file format '{fmt_detected}'. Choose one of {allowed_formats}."
        )

    return fmt_detected


# ---------------------------------------------------------------------------
# metadata


@dataclass
class DocMetadata:
    """Document metadata. Minimal set consisting of location (url) and date."""

    #: document source (URL)
    location: Optional[str]
    #: document date (YYYY-mm-dd)
    date: Optional[str]


@dataclass
class DocAndMeta:
    """Document and metadata. May contain content or just binary (offset, length) info."""

    #: document metadata
    meta: DocMetadata
    #: content
    content: Optional[Union[str, bytes, bytearray]] = None
    #: optional binary offset in file
    offset: Optional[int] = None
    #: optional binary length of both the header and the content
    length: Optional[int] = None


@dataclass
class SentencesAndMeta:
    """List of sentences and document metadata (location and date)."""

    #: document metadata
    meta: DocMetadata
    #: sentences
    sentences: List[str] = field(default_factory=list)


@dataclass
class SentenceAndMeta:
    """Single sentence and document metadata (location and date)."""

    #: document metadata
    meta: DocMetadata
    #: sentences
    sentence: str


# ---------------------------------------------------------------------------
# source format


@dataclass
class SourceDocMetadata(DocMetadata):
    """Document metadata for source format.

    May contain extra metadata fields. Has methods to parse and write
    metadata as source format header line."""

    #: optional additional metadata, like document language from langsepa
    extra: List[str] = field(default_factory=list)

    def to_str(self) -> str:
        return "".join(
            [
                "<source>",
                "<location>",
                self.location or "",
                "</location>",
                "<date>",
                self.date or "",
                "</date>",
                *self.extra,
                "</source>",
                "\n",
            ]
        )

    @classmethod
    def from_str(
        cls: Type["SourceDocMetadata"],
        line: Union[str, bytes],
        encoding: str = ENCODING,
    ) -> Optional["SourceDocMetadata"]:
        if not isinstance(line, str):
            line = line.decode(encoding)

        sourcematch = PATTERN_TAG_LOCATION.search(line)
        datematch = PATTERN_TAG_DATE.search(line)
        if not sourcematch and not datematch:
            return None

        location = sourcematch.groups()[0] if sourcematch else None
        date = datematch.groups()[0] if datematch else None

        meta = cls(location, date)

        # check if there could be extra data
        expected_len = (
            LEN_SOURCE_TAGS
            + ((LEN_LOCATION_TAGS + len(location)) if location is not None else 0)
            + ((LEN_DATE_TAGS + len(date)) if date is not None else 0)
        )
        if expected_len + 5 <= len(line):
            # strip container
            line = line.rstrip("\n")[8:-9]
            # strip mandatory fields
            line = PATTERN_TAG_LOCATION.sub("", line, 1)
            line = PATTERN_TAG_DATE.sub("", line, 1)

            # check for optional fields
            encodingmatch = PATTERN_TAG_ENCODING.search(line)
            if encodingmatch:
                meta.extra.append(encodingmatch.group())
                line = PATTERN_TAG_ENCODING.sub("", line, 1)
            languagesmatch = PATTERN_TAG_LANGUAGES.search(line)
            if languagesmatch:
                meta.extra.append(languagesmatch.group())
                line = PATTERN_TAG_LANGUAGES.sub("", line, 1)

            # rest
            if line:
                meta.extra.append(line)

        return meta

    @classmethod
    def is_header(cls, line: Union[str, bytes]) -> bool:
        if isinstance(line, str):
            return (
                line.startswith("<source><")
                and line.rstrip("\n").endswith("></source>")
                # can not be empty (len of source tags: 2 * 6 + 5; and some extra)
                and len(line) >= (LEN_SOURCE_TAGS + 5)
            )
        else:
            return (
                line.startswith(b"<source><")
                and line.rstrip(b"\n").endswith(b"></source>")
                and len(line) >= (LEN_SOURCE_TAGS + 5)
            )


def parse_source_docs_iter(
    fn: str, add_content: bool = True, encoding: str = ENCODING
) -> Iterator[DocAndMeta]:
    with _open_stream(fn, "rb") as fp:
        cur_byte_pos = offset_start = offset_end = 0
        meta: Optional[SourceDocMetadata] = None
        content_lines: List[bytes] = []

        for line in fp:
            cur_byte_pos += len(line)
            if SourceDocMetadata.is_header(line):
                if fp.seekable():
                    offset_end = fp.tell() - len(line)
                else:
                    offset_end = cur_byte_pos - len(line)

                # for first document
                if meta is None:
                    meta = SourceDocMetadata.from_str(line, encoding=encoding)
                    content_lines = []
                    continue

                # for other documents
                slice = DocAndMeta(
                    meta=meta,
                    content=b"".join(content_lines) if add_content else None,
                    offset=offset_start,
                    length=offset_end - offset_start,
                )
                yield slice

                # parse next header
                meta = SourceDocMetadata.from_str(line, encoding=encoding)
                content_lines = []
                offset_start = offset_end

            elif add_content:
                content_lines.append(line)

        # at file end
        if meta is not None:
            offset_end = fp.tell() if fp.seekable() else cur_byte_pos
            slice = DocAndMeta(
                meta=meta,
                content=b"".join(content_lines) if add_content else None,
                offset=offset_start,
                length=offset_end - offset_start,
            )
            yield slice


def _write_source_doc(fp: io.BufferedIOBase, doc: DocAndMeta, encoding: str = ENCODING):
    # write header
    if isinstance(doc.meta, SourceDocMetadata):
        meta = doc.meta
    else:
        meta = SourceDocMetadata(doc.meta.location, doc.meta.date)
    fp.write(meta.to_str().encode(encoding))

    # write content
    if doc.content is None:
        return

    if isinstance(doc.content, (bytes, bytearray)):
        content = doc.content
    else:
        content = doc.content.encode(encoding)
    fp.write(content)

    # write line break for next source header if missing
    if not content.endswith(b"\n"):
        fp.write(b"\n")


def write_source_docs_iter(
    fn: str, docs: Iterable[DocAndMeta], encoding: str = ENCODING
):
    with _open_stream(fn, "wb") as fp:
        for doc in docs:
            _write_source_doc(fp, doc, encoding=encoding)  # type: ignore[arg-type]


def write_source_docs_split_iter(
    fn: str,
    docs: Iterable[DocAndMeta],
    maxdocs: Optional[int] = None,
    maxbytes: Optional[int] = None,
    encoding: str = ENCODING,
) -> List[str]:
    if maxdocs is not None and maxdocs <= 0:
        maxdocs = None
    if maxbytes is not None and maxbytes <= 0:
        maxbytes = None

    if maxdocs is None and maxbytes is None:
        write_source_docs_iter(fn, docs, encoding=encoding)
        return [fn]

    docs = iter(docs)
    cur_doc = next(docs, None)
    if cur_doc is None:
        return []

    filenames: List[str] = []
    for filename in _generate_filenames(fn):
        # no more docs, finish
        if cur_doc is None:
            break

        # reset counters
        if maxdocs is not None:
            limit_docs = maxdocs
        if maxbytes is not None:
            limit_bytes = maxbytes

        with _open_stream(filename, "wb") as fp:
            LOGGER.debug("Start writing to %s", filename)
            filenames.append(filename)

            while cur_doc is not None:
                if maxdocs is not None and limit_docs <= 0:
                    break

                if maxbytes is not None:
                    if limit_bytes <= 0:
                        break

                    # compute current document length
                    buf = io.BytesIO()
                    _write_source_doc(buf, cur_doc, encoding=encoding)
                    content = buf.getvalue()
                    content_len = len(content)

                    # check if bytes remaining (only if current document is not larger than limit)
                    if content_len <= maxbytes:
                        if fp.seekable() and fp.tell() + content_len > maxbytes:
                            break
                        elif limit_bytes - content_len < 0:
                            break
                    else:
                        LOGGER.warning(
                            "Single document larger than maxbytes: %s > %s",
                            content_len,
                            maxbytes,
                        )
                        # TODO: raise an error if configured by user?

                    limit_bytes -= content_len

                if maxdocs:
                    limit_docs -= 1

                # write document since we had enough bytes left or doc counter still free
                if maxbytes is not None:
                    fp.write(content)
                else:
                    _write_source_doc(fp, cur_doc, encoding=encoding)  # type: ignore[arg-type]

                # fetch next doc
                cur_doc = next(docs, None)

    return filenames


# ---------------------------------------------------------------------------
# medusa format (tsv)


def _convert_from_source_to_medusa(
    docs: Iterable[DocAndMeta], encoding: str = ENCODING
) -> Iterator[SentencesAndMeta]:
    for doc in docs:
        # skip empty
        if not doc.content:
            continue
        # convert to string (if bytes)
        if isinstance(doc.content, (bytes, bytearray)):
            content: str = doc.content.decode(encoding=encoding)
        else:
            content = doc.content
        # make sentences
        sentences = content.splitlines(keepends=False)
        # escape tabs since not allowed in medusa, expensive but better to be safe
        sentences = [
            sent.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            for sent in sentences
        ]
        # converted.
        yield SentencesAndMeta(meta=doc.meta, sentences=sentences)


def parse_sentences_from_medusa_format_iter(
    fn: str, encoding: str = ENCODING
) -> Iterator[SentenceAndMeta]:
    with _open_stream(fn, "r", encoding=encoding) as fp:
        for lno, line in enumerate(fp):
            if not line:
                LOGGER.warning("Empty line in %s", lno)
                continue
            try:
                sentence, date, location = line.rstrip().split(MEDUSA_COL_SEP)
                yield SentenceAndMeta(
                    meta=DocMetadata(location=location, date=date), sentence=sentence
                )
            except ValueError:
                LOGGER.warning(
                    "Invalid line (that should contain tree fields: sentence, date, and url, separated by %r) '%s' at %s",
                    MEDUSA_COL_SEP,
                    line,
                    lno,
                )
                continue


def write_sentences_to_medusa_format_iter(
    fn: str,
    sentence_docs: Union[Iterable[SentencesAndMeta], Iterable[SentenceAndMeta]],
    encoding: str = ENCODING,
    default_date_if_missing: str = DEFAULT_DATE_IF_MISSING,
    default_location_if_missing: str = DEFAULT_LOCATION_IF_MISSING,
):
    with _open_stream(fn, "w", encoding=encoding) as fp:
        for sentence_doc in sentence_docs:
            location = sentence_doc.meta.location or default_location_if_missing
            # TODO: or better to use 0001-01-01 date to mark missing?
            date = sentence_doc.meta.date or default_date_if_missing
            if isinstance(sentence_doc, SentencesAndMeta):
                for sentence in sentence_doc.sentences:
                    fp.write(
                        f"{sentence}{MEDUSA_COL_SEP}{date}{MEDUSA_COL_SEP}{location}\n"
                    )
            elif isinstance(sentence_doc, SentenceAndMeta):
                fp.write(
                    f"{sentence_doc.sentence}{MEDUSA_COL_SEP}{date}{MEDUSA_COL_SEP}{location}\n"
                )


# NOTE: extra split/merge method is not required as the medusa file format is line oriented, just use basic shell tools like 'cat' or 'split'
# maybe if byte size oriented but well that's almost trivial to do in a small script


# ---------------------------------------------------------------------------
# jsonl format
# dialects:
# - document + meta
# - sentence(s) + meta


def parse_document_jsonl(fn: str) -> Iterator[DocAndMeta]:
    with _open_stream(fn, "rb") as fp:
        cur_byte_pos = 0
        for lno, line in enumerate(fp):
            cur_byte_pos += len(line)
            try:
                entry = json.loads(line)
                meta = DocMetadata(location=entry["location"], date=entry["date"])
                yield DocAndMeta(
                    meta=meta,
                    content=entry["content"],
                    offset=(fp.tell() if fp.seekable() else cur_byte_pos) - len(line),
                    length=len(line),
                )
            except json.JSONDecodeError:
                LOGGER.warning(
                    "Error reading and parsing line %s: %s",
                    lno,
                    line[:50] if line else "no content?",
                )
            except KeyError:
                LOGGER.warning(
                    "Error extracting data from line %s: %s",
                    lno,
                    line[:50] if line else "no content?",
                )


def parse_sentence_jsonl(
    fn: str, is_single_sentence: bool = False, encoding: str = ENCODING
) -> Union[Iterator[SentencesAndMeta], Iterator[SentenceAndMeta]]:
    with _open_stream(fn, "r", encoding=encoding) as fp:
        for lno, line in enumerate(fp):
            try:
                entry = json.loads(line)
                meta = DocMetadata(location=entry["location"], date=entry["date"])
                # TODO: also handle "content" field? (split into sentences?)
                if is_single_sentence:
                    if "sentence" in entry:
                        yield SentenceAndMeta(meta=meta, sentence=entry["sentence"])
                    elif "sentences" in entry:
                        for sentence in entry["sentences"]:
                            yield SentenceAndMeta(meta=meta, sentence=sentence)
                    else:
                        raise KeyError("No 'sentence' or 'sentences' in entry!")
                else:
                    if "sentence" in entry:
                        yield SentencesAndMeta(meta=meta, sentences=[entry["sentence"]])
                    elif "sentences" in entry:
                        yield SentencesAndMeta(meta=meta, sentences=entry["sentences"])
                    else:
                        raise KeyError("No 'sentence' or 'sentences' in entry!")
            except json.JSONDecodeError:
                LOGGER.warning(
                    "Error reading and parsing line %s: %s",
                    lno,
                    line[:50] if line else "no content?",
                )
            except KeyError:
                LOGGER.warning(
                    "Error extracting data from line %s: %s",
                    lno,
                    line[:50] if line else "no content?",
                )


def write_document_jsonl(
    fn: str,
    docs: Iterable[DocAndMeta],
    include_extra_meta: bool = False,
    encoding: str = ENCODING,
):
    with _open_stream(fn, "w", encoding=encoding) as fp:
        for doc in docs:
            content = doc.content
            if isinstance(content, (bytes, bytearray)):
                content = content.decode(encoding=encoding)
            line: Dict[str, Union[Optional[str], List[str]]] = {
                "location": doc.meta.location,
                "date": doc.meta.date,
                "content": content,
            }
            if (
                include_extra_meta
                and isinstance(doc.meta, SourceDocMetadata)
                and doc.meta.extra
            ):
                line["extra_meta"] = doc.meta.extra
            json.dump(line, fp)
            fp.write("\n")


def write_sentences_jsonl(
    fn: str,
    sentence_docs: Union[Iterable[SentencesAndMeta], Iterable[SentenceAndMeta]],
    encoding: str = ENCODING,
):
    with _open_stream(fn, "w", encoding=encoding) as fp:
        for sentence_doc in sentence_docs:
            line: Dict[str, Union[Optional[str], List[str]]] = {
                "location": sentence_doc.meta.location,
                "date": sentence_doc.meta.date,
            }
            if isinstance(sentence_doc, SentencesAndMeta):
                line["sentences"] = sentence_doc.sentences
            elif isinstance(sentence_doc, SentenceAndMeta):
                line["sentences"] = [sentence_doc.sentence]
            json.dump(line, fp)
            fp.write("\n")


def write_sentence_jsonl(
    fn: str,
    sentence_docs: Union[Iterable[SentencesAndMeta], Iterable[SentenceAndMeta]],
    encoding: str = ENCODING,
):
    with _open_stream(fn, "w", encoding=encoding) as fp:
        for sentence_doc in sentence_docs:
            line = {
                "location": sentence_doc.meta.location,
                "date": sentence_doc.meta.date,
            }
            if isinstance(sentence_doc, SentencesAndMeta):
                for sentence in sentence_doc.sentences:
                    line["sentence"] = sentence
                    json.dump(line, fp)
                    fp.write("\n")
            elif isinstance(sentence_doc, SentenceAndMeta):
                line["sentence"] = sentence_doc.sentence
                json.dump(line, fp)
                fp.write("\n")


# ---------------------------------------------------------------------------
# warc format

WARC_SUPPORTED = False
try:
    import warcio
    import warcio.recordbuilder
    import warcio.recordloader

    WARC_SUPPORTED = True
except ImportError:
    pass
else:

    @dataclass
    class WARCDocMetadata(DocMetadata):
        """Document metadata for WARC format.

        May contain extra metadata fields. Contains the reference to
        the original WARC record as `recordID` property when parsed.
        """

        #: metadata headers from WARC record
        warc_headers: warcio.StatusAndHeaders

        @property
        def recordID(self) -> str:
            uri = self.warc_headers.get_header("WARC-Record-ID")
            if uri is None:
                raise AttributeError("Header field 'WARC-Record-ID' not found!")
            return uri

        @classmethod
        def from_warcio_headers(
            cls: Type["WARCDocMetadata"], headers: warcio.StatusAndHeaders
        ) -> "WARCDocMetadata":
            location = headers.get_header(
                "WARC-Target-URI", DEFAULT_LOCATION_IF_MISSING
            )
            date = headers.get_header("WARC-Date", DEFAULT_DATE_IF_MISSING)[:10]
            return cls(location=location, date=date, warc_headers=headers)

    def parse_warc_docs_iter(
        fn: str,
        add_content: bool = True,
        record_types: Optional[
            Tuple[
                Literal[
                    "warcinfo",
                    "response",
                    "resource",
                    "request",
                    "metadata",
                    "revisit",
                    "conversion",
                    "continuation",
                ],
                ...,
            ]
        ] = ("response",),
    ) -> Iterator[DocAndMeta]:
        with _open_stream(fn, "rb") as fp:
            archive_iterator = warcio.ArchiveIterator(fp, no_record_parse=True)
            # NOTE: UnseekableYetTellable looks interesting

            records = cast(
                Iterator[warcio.recordloader.ArcWarcRecord], iter(archive_iterator)
            )

            for record in records:
                if record_types:
                    if record.rec_type not in record_types:
                        continue

                meta = WARCDocMetadata.from_warcio_headers(record.rec_headers)
                content = record.content_stream().read() if add_content else None

                # NOTE: access record offset/length after reading the content, otherwise it will be consumed and we get nothing
                offset = archive_iterator.get_record_offset()
                length = archive_iterator.get_record_length()
                # record.length is content length

                doc = DocAndMeta(
                    meta=meta,
                    content=content,
                    offset=offset,
                    length=length,
                )
                yield doc

    def _build_warc_record(
        record_builder: warcio.recordbuilder.RecordBuilder,
        doc: DocAndMeta,
        record_type: str,
        warc_content_type: str = "text/plain",
        allow_empty: bool = False,
        encoding: str = ENCODING,
    ) -> Optional[warcio.recordloader.ArcWarcRecord]:
        if doc.content is None:
            if not allow_empty:
                return None

            content = b""
        else:
            if isinstance(doc.content, (bytes, bytearray)):
                content = doc.content
            else:
                content = doc.content.encode(encoding)

        # warcio.recordbuilder.RecordBuilder._init_warc_headers
        record_id = record_builder._make_warc_id()
        headers = {
            "WARC-Date": f"{doc.meta.date}T00:00:00Z",
            "WARC-Record-ID": record_id,
        }

        # refersTo - link to source record
        if isinstance(doc.meta, WARCDocMetadata):
            headers["WARC-Refers-To"] = doc.meta.recordID
            # revisit fields
            # but we might want to use it for the date of our processing?
            # but what about multiple steps, does it contain the value for the initial record
            # what about the 'WARC-Refers-To' field that only points to the direct parent ...
            # headers["WARC-Refers-To-Date"] = f"{doc.meta.date}T00:00:00Z"
            # headers["WARC-Refers-To-Target-URI"] = doc.meta.location

        # NOTE: other metadata about the conversion process, or as 'metadata' record?
        # https://github.com/iipc/warc-specifications/issues/52
        # https://github.com/iipc/warc-specifications/issues/27

        # NOTE Metadata header fields?
        # WARC-Concurrent-To to point to metadata record
        # https://iipc.github.io/warc-specifications/specifications/warc-format/warc-1.1/#warc-concurrent-to

        record = record_builder.create_warc_record(
            doc.meta.location,
            record_type,
            payload=io.BytesIO(content),
            warc_content_type=warc_content_type,
            warc_headers_dict=headers,
        )
        return record

    def write_warc_docs_iter(
        fn: str,
        docs: Iterable[DocAndMeta],
        record_type: str = "conversion",
        write_empty_records: bool = False,
        encoding: str = ENCODING,
    ):
        with _open_stream(fn, "wb") as fp:
            writer = warcio.WARCWriter(fp, warc_version="WARC/1.1", gzip=False)

            for doc in docs:
                record = _build_warc_record(
                    writer,
                    doc,
                    record_type=record_type,
                    allow_empty=write_empty_records,
                    encoding=encoding,
                )
                if record:
                    writer.write_record(record)


# ---------------------------------------------------------------------------
