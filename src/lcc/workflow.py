import itertools
import logging
import os.path
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import lcc.io
import lcc.stats
from lcc.cleaner import SentenceCleaner
from lcc.io import WARC_SUPPORTED
from lcc.io import FileFormats
from lcc.io import _validate_file_format
from lcc.language.sentence import LANG_UNKNOWN
from lcc.language.sentence import LanIKernel
from lcc.segmentizer import AbstractSegmentizer
from lcc.segmentizer import LineAwareSegmentizer
from lcc.tokenizer import AbstractWordTokenizer
from lcc.tokenizer import CharacterBasedWordTokenizerImproved

# ---------------------------------------------------------------------------


LOGGER = logging.getLogger(__name__)

ENCODING = "utf-8"


# ---------------------------------------------------------------------------


def _validate_file_params(fn_input: str, fn_output: str, allow_stdinout: bool = True):
    # check filenames
    if not fn_input:
        raise ValueError("Argument 'fn_input' is required!")
    if not fn_output:
        raise ValueError("Argument 'fn_output' is required!")
    if not os.path.exists(fn_input):
        if allow_stdinout and fn_input == "-":
            return
        raise FileNotFoundError(f"Input file '{fn_input}' does not exist!")


# ---------------------------------------------------------------------------
# sentence segmentation


def _validate_segmentizer_params(
    segmentizer: Optional[AbstractSegmentizer] = None,
    dn_segmentizer_resources: Optional[str] = None,
) -> AbstractSegmentizer:
    if segmentizer is None:
        if dn_segmentizer_resources is None:
            raise ValueError(
                "Value for 'dn_segmentizer_resources' is required if 'segmentizer' is None!"
            )
        segmentizer = LineAwareSegmentizer.create_default(dn_segmentizer_resources)

    return segmentizer


def sentence_segment(
    fn_input: str,
    fn_output: str,
    fmt_input: Optional[Literal[FileFormats.SOURCE, FileFormats.WARC]] = None,
    fmt_output: Optional[
        Literal[FileFormats.SOURCE, FileFormats.WARC, FileFormats.MEDUSA]
    ] = None,
    segmentizer: Optional[AbstractSegmentizer] = None,
    dn_segmentizer_resources: Optional[str] = None,
):
    # check filenames
    _validate_file_params(fn_input, fn_output)

    # check file formats
    fmt_input_detected = _validate_file_format(
        fn_input, fmt_input, sentence_segment, "fmt_input", "input"
    )
    fmt_output_detected = _validate_file_format(
        fn_output, fmt_output, sentence_segment, "fmt_output", "output"
    )

    # check tool
    segmentizer = _validate_segmentizer_params(segmentizer, dn_segmentizer_resources)

    # load input
    input_iter: Iterator[lcc.io.DocAndMeta]
    if fmt_input_detected is FileFormats.SOURCE:
        input_iter = lcc.io.parse_source_docs_iter(fn_input, add_content=True)
    elif WARC_SUPPORTED and fmt_input_detected is FileFormats.WARC:
        # TODO: might need to specifiy contenttype or other filter for WARC input?
        input_iter = lcc.io.parse_warc_docs_iter(
            fn_input, add_content=True, record_types=("response", "conversion")
        )
    else:
        raise RuntimeError(
            f"Detected input format '{fmt_input_detected}' should be valid here!"
        )

    # perform segmentation
    def _single_doc_segmentation(
        doc: lcc.io.DocAndMeta, encoding: str = ENCODING
    ) -> lcc.io.SentencesAndMeta:
        sentences: List[str] = list()
        content = doc.content
        if isinstance(content, (bytes, bytearray)):
            content = content.decode(encoding=encoding)
        if content:
            sentences = segmentizer.segmentize(content)
        return lcc.io.SentencesAndMeta(meta=doc.meta, sentences=sentences)

    segmented_iter = map(_single_doc_segmentation, input_iter)

    # write output
    if fmt_output_detected is FileFormats.SOURCE:

        def _sentences_to_doc(doc: lcc.io.SentencesAndMeta) -> lcc.io.DocAndMeta:
            return lcc.io.DocAndMeta(meta=doc.meta, content=("\n".join(doc.sentences)))

        converted_iter = map(_sentences_to_doc, segmented_iter)

        lcc.io.write_source_docs_iter(fn_output, converted_iter)

    elif WARC_SUPPORTED and fmt_output_detected is FileFormats.WARC:

        def _sentences_to_doc(doc: lcc.io.SentencesAndMeta) -> lcc.io.DocAndMeta:
            return lcc.io.DocAndMeta(meta=doc.meta, content=("\n".join(doc.sentences)))

        converted_iter = map(_sentences_to_doc, segmented_iter)

        # TODO: contenttype default text/plain
        lcc.io.write_warc_docs_iter(fn_output, converted_iter)

    elif fmt_output_detected is FileFormats.MEDUSA:
        lcc.io.write_sentences_to_medusa_format_iter(fn_output, segmented_iter)
    else:
        raise RuntimeError(
            f"Detected output format '{fmt_output_detected}' should be valid here!"
        )


# ---------------------------------------------------------------------------
# sentence cleaning


def clean_sentences(
    fn_input: str,
    fn_output: str,
    cleaner: SentenceCleaner,
    fmt_input: Optional[
        Literal[FileFormats.SOURCE, FileFormats.WARC, FileFormats.MEDUSA]
    ] = None,
    fmt_output: Optional[
        Literal[FileFormats.SOURCE, FileFormats.WARC, FileFormats.MEDUSA]
    ] = None,
    do_replacements: bool = True,
):
    # check filenames
    _validate_file_params(fn_input, fn_output)

    # check tool
    if cleaner is None:
        raise ValueError("Argument 'cleaner' is required!")

    # check file formats
    fmt_input_detected = _validate_file_format(
        fn_input, fmt_input, clean_sentences, "fmt_input", "input"
    )
    fmt_output_detected = _validate_file_format(
        fn_output, fmt_output, clean_sentences, "fmt_output", "output"
    )

    if fmt_input_detected is FileFormats.MEDUSA and fmt_output_detected in (
        FileFormats.SOURCE,
        FileFormats.WARC,
    ):
        # NOTE: we do not want support merging single sentences back into documents, so abort
        raise ValueError(
            f"Format combination of input format '{fmt_input_detected}' and output format '{fmt_output_detected}' is not supported!"
        )

    if fmt_input_detected in (FileFormats.SOURCE, FileFormats.WARC):
        # load documents
        if fmt_input_detected is FileFormats.SOURCE:
            input_source_iter = lcc.io.parse_source_docs_iter(
                fn_input, add_content=True
            )
        elif WARC_SUPPORTED and fmt_input_detected is FileFormats.WARC:
            # TODO: might need to specifiy contenttype or other filter for WARC input?
            input_source_iter = lcc.io.parse_warc_docs_iter(
                fn_input, add_content=True, record_types=("response", "conversion")
            )
        else:
            raise RuntimeError("Unsupported input format?!")

        # split lines
        def _split_lines(doc: lcc.io.DocAndMeta, encoding: str = ENCODING):
            sentences: List[str] = list()
            content = doc.content
            if isinstance(content, (bytes, bytearray)):
                content = content.decode(encoding=encoding)
            if content:
                sentences = content.splitlines(keepends=False)
            return lcc.io.SentencesAndMeta(meta=doc.meta, sentences=sentences)

        source_sentences_iter = map(_split_lines, input_source_iter)

        # perform sentence cleaning
        def _single_doc_sentence_cleaning(
            doc: lcc.io.SentencesAndMeta,
        ) -> lcc.io.SentencesAndMeta:
            sentences: List[str] = list()
            for sentence in doc.sentences:
                sentence_cleaned = cleaner.filter_sentence(
                    sentence, do_replacements=do_replacements
                )
                if sentence_cleaned:
                    sentences.append(sentence_cleaned)
            return lcc.io.SentencesAndMeta(meta=doc.meta, sentences=sentences)

        source_cleaned_iter = map(_single_doc_sentence_cleaning, source_sentences_iter)

        # write results
        if fmt_output_detected is FileFormats.SOURCE:

            def _sentences_to_doc(doc: lcc.io.SentencesAndMeta) -> lcc.io.DocAndMeta:
                return lcc.io.DocAndMeta(
                    meta=doc.meta, content=("\n".join(doc.sentences))
                )

            converted_iter = map(_sentences_to_doc, source_cleaned_iter)

            lcc.io.write_source_docs_iter(fn_output, converted_iter)
        elif WARC_SUPPORTED and fmt_output_detected is FileFormats.WARC:

            def _sentences_to_doc(doc: lcc.io.SentencesAndMeta) -> lcc.io.DocAndMeta:
                return lcc.io.DocAndMeta(
                    meta=doc.meta,
                    content=("\n".join(doc.sentences) if doc.sentences else None),
                )

            converted_iter = map(_sentences_to_doc, source_cleaned_iter)

            lcc.io.write_warc_docs_iter(fn_output, converted_iter)
        elif fmt_output_detected is FileFormats.MEDUSA:
            lcc.io.write_sentences_to_medusa_format_iter(fn_output, source_cleaned_iter)
        else:
            raise RuntimeError(
                f"Detected output format '{fmt_output_detected}' should be valid here!"
            )

    elif fmt_input_detected is FileFormats.MEDUSA:
        # load sentences
        input_sentences_iter = lcc.io.parse_sentences_from_medusa_format_iter(fn_input)

        # perform sentence cleaning
        def _single_sentence_cleaning(
            sentence: lcc.io.SentenceAndMeta,
        ) -> Optional[lcc.io.SentenceAndMeta]:
            sentence_cleaned = cleaner.filter_sentence(
                sentence.sentence, do_replacements=do_replacements
            )
            if not sentence_cleaned:
                return None
            return lcc.io.SentenceAndMeta(meta=sentence.meta, sentence=sentence_cleaned)

        sentences_cleaned_iter = map(_single_sentence_cleaning, input_sentences_iter)

        # now filter out Nones
        sentences_cleaned_filtered_iter = (
            sen for sen in sentences_cleaned_iter if sen is not None
        )

        # write results
        if fmt_output_detected is FileFormats.MEDUSA:
            lcc.io.write_sentences_to_medusa_format_iter(
                fn_output, sentences_cleaned_filtered_iter
            )
        else:
            raise RuntimeError(
                f"Detected output format '{fmt_output_detected}' should be valid here!"
            )

    else:
        raise RuntimeError(
            f"Detected input format '{fmt_input_detected}' should be valid here!"
        )


# ---------------------------------------------------------------------------
# sentence language identification


def identify_language(
    fn_input: str,
    dn_output: str,
    lani: LanIKernel,
    fmt_input: Optional[Literal[FileFormats.SOURCE]] = None,
    fmt_output: Optional[Literal[FileFormats.SOURCE]] = FileFormats.SOURCE,
):
    # check filenames
    if not fn_input:
        raise ValueError("Argument 'fn_input' is required!")
    if not dn_output:
        raise ValueError("Argument 'dn_output' is required!")
    if not os.path.exists(fn_input) and fn_input != "-":
        raise FileNotFoundError(f"Input file '{fn_input}' does not exist!")

    # check tool
    if lani is None:
        raise ValueError("Argument 'lani' is required!")

    # check file formats
    fmt_input_detected = _validate_file_format(
        fn_input, fmt_input, identify_language, "fmt_input", "input"
    )
    fmt_output_detected = fmt_output

    if fmt_input_detected in (FileFormats.SOURCE,):
        # load documents
        if fmt_input_detected is FileFormats.SOURCE:
            input_source_iter = lcc.io.parse_source_docs_iter(
                fn_input, add_content=True
            )
        else:
            raise RuntimeError("Unsupported input format?!")

        # split lines
        def _split_lines(doc: lcc.io.DocAndMeta, encoding: str = ENCODING):
            sentences: List[str] = list()
            content = doc.content
            if isinstance(content, (bytes, bytearray)):
                content = content.decode(encoding=encoding)
            if content:
                sentences = content.splitlines(keepends=False)
            return lcc.io.SentencesAndMeta(meta=doc.meta, sentences=sentences)

        source_sentences_iter = map(_split_lines, input_source_iter)

        # perform sentence language identification
        def _single_doc_sentence_language_identification(
            doc: lcc.io.SentencesAndMeta,
        ) -> Dict[str, lcc.io.SentencesAndMeta]:
            lang_split_sentences: Dict[str, List[str]] = dict()

            for sentence in doc.sentences:
                result = lani.evaluate(sentence)
                language: str = LANG_UNKNOWN
                if result:
                    language = result.get_result().language
                if language not in lang_split_sentences:
                    lang_split_sentences[language] = list()
                lang_split_sentences[language].append(sentence)

            return {
                lang: lcc.io.SentencesAndMeta(meta=doc.meta, sentences=sents)
                for lang, sents in lang_split_sentences.items()
            }

        sources_lani_iter = map(
            _single_doc_sentence_language_identification, source_sentences_iter
        )

        # write results
        if fmt_output_detected is FileFormats.SOURCE:
            # TODO: temporary, will probably need to restructure stuff here
            for item in sources_lani_iter:
                for language, sdoc in item.items():
                    doc = lcc.io.DocAndMeta(
                        meta=sdoc.meta, content=("\n".join(sdoc.sentences))
                    )
                    fn_output = os.path.join(
                        dn_output, f"{language}.{fmt_output_detected.value}"
                    )
                    lcc.io.write_source_docs_iter(fn_output, [doc], mode="ab")
        else:
            raise RuntimeError(
                f"Detected output format '{fmt_output_detected}' should be valid here!"
            )

    else:
        raise RuntimeError(
            f"Detected input format '{fmt_input_detected}' should be valid here!"
        )


# ---------------------------------------------------------------------------
# tokenization


def _validate_tokenizer_params(
    tokenizer: Optional[AbstractWordTokenizer] = None,
    dn_tokenizer_resources: Optional[str] = None,
) -> AbstractWordTokenizer:
    if tokenizer is None:
        if dn_tokenizer_resources is None:
            raise ValueError(
                "Value for 'dn_tokenizer_resources' is required if 'tokenizer' is None!"
            )
        tokenizer = CharacterBasedWordTokenizerImproved.create_default(
            dn_tokenizer_resources
        )

    return tokenizer


def tokenize_sentence(
    fn_input: str,
    fn_output: str,
    fmt_input: Optional[Literal[FileFormats.MEDUSA]] = None,
    fmt_output: Optional[Literal[FileFormats.MEDUSA, FileFormats.JSONL]] = None,
    tokenizer: Optional[AbstractWordTokenizer] = None,
    dn_tokenizer_resources: Optional[str] = None,
):
    # check filenames
    _validate_file_params(fn_input, fn_output)

    # check file formats
    fmt_input_detected = _validate_file_format(
        fn_input, fmt_input, tokenize_sentence, "fmt_input", "input"
    )
    fmt_output_detected = _validate_file_format(
        fn_output, fmt_output, tokenize_sentence, "fmt_output", "output"
    )

    # check tool
    tokenizer = _validate_tokenizer_params(tokenizer, dn_tokenizer_resources)

    # load input
    input_iter: Iterator[lcc.io.SentenceAndMeta]
    if fmt_input_detected is FileFormats.MEDUSA:
        input_iter = lcc.io.parse_sentences_from_medusa_format_iter(fn_input)
    else:
        raise RuntimeError(
            f"Detected input format '{fmt_input_detected}' should be valid here!"
        )

    # perform segmentation
    def _single_sentence_tokenization(
        sentence: lcc.io.SentenceAndMeta, encoding: str = ENCODING
    ) -> lcc.io.SentenceAndMeta:
        sentence_tok = tokenizer.execute(sentence.sentence)
        return lcc.io.SentenceAndMeta(meta=sentence.meta, sentence=sentence_tok)

    tokenized_iter = map(_single_sentence_tokenization, input_iter)

    # write output
    if fmt_output_detected is FileFormats.MEDUSA:
        lcc.io.write_sentences_to_medusa_format_iter(fn_output, tokenized_iter)
    elif fmt_output_detected is FileFormats.JSONL:
        lcc.io.write_sentence_jsonl(fn_output, tokenized_iter)
    else:
        raise RuntimeError(
            f"Detected output format '{fmt_output_detected}' should be valid here!"
        )


# ---------------------------------------------------------------------------
# format conversion


def convert_source_to_medusa(fn_input: str, fn_output: str, encoding: str = ENCODING):
    # check filenames
    _validate_file_params(fn_input, fn_output)

    # parse input
    docs_iter = lcc.io.parse_source_docs_iter(
        fn_input, add_content=True, encoding=encoding
    )

    # convert (split on line breaks)
    converted_iter = lcc.io._convert_from_source_to_medusa(docs_iter)

    # write output
    lcc.io.write_sentences_to_medusa_format_iter(fn_output, converted_iter)


def convert_source_to_jsonl(fn_input: str, fn_output: str, encoding: str = ENCODING):
    # check filenames
    _validate_file_params(fn_input, fn_output)

    # parse input
    docs_iter = lcc.io.parse_source_docs_iter(
        fn_input, add_content=True, encoding=encoding
    )

    # write output
    lcc.io.write_document_jsonl(fn_output, docs_iter, encoding=encoding)


def convert_medusa_to_jsonl(fn_input: str, fn_output: str, encoding: str = ENCODING):
    # check filenames
    _validate_file_params(fn_input, fn_output)

    # parse input
    sentences_iter = lcc.io.parse_sentences_from_medusa_format_iter(
        fn_input, encoding=encoding
    )

    # write output
    lcc.io.write_sentence_jsonl(fn_output, sentences_iter, encoding=encoding)


def convert_source_to_warc(fn_input: str, fn_output: str, encoding: str = ENCODING):
    # check filenames
    _validate_file_params(fn_input, fn_output)

    # parse input
    docs_iter = lcc.io.parse_source_docs_iter(
        fn_input, add_content=True, encoding=encoding
    )

    # write output
    lcc.io.write_warc_docs_iter(fn_output, docs_iter, write_empty_records=True)


def convert_warc_to_source(fn_input: str, fn_output: str, encoding: str = ENCODING):
    # check filenames
    _validate_file_params(fn_input, fn_output)

    # parse input
    docs_iter = lcc.io.parse_warc_docs_iter(
        fn_input, add_content=True, record_types=("conversion", "response")
    )

    # write output
    lcc.io.write_source_docs_iter(fn_output, docs_iter, encoding=encoding)


def convert_warc_to_jsonl(fn_input: str, fn_output: str, encoding: str = ENCODING):
    # check filenames
    _validate_file_params(fn_input, fn_output)

    # parse input
    docs_iter = lcc.io.parse_warc_docs_iter(
        fn_input, add_content=True, record_types=("conversion", "response")
    )

    # write output
    lcc.io.write_document_jsonl(fn_output, docs_iter, encoding=encoding)


CONVERSIONS: Dict[
    Tuple[FileFormats, FileFormats], Callable[[str, str, str], None]
] = dict()
CONVERSIONS[(FileFormats.SOURCE, FileFormats.MEDUSA)] = convert_source_to_medusa
CONVERSIONS[(FileFormats.SOURCE, FileFormats.JSONL)] = convert_source_to_jsonl
CONVERSIONS[(FileFormats.SOURCE, FileFormats.WARC)] = convert_source_to_warc
CONVERSIONS[(FileFormats.WARC, FileFormats.SOURCE)] = convert_warc_to_source
CONVERSIONS[(FileFormats.WARC, FileFormats.JSONL)] = convert_warc_to_jsonl
CONVERSIONS[(FileFormats.MEDUSA, FileFormats.JSONL)] = convert_medusa_to_jsonl


# ---------------------------------------------------------------------------
# split/merge source


def split_source_file(
    fn_input: str,
    fn_output: str,
    maxdocs: Optional[int] = None,
    maxbytes: Optional[int] = None,
    encoding: str = ENCODING,
):
    # check filenames
    _validate_file_params(fn_input, fn_output)

    # check max* params
    if maxdocs is not None and maxdocs <= 0:
        maxdocs = None
    if maxbytes is not None and maxbytes <= 0:
        maxdocs = None
    if maxdocs is None and maxbytes is None:
        raise ValueError(
            "Both 'maxdocs' and 'maxbytes' are disabled (or have invalid values)!"
        )

    # parse input
    LOGGER.info("Load documents from input file %s", fn_input)
    docs_iter = lcc.io.parse_source_docs_iter(
        fn_input, add_content=True, encoding=encoding
    )

    # write output
    filenames = lcc.io.write_source_docs_split_iter(
        fn_output, docs_iter, maxdocs=maxdocs, maxbytes=maxbytes, encoding=encoding
    )
    LOGGER.info("Wrote %s files.", len(filenames))


def _validate_slice_params(
    start: Optional[int] = 1, stop: Optional[int] = None
) -> Tuple[Optional[int], Optional[int]]:
    # use inclusive indices, 1-based
    # so 2--3 for [1, 2, 3, 4, 5] => [2, 3]
    if start is None and stop is None:
        raise ValueError("Both 'start' and 'stop' is None! That is no slicing!")

    if start is not None and start <= 0:
        start = 1

    if stop is not None:
        if start is not None and stop <= start:
            stop = start
        elif stop <= 0:
            raise ValueError("'stop' should not be negative if 'start' is None")

    return start, stop


def slice_source_file(
    fn_input: str,
    fn_output: str,
    start: Optional[int] = 1,
    stop: Optional[int] = None,
    encoding: str = ENCODING,
):
    # check filenames
    _validate_file_params(fn_input, fn_output)

    start, stop = _validate_slice_params(start, stop)

    # parse input
    LOGGER.info("Load documents from input file %s", fn_input)
    docs_iter = lcc.io.parse_source_docs_iter(
        fn_input, add_content=True, encoding=encoding
    )

    # make indices of islice
    if start is not None:
        start -= 1

    # use itertools to extract slice
    docs_iter = itertools.islice(docs_iter, start, stop)

    # write output
    filenames = lcc.io.write_source_docs_iter(fn_output, docs_iter, encoding=encoding)
    LOGGER.info("Wrote slice [%s--%s] to %s", start, stop, len(filenames))


def merge_source_files(
    dn_input: str, fn_input_pattern: str, fn_output: str, encoding: str = ENCODING
):
    # check filenames and params
    if not dn_input:
        raise ValueError("Argument 'dn_input' is required!")
    if not fn_input_pattern:
        raise ValueError("Argument 'fn_input_pattern' is required!")
    if not fn_output:
        raise ValueError("Argument 'fn_output' is required!")
    if not os.path.exists(dn_input):
        raise FileNotFoundError(f"Input file '{dn_input}' does not exist!")

    fns_input = lcc.io._find_files(dn_input, fn_input_pattern)
    if not fns_input:
        FileNotFoundError(
            f"Did not find any files in '{dn_input}' matching pattern '{fn_input_pattern}'!"
        )

    fns_input = sorted(fns_input)

    # parse input
    docs_iter = itertools.chain.from_iterable(
        lcc.io.parse_source_docs_iter(
            str(fn_input), add_content=True, encoding=encoding
        )
        for fn_input in fns_input
    )

    # write output
    lcc.io.write_source_docs_iter(fn_output, docs_iter, encoding=encoding)


# TODO: merge and split?


# ---------------------------------------------------------------------------
# statistics


def compute_docs_stats_heuristic(
    fn_input: str,
    fmt_input: Optional[Literal[FileFormats.SOURCE, FileFormats.WARC]] = None,
    encoding: str = ENCODING,
) -> lcc.stats.DocStats:
    # check filenames
    if not fn_input:
        raise ValueError("Argument 'fn_input' is required!")
    if not os.path.exists(fn_input) and not fn_input == "-":
        raise FileNotFoundError(f"Input file '{fn_input}' does not exist!")

    # check format
    fmt_input_detected = _validate_file_format(
        fn_input, fmt_input, compute_docs_stats_heuristic, "fmt_input", "input"
    )

    doc_iter: Iterator[lcc.io.DocAndMeta]
    if fmt_input_detected is FileFormats.SOURCE:
        doc_iter = lcc.io.parse_source_docs_iter(
            fn_input, add_content=True, encoding=encoding
        )
    elif WARC_SUPPORTED and fmt_input_detected is FileFormats.WARC:
        doc_iter = lcc.io.parse_warc_docs_iter(
            fn_input, add_content=True, record_types=("response", "conversion")
        )
    else:
        raise RuntimeError("Unsupported input format?!")

    stats = lcc.stats.compute_docs_stats_heuristic(doc_iter, encoding=encoding)
    return stats


def compute_medusa_stats_heuristic(
    fn_input: str,
    count_sources: bool = True,
    encoding: str = ENCODING,
) -> lcc.stats.MedusaStats:
    # check filenames
    if not fn_input:
        raise ValueError("Argument 'fn_input' is required!")
    if not os.path.exists(fn_input) and not fn_input == "-":
        raise FileNotFoundError(f"Input file '{fn_input}' does not exist!")

    sent_iter = lcc.io.parse_sentences_from_medusa_format_iter(
        fn_input, encoding=encoding
    )

    stats = lcc.stats.compute_sentences_stats_heuristic(
        sent_iter, count_sources=count_sources, encoding=encoding
    )
    return stats


# ---------------------------------------------------------------------------
