import datetime
import logging
import os.path
import time
from contextlib import ExitStack
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from lcc.tokenizer import TOKEN_SENT_END
from lcc.tokenizer import TOKEN_SENT_START
from lcc.tokenizer import AbstractWordTokenizer
from lcc.tokenizer import MWUWordTokenizerMixin
from lcc.util import tqdm

# ---------------------------------------------------------------------------

# de.uni_leipzig.asv.medusa.config.ConfigurationContainer
#  - file names
# de.uni_leipzig.asv.medusa.controlflow.AbstractControlFlow
#  - abstract? - filters?
# de.uni_leipzig.asv.medusa.controlflow.StandardASVControlFlowImpl
#  - calls StandardASVControlFlowImpl, ...
# > de.uni_leipzig.asv.medusa.controlflow.MWUDetectionControlFlowImpl
#    - calls DefaultControlFlowImpl
# > > de.uni_leipzig.asv.medusa.controlflow.DefaultControlFlowImpl
#      - calls SentencePreparer
#      - calls WordNumbers, WordNumberTransformerImpl, DBWordListPreparerImpl
#      - calls SentenceToIDX, de.uni_leipzig.asv.medusa.filter.sidx.IDXDistinctWordFrequencyCounterFilterImpl
#      - loadMemoryAllocator, runFilterComponent, loadParser, loadExporter
#        - EXPORTER: de.uni_leipzig.asv.medusa.export.FlatFileExporterImpl (from MWU)?
#        - PARSER: de.uni_leipzig.asv.medusa.filter.sidx.IDXNeighbourhoodFilterImpl
#        - MEMORY: de.uni_leipzig.asv.medusa.config.DefaultMemoryAllocatorImpl

# de.uni_leipzig.asv.medusa.input.SentencePreparer
#  - http://www.massapi.com/class/bak/pcj/map/ObjectKeyIntOpenHashMap.html
#  - SentenceSignature, PrefixBasedStringHashFunctionImpl
# > de.uni_leipzig.asv.medusa.input.AbstractInput
#    - parsing sentence file
#    - load, parse, check for MWUs

# ---------------------------------------------------------------------------


LOGGER = logging.getLogger(__name__)

ENCODING: str = "utf-8"
TQDM_MINITERS = 1_000
TQDM_MININTERVAL = 1

SOURCE_NULL = "NULL"
DATE_NULL = "NULL"
TIMESTAMP_NOT_SET = -100


# ---------------------------------------------------------------------------
# workflow

# de.uni_leipzig.asv.medusa.controlflow.DefaultControlFlowImpl


def standard_asv_flow(
    # base input filename
    fn_raw_sentences: str,
    # tokenizer (optionally with MWU support)
    tokenizer: AbstractWordTokenizer,
    # tokenization params
    write_db_files: bool = True,
    write_paras_file: bool = True,
    # word number params
    write_wswn_file: bool = True,
    write_wnc_file: bool = True,
    # general
    overwrite: bool = False,
    show_progress: bool = True,
    encoding: str = ENCODING,
):
    tm_global_start = time.time()

    # filenames for skipping if they already exists and not overwriting
    fn_sentences_tokenized = f"{fn_raw_sentences}.tok"
    fn_sentences_for_db = f"{fn_raw_sentences}.db"
    fn_sentences_tokenized_for_db = f"{fn_raw_sentences}.tok.db"
    fn_sentences_paras = f"{fn_raw_sentences}.para_s"
    fn_sources = f"{fn_raw_sentences}.src"
    fn_word_numbers = f"{fn_raw_sentences}.internal.wn"
    fn_word_frequencies = f"{fn_raw_sentences}.internal.wf"
    fn_wswn = f"{fn_raw_sentences}.wswn"
    fn_wnc = f"{fn_raw_sentences}.wnc"
    fn_word_numbers_external = f"{fn_raw_sentences}.wn"
    fn_word_frequencies_external = f"{fn_raw_sentences}.wf"
    fn_sentences_idx = f"{fn_raw_sentences}.sidx"

    # tokenizing sentence file
    if (
        overwrite
        or not os.path.exists(fn_sources)
        or not os.path.exists(fn_sentences_tokenized)
        or (
            write_db_files
            and (
                not os.path.exists(fn_sentences_for_db)
                or not os.path.exists(fn_sentences_tokenized_for_db)
            )
        )
        or (write_paras_file and not os.path.exists(fn_sentences_paras))
    ):
        LOGGER.info("Generate tokenized sentence files ...")
        tm_tokenize_start = time.time()
        generate_sentence_files(
            fn_raw_sentences,
            tokenizer,
            write_db_files=write_db_files,
            write_paras_file=write_paras_file,
            show_progress=show_progress,
            encoding=encoding,
        )
        tm_tokenize_end = time.time()
        LOGGER.info(
            "Time spent for <tokenization> (db=%s, paraS=%s): %s",
            write_db_files,
            write_paras_file,
            datetime.timedelta(seconds=tm_tokenize_end - tm_tokenize_start),
        )
        overwrite = True

    tm_ld_sm_start = time.time()
    source_map = load_source_mapping(fn_raw_sentences, encoding=encoding)
    tm_ld_sm_end = time.time()
    LOGGER.info(
        "Time spent for <load:source-map>: %s",
        datetime.timedelta(seconds=tm_ld_sm_end - tm_ld_sm_start),
    )

    # generating word numbers
    if (
        overwrite
        or not os.path.exists(fn_word_numbers)
        or not os.path.exists(fn_word_frequencies)
    ):
        LOGGER.info("Generate raw word number and frequency files ...")
        tm_wn_start = time.time()
        generate_word_numbers(
            fn_raw_sentences,
            tokenizer,
            source_map,
            show_progress=show_progress,
            encoding=encoding,
        )
        tm_wn_end = time.time()
        LOGGER.info(
            "Time spent for <word-numbers>: %s",
            datetime.timedelta(seconds=tm_wn_end - tm_wn_start),
        )
        overwrite = True

    # generating wortschatz word numbers (wswn) and wn complete (wnc) files
    fn_known_word_numbers = "resources/tokenizer/100-wn-all.txt"
    fn_known_word_numbers = tokenizer.TOKENISATION_CHARACTERS_FILE_NAME
    if (
        overwrite
        or not os.path.exists(fn_word_numbers_external)
        or not os.path.exists(fn_word_frequencies_external)
        or (write_wswn_file and not os.path.exists(fn_wswn))
        or (write_wnc_file and not os.path.exists(fn_wnc))
    ):
        LOGGER.info("Transform to WS word numbers ...")
        tm_wswn_start = time.time()
        make_wswn(
            fn_raw_sentences,
            fn_known_word_numbers,
            write_wswn_file=write_wswn_file,
            write_wnc_file=write_wnc_file,
            show_progress=show_progress,
            encoding=encoding,
        )
        tm_wswn_end = time.time()
        LOGGER.info(
            "Time spent for <ws-word-numbers>: %s",
            datetime.timedelta(seconds=tm_wswn_end - tm_wswn_start),
        )
        overwrite = True

    tm_ld_wi_start = time.time()
    word2id = _load_word_numbers(fn_raw_sentences, encoding=encoding)
    tm_ld_wi_end = time.time()
    LOGGER.info(
        "Time spent for <load:wordid-map>: %s",
        datetime.timedelta(seconds=tm_ld_wi_end - tm_ld_wi_start),
    )

    # generating idx
    if overwrite or not os.path.exists(fn_sentences_idx):
        tm_sidx_start = time.time()
        generate_sentence_idx(
            fn_raw_sentences,
            tokenizer,
            source_map,
            word2id,
            show_progress=show_progress,
            encoding=encoding,
        )
        tm_sidx_end = time.time()
        LOGGER.info(
            "Time spent for <sidx>: %s",
            datetime.timedelta(seconds=tm_sidx_end - tm_sidx_start),
        )
        overwrite = True

    # generating bow word list

    # fill hashes ... ?!

    tm_global_end = time.time()
    LOGGER.info(
        "Total <standard_asv_flow> time spent: %s",
        datetime.timedelta(seconds=tm_global_end - tm_global_start),
    )


# ---------------------------------------------------------------------------
# sentence preparer

# de.uni_leipzig.asv.medusa.input.AbstractInput
# de.uni_leipzig.asv.medusa.input.SentencePreparer
# de.uni_leipzig.asv.tools.medusa.dependencies.FileSort.FileSort
# de.uni_leipzig.asv.tools.medusa.dependencies.FileSort.FileSortLine


# tokenizes
def generate_sentence_files(
    fn_raw_sentences: str,
    tokenizer: AbstractWordTokenizer,
    with_sentence_boundaries: bool = False,
    write_db_files: bool = True,
    write_paras_file: bool = True,
    show_progress: bool = True,
    encoding: str = ENCODING,
):
    fn_sentences_tokenized = f"{fn_raw_sentences}.tok"
    fn_sentences_for_db = f"{fn_raw_sentences}.db"
    fn_sentences_tokenized_for_db = f"{fn_raw_sentences}.tok.db"
    fn_sentences_paras = f"{fn_raw_sentences}.para_s"
    fn_sources = f"{fn_raw_sentences}.src"

    source_map: Dict[Tuple[str, str], int] = dict()

    # TODO: guess input format beforehand?

    with open(fn_raw_sentences, "r", encoding=encoding) as fp_in, open(
        fn_sentences_tokenized, "w", encoding=encoding
    ) as fp_out_tok, ExitStack() as fps_out:
        if write_db_files:
            fp_out_s_db = fps_out.enter_context(
                open(fn_sentences_for_db, "w", encoding=encoding)
            )
            fp_out_st_db = fps_out.enter_context(
                open(fn_sentences_tokenized_for_db, "w", encoding=encoding)
            )
        if write_paras_file:
            fp_out_ps = fps_out.enter_context(
                open(fn_sentences_paras, "w", encoding=encoding)
            )

        if show_progress:
            fp_in = tqdm(
                fp_in,
                unit="sentences",
                desc=f"tokenize {os.path.basename(fn_raw_sentences)}",
                miniters=TQDM_MINITERS,
                mininterval=TQDM_MININTERVAL,
            )

        sentence_id = 0

        for line in fp_in:
            line = line.rstrip("\r\n")
            parts = line.split("\t")
            # TODO: we assume we have a three column format here!

            sentence_raw = parts[0].strip()
            sentence_id += 1

            sentence_date = parts[1] if len(parts) == 3 else DATE_NULL
            sentence_source = parts[2] if len(parts) == 3 else SOURCE_NULL

            # count source_key occurrences
            source_key = (sentence_source, sentence_date)
            try:
                source_map[source_key] += 1
            except KeyError:
                source_map[source_key] = 1

            # perform tokenization
            # TODO: maybe parallel? -> ProcessPool.imap
            sentence_tokenized = tokenizer.execute(sentence_raw).strip()
            if with_sentence_boundaries:
                sentence_tokenized = (
                    f"{TOKEN_SENT_START} {sentence_tokenized} {TOKEN_SENT_END}".strip()
                )

            # write tokenized
            fp_out_tok.write(
                f"{sentence_id}\t{sentence_tokenized}\t"
                f"{sentence_date}\t{sentence_source}\n"
            )

            # write db import files
            if write_db_files:
                fp_out_s_db.write(f"{sentence_id}\t{sentence_raw}\n")
                fp_out_st_db.write(f"{sentence_id}\t{sentence_tokenized}\n")

            # write ParaS file?
            if write_paras_file:
                fp_out_ps.write(
                    f"{sentence_id}\t"
                    f"{calc_bucket_id(bytes(sentence_raw, encoding))}\t"  # or "0" if failure
                    f"{word_length_signature(sentence_raw.split(' '))}\t"
                    f"{word_length_signature(sentence_tokenized.split(' '))}\t"
                    "NULL\tNULL\tNULL\n"
                )

    LOGGER.info(
        "Read %s sentences with %s source-date keys.", sentence_id, len(source_map)
    )

    # sort
    source_map_sorted = sorted(
        source_map.items(), key=lambda x: (-x[1], x[0][1], x[0][0]), reverse=False
    )

    # write sources (<id><tab><source><tab><date><nl>)
    with open(fn_sources, "w", encoding=encoding) as fp:
        source_id = 0
        for (source, date), count in source_map_sorted:
            source_id += 1
            fp.write(f"{source_id}\t{source}\t{date}\n")


def load_source_mapping(
    fn_raw_sentences: str, encoding: str = ENCODING
) -> Dict[Tuple[str, str], int]:
    fn_sources = f"{fn_raw_sentences}.src"

    source_map: Dict[Tuple[str, str], int] = dict()

    # read sources (<id><tab><source><tab><date><nl>)
    with open(fn_sources, "r", encoding=encoding) as fp:
        for lno, line in enumerate(fp):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if not len(parts) == 3:
                LOGGER.warning(
                    "Invalid line in source mapping file at line %s"
                    ": no three column format",
                    lno,
                )
                continue

            source_id_s, source, date = parts
            try:
                source_id = int(source_id_s)
            except ValueError:
                LOGGER.warning(
                    "Invalid line in source mapping file at line %s"
                    ": source id not a number",
                    lno,
                )
                continue
            source_map[(source, date)] = source_id

    return source_map


# ---------------------------------------------------------------------------
# word numbers

# de.uni_leipzig.asv.medusa.input.WordNumbers
# de.uni_leipzig.asv.medusa.transform.WordNumberTransformerImpl
# de.uni_leipzig.asv.medusa.transform.AbstractTransformer
#  - data/wordlists/100-wn-all.txt (KNOWN_WORD_NUMBERS_FILE_NAME)
# de.uni_leipzig.asv.medusa.input.DBWordListPreparerImpl
# de.uni_leipzig.asv.medusa.input.SentenceToIDX
# de.uni_leipzig.asv.medusa.filter.sidx.IDXDistinctWordFrequencyCounterFilterImpl ?

# for MWU loading see tokenizer.py --> MWUWordTokenizerMixin.load_MWU()
# TODO: WordNumbers --> EXCLUDE_FROM_MFW_FILE_NAME ?
# TODO: --> setStatisticsProperty + MetaInformation.setXYZ


def generate_MWU_mapping_file(
    fn_mwu: str, tokenizer: AbstractWordTokenizer, encoding: str = ENCODING
) -> Tuple[MWUWordTokenizerMixin, str]:
    fn_mwu_map = f"{fn_mwu}.map"

    if isinstance(tokenizer, MWUWordTokenizerMixin):
        worker = tokenizer
    else:
        worker = MWUWordTokenizerMixin()

    worker.MWU_FILE_NAME = fn_mwu
    worker.load_MWU(tokenizer)

    with open(fn_mwu_map, "w", encoding=encoding) as fp:
        for mwu_tok, mwu in worker.MWUMap.items():
            fp.write(f"{mwu_tok}\t{mwu}\n")

    return worker, fn_mwu_map


def generate_word_numbers(
    fn_raw_sentences: str,
    tokenizer: AbstractWordTokenizer,
    source_map: Dict[Tuple[str, str], int],
    show_progress: bool = True,
    encoding: str = ENCODING,
):
    fn_sentences_tokenized = f"{fn_raw_sentences}.tok"
    fn_word_numbers = f"{fn_raw_sentences}.internal.wn"
    fn_word_frequencies = f"{fn_raw_sentences}.internal.wf"

    word_freq: Dict[str, int] = dict()
    has_mwus = isinstance(tokenizer, MWUWordTokenizerMixin)
    num_token_total: int = 0
    num_mwutoken_total: int = 0

    # read
    with open(fn_sentences_tokenized, "r", encoding=encoding) as fp:
        if show_progress:
            fp = tqdm(
                fp,
                unit="sentences",
                desc=f"count tokens from {os.path.basename(fn_sentences_tokenized)}",
                miniters=TQDM_MINITERS,
                mininterval=TQDM_MININTERVAL,
            )

        for lno, line in enumerate(fp):
            line = line.rstrip()
            if not line:
                continue

            parts = line.split("\t")
            if not len(parts) == 4:
                LOGGER.warning(
                    "Invalid tokenized sentences file format at line %s"
                    ": expected four columns",
                    lno,
                )
                continue

            sentence_id_s, sentence_tokenized, sentence_date, sentence_source = parts
            try:
                sentence_id = int(sentence_id_s)
            except ValueError:
                LOGGER.warning(
                    "Invalid tokenized sentences file format at line %s"
                    ": first column should be a number",
                    lno,
                )
                continue

            try:
                source_id = source_map[(sentence_source, sentence_date)]
            except KeyError:
                LOGGER.warning("Missing source-date key at line %s", lno)
                continue

            sentence_tokenized = sentence_tokenized.strip()

            tokens = sentence_tokenized.split(" ")
            num_token_total += len(tokens)
            for token in tokens:
                # TODO: use collections.Counter ?
                # with multiprocessing and 1000 line batches and then merge results?
                try:
                    word_freq[token] += 1
                except KeyError:
                    word_freq[token] = 1

            # same for MWU tokens
            if has_mwus:
                mwu_tokens = tokenizer.get_MWU_tokens(
                    sentence_tokenized,
                    tokenizer.get_whitespace_positions(sentence_tokenized),
                )
                num_mwutoken_total += len(mwu_tokens)
                for token in mwu_tokens:
                    try:
                        word_freq[token] += 1
                    except KeyError:
                        word_freq[token] = 1

    LOGGER.info(
        "Read %s tokens (+ %s MWU tokens), %s types in %s lines",
        num_token_total,
        num_mwutoken_total,
        len(word_freq),
        lno + 1,
    )

    # sort
    word_freq_sorted = sorted(
        word_freq.items(), key=lambda x: (-x[1], x[0]), reverse=False
    )

    # write
    with open(fn_word_numbers, "w", encoding=encoding) as fp_out_wn, open(
        fn_word_frequencies, "w", encoding=encoding
    ) as fp_out_wf:
        word_id = 0
        for word, freq in word_freq_sorted:
            word_id += 1

            fp_out_wn.write(f"{word_id}\t{word}\n")
            fp_out_wf.write(f"{word_id}\t{freq}\n")


def load_known_word_numbers(
    fn_known_word_numbers: str, encoding: str = ENCODING
) -> Tuple[int, Dict[str, int]]:
    max_known_word_number = 1
    known_word_numbers: Dict[str, int] = dict()
    if not os.path.exists(fn_known_word_numbers):
        return max_known_word_number, known_word_numbers

    with open(fn_known_word_numbers, "r", encoding=encoding) as fp:
        for lno, line in enumerate(fp):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if not len(parts) == 2:
                LOGGER.warning(
                    "Invalid known word number line format at line %s"
                    ": expected two columns",
                    lno,
                )
                continue

            word_nr_s, word = parts

            try:
                word_nr = int(word_nr_s)
            except ValueError:
                LOGGER.warning(
                    "Invalid known word number line format at line %s"
                    ": first column should be a number",
                    lno,
                )
                continue

            known_word_numbers[word] = word_nr
            max_known_word_number = max(max_known_word_number, word_nr)

    return max_known_word_number + 1, known_word_numbers


def make_wswn(
    fn_raw_sentences: str,
    fn_known_word_numbers: Optional[str] = None,
    ignore_non_found_known_words: bool = True,
    write_wswn_file: bool = True,
    write_wnc_file: bool = True,
    show_progress: bool = True,
    encoding: str = ENCODING,
):
    if not fn_known_word_numbers:
        fn_known_word_numbers = f"{fn_raw_sentences}.kwn"

    fn_wswn = f"{fn_raw_sentences}.wswn"
    fn_wnc = f"{fn_raw_sentences}.wnc"
    fn_word_numbers = f"{fn_raw_sentences}.internal.wn"
    fn_word_frequencies = f"{fn_raw_sentences}.internal.wf"
    fn_word_numbers_external = f"{fn_raw_sentences}.wn"
    fn_word_frequencies_external = f"{fn_raw_sentences}.wf"

    next_wn, word2nr = load_known_word_numbers(fn_known_word_numbers, encoding=encoding)
    # NOTE: MWU_MAP_FILE_NAME seems to be not used?
    tok2untok: Dict[str, str] = dict()

    word_entries: List[Tuple[int, str, int, int]] = list()

    # iterate found words and check if they are in the known numbers list
    with open(fn_word_numbers, "r", encoding=encoding) as fp_in_wn, open(
        fn_word_frequencies, "r", encoding=encoding
    ) as fp_in_wf:
        if show_progress:
            fp_in_wn = tqdm(
                fp_in_wn,
                unit="words",
                desc=f"read {os.path.basename(fn_word_numbers)}",
                miniters=TQDM_MINITERS,
                mininterval=TQDM_MININTERVAL,
            )

        for lno, (line_wn, line_wf) in enumerate(zip(fp_in_wn, fp_in_wf)):
            word_id_s, word = line_wn.strip().split("\t")
            word_id_s2, word_freq_s = line_wf.strip().split("\t")
            assert (
                word_id_s == word_id_s2
            ), f"Word IDs must match between word numbers and word frequencies file! (line {lno})"

            word_id = int(word_id_s)
            word_freq = int(word_freq_s)

            if word in tok2untok:
                word = tok2untok.pop(word)

            if word in word2nr:
                entry = (word2nr.pop(word), word, word_freq, word_id)
            else:
                # all not known words with the same max word number
                entry = (next_wn, word, word_freq, word_id)

            word_entries.append(entry)

    if not ignore_non_found_known_words:
        for word, word_wn in word2nr.items():
            word_entries.append((word_wn, word, 0, 0))

    # sort entries
    word_entries = sorted(
        word_entries,
        key=lambda x: (
            x[0],  # word_nr
            -x[2],  # word_freq
            x[1],  # word
            x[3],  # word_id
        ),
    )

    WSWN_map: Dict[int, int] = dict()

    # write external wn/wf files
    with open(fn_word_numbers_external, "w", encoding=encoding) as fp_out_wn, open(
        fn_word_frequencies_external, "w", encoding=encoding
    ) as fp_out_wf:
        if show_progress:
            word_entries = tqdm(
                word_entries,
                unit="words",
                desc=f"write {os.path.basename(fn_word_numbers_external)}",
                miniters=TQDM_MINITERS,
                mininterval=TQDM_MININTERVAL,
            )

        wn_cnt = next_wn - 1
        for word_nr, word, word_freq, word_id_old in word_entries:
            if word_nr >= next_wn:
                wn_cnt += 1
                word_nr = wn_cnt

            fp_out_wn.write(f"{word_nr}\t{word}\n")
            fp_out_wf.write(f"{word_nr}\t{word_freq}\n")

            if write_wswn_file and word_freq > 0:
                WSWN_map[word_id_old] = word_nr

    if write_wswn_file:
        with open(fn_wswn, "wb") as fp_out:
            for wn in WSWN_map:
                # TODO: long instead of int for large numbers?
                fp_out.write(int2bytes(wn))

    if write_wnc_file:
        with open(fn_wnc, "w", encoding=encoding) as fp_out:
            wn_cnt = next_wn - 1
            for word_nr, word, word_freq, word_id_old in word_entries:
                if word_nr >= next_wn:
                    wn_cnt += 1
                    word_nr = wn_cnt

                fp_out.write(f"{word_nr}\t{word}\t{len(word)}\t{word_freq}\n")


# ---------------------------------------------------------------------------
# sentence idx

# de.uni_leipzig.asv.medusa.input.SentenceToIDX


def _load_word_numbers(
    fn_raw_sentences: str, encoding: str = ENCODING
) -> Dict[str, int]:
    # NOTE: similar to load_known_word_numbers(), maybe consolidate?
    fn_word_numbers = f"{fn_raw_sentences}.internal.wn"

    word2id: Dict[str, int] = dict()

    with open(fn_word_numbers, "r", encoding=encoding) as fp:
        for lno, line in enumerate(fp):
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if not len(parts) == 2:
                LOGGER.warning(
                    "Invalid word number line format at line %s"
                    ": expected two columns",
                    lno,
                )
                continue

            word_nr_s, word = parts

            try:
                word_nr = int(word_nr_s)
            except ValueError:
                LOGGER.warning(
                    "Invalid word number line format at line %s"
                    ": first column should be a number",
                    lno,
                )
                continue

            word2id[word] = word_nr

    return word2id


def generate_sentence_idx(
    fn_raw_sentences: str,
    tokenizer: AbstractWordTokenizer,
    source_map: Dict[Tuple[str, str], int],
    word2id: Dict[str, int],
    show_progress: bool = True,
    encoding: str = ENCODING,
):
    # NOTE: similar to generate_word_numbers(), maybe consolidate?
    fn_sentences_tokenized = f"{fn_raw_sentences}.tok"
    fn_sentences_idx = f"{fn_raw_sentences}.sidx"

    has_mwus = isinstance(tokenizer, MWUWordTokenizerMixin)
    num_token_total: int = 0
    num_mwutoken_total: int = 0

    # read
    with open(fn_sentences_tokenized, "r", encoding=encoding) as fp_in, open(
        fn_sentences_idx, "wb"
    ) as fp_out:
        if show_progress:
            fp_in = tqdm(
                fp_in,
                unit="sentences",
                desc=f"build idx for {os.path.basename(fn_sentences_tokenized)}",
                miniters=TQDM_MINITERS,
                mininterval=TQDM_MININTERVAL,
            )

        for lno, line in enumerate(fp_in):
            line = line.rstrip()
            if not line:
                continue

            parts = line.split("\t")
            if not len(parts) == 4:
                LOGGER.warning(
                    "Invalid tokenized sentences file format at line %s"
                    ": expected four columns",
                    lno,
                )
                continue

            sentence_id_s, sentence_tokenized, sentence_date, sentence_source = parts
            try:
                sentence_id = int(sentence_id_s)
            except ValueError:
                LOGGER.warning(
                    "Invalid tokenized sentences file format at line %s"
                    ": first column should be a number",
                    lno,
                )
                continue

            # TODO: long instead of int?
            fp_out.write(int2bytes(sentence_id))

            try:
                source_id = source_map[(sentence_source, sentence_date)]
            except KeyError:
                LOGGER.warning("Missing source-date key at line %s", lno)
                continue

            # TODO: long instead of int?
            fp_out.write(int2bytes(source_id))

            try:
                timestamp = int(
                    datetime.datetime.fromisoformat(sentence_date).timestamp() // 60
                )
            except ValueError:
                timestamp = TIMESTAMP_NOT_SET

            # timestamp
            fp_out.write(int2bytes(timestamp))

            sentence_tokenized = sentence_tokenized.strip()
            tokens = sentence_tokenized.split(" ")
            num_token_total += len(tokens)
            for token in tokens:
                try:
                    word_id = word2id[token]

                    # TODO: long instead of int?
                    fp_out.write(int2bytes(word_id))
                except KeyError:
                    LOGGER.warning(
                        "Word number for word '%s' not found in sentence with id %s (line %s)",
                        token,
                        sentence_id,
                        lno,
                    )

            fp_out.write(b"\xFF\xFF\xFF\xFF")

            # same for MWU tokens
            if has_mwus:
                # NOTE: should be correct, but verify later

                whitespace_positions: List[int] = tokenizer.get_whitespace_positions(
                    sentence_tokenized
                )
                mwu_tokens = tokenizer.get_MWU_tokens(
                    sentence_tokenized, whitespace_positions
                )
                fp_out.write(int2bytes(len(mwu_tokens)))

                num_mwutoken_total += len(mwu_tokens)
                processed_tokens: Dict[int, int] = dict()
                for token in mwu_tokens:
                    try:
                        word_id = word2id[token]
                    except KeyError:
                        LOGGER.warning(
                            "Word number for MWU word '%s' not found in sentence with id %s (line %s)",
                            token,
                            sentence_id,
                            lno,
                        )
                        raise

                    # find word position of MWU token start
                    offset: int = processed_tokens.get(word_id, -1) + 1
                    mwu_offset: int = sentence_tokenized.find(token, offset)
                    processed_tokens[word_id] = mwu_offset
                    mwu_pos: int = 0

                    if mwu_offset > 0:
                        for word_pos, word_offset in enumerate(whitespace_positions, 1):
                            if word_offset + 1 == mwu_offset:
                                mwu_pos = word_pos
                                break

                    fp_out.write(b"\xFF\xFF\xFF\xFF")

                    for offset in range(len(token.split(" "))):
                        fp_out.write(int2bytes(mwu_pos + offset))

                    # TODO: long instead of int?
                    fp_out.write(int2bytes(word_id))

            else:
                fp_out.write(int2bytes(0))

            fp_out.write(b"\x00\x00\x00\x00")

    LOGGER.info(
        "Read %s tokens (+ %s MWU tokens) in %s lines",
        num_token_total,
        num_mwutoken_total,
        lno + 1,
    )

    # stats
    # SENTENCES
    # WORD_TOKENS


# ---------------------------------------------------------------------------
# sentence signatures

# de.uni_leipzig.asv.medusa.input.SentenceSignature
# - data/internal/signature-mapping.txt
# - intMinLengthReplace = 4

SIGNATURE_MAPPING: Dict[int, str] = {}
SIGNATURE_MAPPING.update({0: "null"})
SIGNATURE_MAPPING.update({i: str(i) for i in range(1, 10)})
SIGNATURE_MAPPING.update(
    {(c - (65 + 32) + 10): chr(c) for c in range(65 + 32, 65 + 32 + 26)}
)


def word_length_signature(
    tokens: List[str],
    min_len_replace: int = 4,
    mapping: Optional[Dict[str, int]] = None,
) -> str:
    if not mapping:
        mapping = SIGNATURE_MAPPING

    parts = []

    for token in tokens:
        wlen = len(token.strip())
        if wlen < min_len_replace:
            parts.append(token)
        elif wlen < len(SIGNATURE_MAPPING):
            parts.append(mapping[wlen])
        else:
            parts.append("0")

    return "".join(parts)


# ---------------------------------------------------------------------------
# hashes

# de.uni_leipzig.asv.medusa.hash.function.PrefixBasedStringHashFunctionImpl
# de.uni_leipzig.asv.medusa.hash.function.AbstractHashFunction


def calc_bucket_id(barray: bytes) -> int:
    # NOTE: see str_hash() in segmentizer.py
    #: hash value
    h = 0
    #: current character in string
    off = 0
    #: calculate hash for the first 15 characters
    tmpLen = min(len(barray), 15)
    for _ in range(tmpLen, 0, -1):
        #: unsigned to signed byte: sbyte = (((ubyte & 0xFF) ^ 0x80) - 0x80)
        h = (h * 37) + (((barray[off] & 0xFF) ^ 0x80) - 0x80)
        h = ((h & 0xFFFFFFFF) ^ 0x80000000) - 0x80000000
        off += 1
    #: process the rest of the string
    if len(barray) >= 16:
        #: only sample some characters from rest
        skip = len(barray) // 16
        for _ in range(len(barray), 15, -skip):
            h = (h * 39) + (((barray[off] & 0xFF) ^ 0x80) - 0x80)
            h = ((h & 0xFFFFFFFF) ^ 0x80000000) - 0x80000000
            off += skip
    return abs(h)


def print_bits(barray: bytes, blen: Optional[int] = None) -> str:
    # bytearray(str, 'utf-8')
    return " ".join(
        format(x, "b") for x in barray[0 : len(blen) if blen else len(barray)]
    )


def calc_buffer_size(blen: int) -> int:
    buflen = blen // 8
    if blen % 8 > 0:
        buflen += 1
    return buflen


def int2bytes(val: int) -> bytes:
    return val.to_bytes(4, "little", signed=True)


def long2bytes(val: int) -> bytes:
    return val.to_bytes(8, "little", signed=True)


def bytes2int(val: bytes) -> int:
    return int.from_bytes(val, "little", signed=True)


# ---------------------------------------------------------------------------

# SENTENCES - input file
# SOURCES - number of lines of *.src
# WORD_TOKENS - from word number computation, inv file
# WORD_TYPES - from word number file?

# ---------------------------------------------------------------------------

# WIP / TEST

if __name__ == "__main__":
    import logging
    import lcc.tokenizer

    logging.basicConfig(level=logging.INFO)

    logging.getLogger().warning("TESTING ONLY!")

    tokenizer = lcc.tokenizer.CharacterBasedWordTokenizerImproved.create_default(
        "resources/tokenizer"
    )
    standard_asv_flow("data-medusa-new/eng_news-economy_2023.medusa", tokenizer)


# ---------------------------------------------------------------------------
