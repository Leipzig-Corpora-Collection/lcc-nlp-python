import concurrent.futures
import itertools
import logging
import multiprocessing
import multiprocessing.queues
import os.path
import queue
import threading
import typing
from typing import Dict
from typing import Iterator
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import lcc.io
from lcc.cleaner import SentenceCleaner
from lcc.io import DocAndMeta
from lcc.io import FileFormats
from lcc.io import SentenceAndMeta
from lcc.io import SentencesAndMeta
from lcc.io import _validate_file_format
from lcc.segmentizer import AbstractSegmentizer
from lcc.tokenizer import AbstractWordTokenizer
from lcc.util import tqdm
from lcc.workflow import _validate_file_params
from lcc.workflow import _validate_segmentizer_params
from lcc.workflow import _validate_tokenizer_params
from lcc.workflow import clean_sentences
from lcc.workflow import sentence_segment
from lcc.workflow import tokenize_sentence

# ---------------------------------------------------------------------------


LOGGER = logging.getLogger(__name__)

ENCODING = "utf-8"
TQDM = False


# ---------------------------------------------------------------------------


def _validate_ncpus(n_cpus: Optional[int] = None) -> int:
    if n_cpus is None:
        n_cpus = (os.cpu_count() or 1) - 1

    LOGGER.debug("n_cpus=%s", n_cpus)
    return n_cpus


# ---------------------------------------------------------------------------
# sentence segmentation


def _single_doc_segmentation(
    segmentizer: AbstractSegmentizer, doc: DocAndMeta, encoding: str = ENCODING
) -> SentencesAndMeta:
    sentences: List[str] = list()
    content = doc.content
    if isinstance(content, (bytes, bytearray)):
        content = content.decode(encoding=ENCODING)
    if content:
        sentences = segmentizer.segmentize(content)
    return SentencesAndMeta(meta=doc.meta, sentences=sentences)


def _single_doc_segmentation_wrapper(
    args: Union[
        Tuple[AbstractSegmentizer, DocAndMeta, str],
        Tuple[AbstractSegmentizer, DocAndMeta],
    ]
) -> SentencesAndMeta:
    return _single_doc_segmentation(*args)


T = TypeVar("T")

# TODO: not sure how best to abort/exit worker threads (either fail on closed queue or drain which might increase load for a time)


def _queue_put_until_ok(
    pqueue: "multiprocessing.queues.Queue[T]", item: T, timeout: Optional[int] = 60
):
    while True:
        try:
            pqueue.put(item, timeout=timeout)
            return
        except queue.Full:
            pass
        except ValueError:
            # return
            raise


def _queue_get_until_ok(
    gqueue: "multiprocessing.queues.Queue[T]", timeout: Optional[int] = 60
) -> Optional[T]:
    while True:
        try:
            return gqueue.get(timeout=timeout)
        except queue.Empty:
            pass
        except ValueError:
            # return None
            raise


def sentence_segment_parallel_poolimap(
    fn_input: str,
    fn_output: str,
    fmt_input: Optional[Literal[FileFormats.SOURCE, FileFormats.WARC]] = None,
    fmt_output: Optional[
        Literal[FileFormats.SOURCE, FileFormats.WARC, FileFormats.MEDUSA]
    ] = None,
    segmentizer: Optional[AbstractSegmentizer] = None,
    dn_segmentizer_resources: Optional[str] = None,
    n_cpus: Optional[int] = None,
    chunksize: int = 10,
):
    # check CPUs
    n_cpus = _validate_ncpus(n_cpus)
    if n_cpus <= 0:
        LOGGER.debug("Parallelism disabled (n_cpus=%s), just run sequentially.", n_cpus)
        sentence_segment(
            fn_input=fn_input,
            fn_output=fn_output,
            fmt_input=fmt_input,
            fmt_output=fmt_output,
            segmentizer=segmentizer,
            dn_segmentizer_resources=dn_segmentizer_resources,
        )
        return

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

    with multiprocessing.Pool(processes=n_cpus) as pool:
        # load input
        input_iter: Iterator[DocAndMeta]
        if fmt_input_detected is FileFormats.SOURCE:
            input_iter = lcc.io.parse_source_docs_iter(fn_input, add_content=True)
        elif fmt_input_detected is FileFormats.WARC:
            # TODO: might need to specifiy contenttype or other filter for WARC input?
            input_iter = lcc.io.parse_warc_docs_iter(
                fn_input, add_content=True, record_types=("response", "conversion")
            )
        else:
            raise RuntimeError(
                f"Detected input format '{fmt_input_detected}' should be valid here!"
            )

        if TQDM:
            input_iter = typing.cast(
                Iterator[DocAndMeta], tqdm(input_iter, desc="input", unit="docs")
            )

        # perform segmentation
        segmented_iter = typing.cast(
            Iterator[SentencesAndMeta],
            pool.imap(
                _single_doc_segmentation_wrapper,
                zip(itertools.repeat(segmentizer), input_iter),
                chunksize=chunksize,
            ),
        )

        if TQDM:
            segmented_iter = typing.cast(
                Iterator[SentencesAndMeta],
                tqdm(segmented_iter, desc="output", unit="docs"),
            )

        # write output
        if fmt_output_detected is FileFormats.SOURCE:

            def _sentences_to_doc(doc: SentencesAndMeta) -> DocAndMeta:
                return DocAndMeta(meta=doc.meta, content=("\n".join(doc.sentences)))

            converted_iter = map(_sentences_to_doc, segmented_iter)

            lcc.io.write_source_docs_iter(fn_output, converted_iter)

        elif fmt_output_detected is FileFormats.WARC:

            def _sentences_to_doc(doc: lcc.io.SentencesAndMeta) -> lcc.io.DocAndMeta:
                return lcc.io.DocAndMeta(
                    meta=doc.meta, content=("\n".join(doc.sentences))
                )

            converted_iter = map(_sentences_to_doc, segmented_iter)

            # TODO: contenttype default text/plain
            lcc.io.write_warc_docs_iter(fn_output, converted_iter)

        elif fmt_output_detected is FileFormats.MEDUSA:
            lcc.io.write_sentences_to_medusa_format_iter(fn_output, segmented_iter)
        else:
            raise RuntimeError(
                f"Detected output format '{fmt_output_detected}' should be valid here!"
            )


def sentence_segment_parallel_poolimap_buffered(
    fn_input: str,
    fn_output: str,
    fmt_input: Optional[Literal[FileFormats.SOURCE, FileFormats.WARC]] = None,
    fmt_output: Optional[
        Literal[FileFormats.SOURCE, FileFormats.WARC, FileFormats.MEDUSA]
    ] = None,
    segmentizer: Optional[AbstractSegmentizer] = None,
    dn_segmentizer_resources: Optional[str] = None,
    n_cpus: Optional[int] = None,
    chunksize: int = 10,
    buffer_input_size: int = 100,
    buffer_output_size: int = 500,
):
    # check CPUs
    n_cpus = _validate_ncpus(n_cpus)
    if n_cpus <= 0:
        LOGGER.debug("Parallelism disabled (n_cpus=%s), just run sequentially.", n_cpus)
        sentence_segment(
            fn_input=fn_input,
            fn_output=fn_output,
            fmt_input=fmt_input,
            fmt_output=fmt_output,
            segmentizer=segmentizer,
            dn_segmentizer_resources=dn_segmentizer_resources,
        )
        return

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

    # wrappers (thread functions)
    def produce_input(
        fn_input: str,
        fmt_input_detected: FileFormats,
        input_queue: "multiprocessing.queues.Queue[Optional[DocAndMeta]]",
    ):
        LOGGER.debug("produce_input: started")
        threading.current_thread().name = "produce_input"

        input_iter: Iterator[DocAndMeta]
        if fmt_input_detected is FileFormats.SOURCE:
            input_iter = lcc.io.parse_source_docs_iter(fn_input, add_content=True)
        elif fmt_input_detected is FileFormats.WARC:
            # TODO: might need to specifiy contenttype or other filter for WARC input?
            input_iter = lcc.io.parse_warc_docs_iter(
                fn_input, add_content=True, record_types=("response", "conversion")
            )

        # input_iter = tqdm(input_iter, desc="input", unit="docs")
        for input_item in input_iter:
            _queue_put_until_ok(input_queue, input_item)

        LOGGER.debug("produce_input: signal finished")
        _queue_put_until_ok(input_queue, None)
        LOGGER.debug("produce_input: finished")

    def consume_output(
        fn_output: str,
        fmt_output_detected: FileFormats,
        output_queue: "multiprocessing.queues.Queue[Optional[SentencesAndMeta]]",
    ):
        LOGGER.debug("consume_output: started")
        threading.current_thread().name = "consume_output"

        def _producer() -> Iterator[SentencesAndMeta]:
            while True:
                item = _queue_get_until_ok(output_queue)
                if item is None:
                    LOGGER.debug("consume_output: finished input")
                    break
                yield item

        segmented_iter = _producer()

        if fmt_output_detected is FileFormats.SOURCE:

            def _sentences_to_doc(
                doc: SentencesAndMeta,
            ) -> DocAndMeta:
                return DocAndMeta(meta=doc.meta, content=("\n".join(doc.sentences)))

            converted_iter = map(_sentences_to_doc, segmented_iter)

            lcc.io.write_source_docs_iter(fn_output, converted_iter)

        elif fmt_output_detected is FileFormats.WARC:

            def _sentences_to_doc(doc: lcc.io.SentencesAndMeta) -> lcc.io.DocAndMeta:
                return lcc.io.DocAndMeta(
                    meta=doc.meta, content=("\n".join(doc.sentences))
                )

            converted_iter = map(_sentences_to_doc, segmented_iter)

            # TODO: contenttype default text/plain
            lcc.io.write_warc_docs_iter(fn_output, converted_iter)

        elif fmt_output_detected is FileFormats.MEDUSA:
            lcc.io.write_sentences_to_medusa_format_iter(fn_output, segmented_iter)

        LOGGER.debug("consume_output: finished")

    def perform_work(
        segmentizer: AbstractSegmentizer,
        input_queue: "multiprocessing.queues.Queue[Optional[DocAndMeta]]",
        output_queue: "multiprocessing.queues.Queue[Optional[SentencesAndMeta]]",
        n_cpus: int,
        chunksize: int,
    ):
        LOGGER.debug("perform_work: started")
        threading.current_thread().name = "perform_work"

        def _producer() -> Iterator[DocAndMeta]:
            while True:
                item = _queue_get_until_ok(input_queue)
                if item is None:
                    LOGGER.debug("perform_work: finished input")
                    break
                yield item

        input_iter = _producer()
        if TQDM:
            input_iter = typing.cast(
                Iterator[DocAndMeta], tqdm(input_iter, desc="input", unit="docs")
            )

        with multiprocessing.Pool(processes=n_cpus) as pool:
            # perform segmentation
            segmented_iter = typing.cast(
                Iterator[SentencesAndMeta],
                pool.imap(
                    _single_doc_segmentation_wrapper,
                    zip(itertools.repeat(segmentizer), input_iter),
                    chunksize=chunksize,
                ),
            )
            if TQDM:
                segmented_iter = typing.cast(
                    Iterator[SentencesAndMeta],
                    tqdm(segmented_iter, desc="output", unit="docs"),
                )

            for item in segmented_iter:
                _queue_put_until_ok(output_queue, item)

        LOGGER.debug("perform_work: signal finished")
        _queue_put_until_ok(output_queue, None)
        LOGGER.debug("perform_work: finished")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        input_queue: "multiprocessing.queues.Queue[Optional[DocAndMeta]]" = (
            multiprocessing.Queue(maxsize=buffer_input_size)
        )
        output_queue: "multiprocessing.queues.Queue[Optional[SentencesAndMeta]]" = (
            multiprocessing.Queue(maxsize=buffer_output_size)
        )
        # _sentinel value is None

        LOGGER.debug("Start data and work tasks")

        # load input
        task_input = executor.submit(
            produce_input, fn_input, fmt_input_detected, input_queue
        )

        # write output
        task_output = executor.submit(
            consume_output, fn_output, fmt_output_detected, output_queue
        )

        # process
        task_process = executor.submit(
            perform_work,
            segmentizer,
            input_queue,
            output_queue,
            max(1, n_cpus - 1),
            chunksize,
        )

        for future in concurrent.futures.as_completed(
            [task_input, task_output, task_process]
        ):
            if future.exception(10) is not None:
                LOGGER.warning("Error in task: %s: %s", future, future.exception(0))
                executor.shutdown(wait=False, cancel_futures=True)
                # close queues
                input_queue.close()
                output_queue.close()

    LOGGER.debug("Finished data and work tasks")


def sentence_segment_parallel_poolexector(
    fn_input: str,
    fn_output: str,
    fmt_input: Optional[Literal[FileFormats.SOURCE, FileFormats.WARC]] = None,
    fmt_output: Optional[
        Literal[FileFormats.SOURCE, FileFormats.WARC, FileFormats.MEDUSA]
    ] = None,
    segmentizer: Optional[AbstractSegmentizer] = None,
    dn_segmentizer_resources: Optional[str] = None,
    n_cpus: Optional[int] = None,
    buffer_input_size: int = 10,
    buffer_output_size: int = 500,
    buffer_in_work_size: int = 50,
    buffer_waiting_for_order_size: int = 300,
):
    # check CPUs
    n_cpus = _validate_ncpus(n_cpus)
    if n_cpus <= 0:
        LOGGER.debug("Parallelism disabled (n_cpus=%s), just run sequentially.", n_cpus)
        sentence_segment(
            fn_input=fn_input,
            fn_output=fn_output,
            fmt_input=fmt_input,
            fmt_output=fmt_output,
            segmentizer=segmentizer,
            dn_segmentizer_resources=dn_segmentizer_resources,
        )
        return

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

    if fmt_input_detected not in (FileFormats.SOURCE, FileFormats.WARC):
        raise RuntimeError(
            f"Detected input format '{fmt_input_detected}' should be valid here!"
        )

    if fmt_output_detected not in (
        FileFormats.SOURCE,
        FileFormats.WARC,
        FileFormats.MEDUSA,
    ):
        raise RuntimeError(
            f"Detected output format '{fmt_output_detected}' should be valid here!"
        )

    # wrappers (thread functions)
    def produce_input(
        fn_input: str,
        fmt_input_detected: FileFormats,
        input_queue: "multiprocessing.queues.Queue[Tuple[int, Optional[DocAndMeta]]]",
    ):
        LOGGER.debug("produce_input: started")
        threading.current_thread().name = "produce_input"

        input_iter: Iterator[DocAndMeta]
        if fmt_input_detected is FileFormats.SOURCE:
            input_iter = lcc.io.parse_source_docs_iter(fn_input, add_content=True)
        elif fmt_input_detected is FileFormats.WARC:
            # TODO: might need to specifiy contenttype or other filter for WARC input?
            input_iter = lcc.io.parse_warc_docs_iter(
                fn_input, add_content=True, record_types=("response", "conversion")
            )

        if TQDM:
            input_iter = typing.cast(
                Iterator[DocAndMeta], tqdm(input_iter, desc="input", unit="docs")
            )

        for idx, input_item in enumerate(input_iter):
            _queue_put_until_ok(input_queue, (idx, input_item))
            # LOGGER.debug("produce_input: %s", idx)

        LOGGER.debug("produce_input: signal finished")
        _queue_put_until_ok(input_queue, (-1, None))
        LOGGER.debug("produce_input: finished")

    def consume_output(
        fn_output: str,
        fmt_output_detected: FileFormats,
        output_queue: "multiprocessing.queues.Queue[Optional[SentencesAndMeta]]",
    ):
        LOGGER.debug("consume_output: started")
        threading.current_thread().name = "consume_output"

        def _producer() -> Iterator[SentencesAndMeta]:
            while True:
                item = _queue_get_until_ok(output_queue)
                if item is None:
                    LOGGER.debug("consume_output: finished input")
                    break
                yield item

        segmented_iter = _producer()
        if TQDM:
            segmented_iter = typing.cast(
                Iterator[SentencesAndMeta],
                tqdm(segmented_iter, desc="output", unit="docs"),
            )

        if fmt_output_detected is FileFormats.SOURCE:

            def _sentences_to_doc(
                doc: SentencesAndMeta,
            ) -> DocAndMeta:
                return DocAndMeta(meta=doc.meta, content=("\n".join(doc.sentences)))

            converted_iter = map(_sentences_to_doc, segmented_iter)

            lcc.io.write_source_docs_iter(fn_output, converted_iter)

        elif fmt_output_detected is FileFormats.WARC:

            def _sentences_to_doc(doc: lcc.io.SentencesAndMeta) -> lcc.io.DocAndMeta:
                return lcc.io.DocAndMeta(
                    meta=doc.meta, content=("\n".join(doc.sentences))
                )

            converted_iter = map(_sentences_to_doc, segmented_iter)

            # TODO: contenttype default text/plain
            lcc.io.write_warc_docs_iter(fn_output, converted_iter)

        elif fmt_output_detected is FileFormats.MEDUSA:
            lcc.io.write_sentences_to_medusa_format_iter(fn_output, segmented_iter)

        LOGGER.debug("consume_output: finished")

    def perform_work(
        segmentizer: AbstractSegmentizer,
        input_queue: "multiprocessing.queues.Queue[Tuple[int, Optional[DocAndMeta]]]",
        output_queue: "multiprocessing.queues.Queue[Optional[SentencesAndMeta]]",
        n_cpus: int,
        buffer_in_work_size: int,
        buffer_waiting_for_order_size: int,
    ):
        LOGGER.debug("perform_work: started")
        threading.current_thread().name = "perform_work"

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_cpus) as executor:
            last_id: int = -1
            futures2id: Dict[concurrent.futures.Future[SentencesAndMeta], int] = dict()
            id2results: Dict[int, SentencesAndMeta] = dict()
            finished: bool = False

            while not finished:
                try:
                    while True:
                        # if full, try to wait for futures to finish
                        if len(futures2id) >= buffer_in_work_size:
                            break
                        # if already a lot of results ready just waiting for an early element, also wait to start new work
                        if len(id2results) >= buffer_waiting_for_order_size:
                            break

                        # otherwise try to get new work and start new futures
                        idxAndItem = _queue_get_until_ok(input_queue)
                        idx, item = idxAndItem if idxAndItem is not None else (-1, None)
                        # LOGGER.debug("perform_work: %s", idx)
                        if idx == -1 or item is None:
                            LOGGER.debug("perform_work: finished input")
                            finished = True
                            break

                        future = executor.submit(
                            _single_doc_segmentation, segmentizer, item
                        )
                        futures2id[future] = idx
                except queue.Empty:
                    pass

                # there is a buffer of working items, so we wait until one finishes
                # till we add more futures to wait on
                # the longer it runs the more futures will probably be finished
                for future in concurrent.futures.as_completed(futures2id.keys()):
                    idx = futures2id.pop(future)
                    result = future.result()
                    # LOGGER.debug(
                    #     "Got future for %s (waiting=%s, working=%s)",
                    #     idx,
                    #     len(id2results),
                    #     len(futures2id),
                    # )
                    id2results[idx] = result

                    #      idx == 10   - assume our current result idx
                    # 1) ...+1  < 10   - up to 8 are still missing
                    # 2) l_idx >= 10   - from 10 (we already processed or are further along - error)
                    # 3) [9]+1 == 10   - exactly 9 (what we were waiting on)

                    if last_id + 1 < idx:
                        # wait for more results if our sorting buffer is full
                        # otherwise stop waiting for results and try to add more futures
                        if len(id2results) >= buffer_waiting_for_order_size:
                            continue
                        if finished:
                            continue
                        break

                    elif last_id + 1 == idx:
                        # LOGGER.debug("Collect results for %s ...", idx)
                        while last_id + 1 in id2results.keys():
                            _queue_put_until_ok(
                                output_queue, id2results.pop(last_id + 1)
                            )
                            last_id = last_id + 1
                        LOGGER.debug(
                            "Collected results for %s--%s => %s (waiting=%s, working=%s)",
                            idx,
                            last_id,
                            last_id - idx + 1,
                            len(id2results),
                            len(futures2id),
                        )

                        # we added results to our output, we can now look for more futures
                        # except if we are finished, then we wait for futures to finish
                        # or if our waiting for results buffer is still full
                        if len(id2results) >= buffer_waiting_for_order_size:
                            continue
                        if finished:
                            continue
                        break

                    elif last_id >= idx:
                        raise RuntimeError(
                            "There should never be a smaller id than what we already have processed."
                        )

                    else:
                        raise RuntimeError("We never should arrive here?!")

        LOGGER.debug("perform_work: signal finished")
        _queue_put_until_ok(output_queue, None)
        LOGGER.debug("perform_work: finished")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        input_queue: "multiprocessing.queues.Queue[Tuple[int, Optional[DocAndMeta]]]" = multiprocessing.Queue(
            maxsize=buffer_input_size
        )
        output_queue: "multiprocessing.queues.Queue[Optional[SentencesAndMeta]]" = (
            multiprocessing.Queue(maxsize=buffer_output_size)
        )
        # _sentinel value is -1/None

        LOGGER.debug("Start data and work tasks")

        # TODO: need to check whether any task errors, then it might hang forever!

        # load input
        task_input = executor.submit(
            produce_input, fn_input, fmt_input_detected, input_queue
        )

        # write output
        task_output = executor.submit(
            consume_output, fn_output, fmt_output_detected, output_queue
        )

        # process
        task_process = executor.submit(
            perform_work,
            segmentizer,
            input_queue,
            output_queue,
            # NOTE: take care to at least have 1 CPU otherwise it might hang (ValueError in worker, waiting queues in input/output) ...
            max(1, n_cpus - 1),
            buffer_in_work_size,
            buffer_waiting_for_order_size,
        )

        for future in concurrent.futures.as_completed(
            [task_input, task_output, task_process]
        ):
            if future.exception(10) is not None:
                LOGGER.warning("Error in task: %s: %s", future, future.exception(0))
                executor.shutdown(wait=False, cancel_futures=True)
                # close queues
                input_queue.close()
                output_queue.close()

    LOGGER.debug("Finished data and work tasks")


# TODO: def sentence_segment_parallel_poolexecutor_queue
# TODO: signal handling
# TODO: only processes? --> process.pool instead of processpoolexecutors?


# poolimap ? poolimap_buffered > poolexecutor
# sentence_segment_parallel = sentence_segment_parallel_poolimap
sentence_segment_parallel = sentence_segment_parallel_poolimap_buffered
# sentence_segment_parallel = sentence_segment_parallel_poolexector


# ---------------------------------------------------------------------------
# sentence cleaning


def _single_doc_sentence_cleaning(
    cleaner: SentenceCleaner, doc: SentencesAndMeta, do_replacements: bool = True
) -> SentencesAndMeta:
    sentences: List[str] = list()
    for sentence in doc.sentences:
        sentence_cleaned = cleaner.filter_sentence(
            sentence, do_replacements=do_replacements
        )
        if sentence_cleaned:
            sentences.append(sentence_cleaned)
    return SentencesAndMeta(meta=doc.meta, sentences=sentences)


def _single_doc_sentence_cleaning_wrapper(
    args: Union[
        Tuple[SentenceCleaner, SentencesAndMeta, bool],
        Tuple[SentenceCleaner, SentencesAndMeta],
    ]
) -> SentencesAndMeta:
    return _single_doc_sentence_cleaning(*args)


def _single_sentence_cleaning(
    cleaner: SentenceCleaner,
    sentence: SentenceAndMeta,
    do_replacements: bool = True,
) -> Optional[SentenceAndMeta]:
    sentence_cleaned = cleaner.filter_sentence(
        sentence.sentence, do_replacements=do_replacements
    )
    if not sentence_cleaned:
        return None
    return SentenceAndMeta(meta=sentence.meta, sentence=sentence_cleaned)


def _single_sentence_cleaning_wrapper(
    args: Union[
        Tuple[SentenceCleaner, SentenceAndMeta, bool],
        Tuple[SentenceCleaner, SentenceAndMeta],
    ]
) -> Optional[SentenceAndMeta]:
    return _single_sentence_cleaning(*args)


def clean_sentences_parallel_poolimap(
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
    n_cpus: Optional[int] = None,
    chunksize: int = 10,
):
    # check CPUs
    n_cpus = _validate_ncpus(n_cpus)
    if n_cpus <= 0:
        LOGGER.debug("Parallelism disabled (n_cpus=%s), just run sequentially.", n_cpus)
        clean_sentences(
            fn_input=fn_input,
            fn_output=fn_output,
            fmt_input=fmt_input,
            fmt_output=fmt_output,
            cleaner=cleaner,
            do_replacements=do_replacements,
        )
        return

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

    with multiprocessing.Pool(processes=n_cpus) as pool:
        if fmt_input_detected in (FileFormats.SOURCE, FileFormats.WARC):
            # load documents
            if fmt_input_detected is FileFormats.SOURCE:
                input_source_iter = lcc.io.parse_source_docs_iter(
                    fn_input, add_content=True
                )
            elif fmt_input_detected is FileFormats.WARC:
                # TODO: might need to specifiy contenttype or other filter for WARC input?
                input_source_iter = lcc.io.parse_warc_docs_iter(
                    fn_input, add_content=True, record_types=("response", "conversion")
                )

            # split lines
            def _split_lines(doc: DocAndMeta, encoding: str = ENCODING):
                sentences: List[str] = list()
                content = doc.content
                if isinstance(content, (bytes, bytearray)):
                    content = content.decode(encoding=encoding)
                if content:
                    sentences = content.splitlines(keepends=False)
                return SentencesAndMeta(meta=doc.meta, sentences=sentences)

            source_sentences_iter = typing.cast(
                Iterator[SentencesAndMeta], map(_split_lines, input_source_iter)
            )

            if TQDM:
                source_sentences_iter = typing.cast(
                    Iterator[SentencesAndMeta],
                    tqdm(source_sentences_iter, desc="input", unit="docs"),
                )

            # perform sentence cleaning
            source_cleaned_iter = typing.cast(
                Iterator[SentencesAndMeta],
                pool.imap(
                    _single_doc_sentence_cleaning_wrapper,
                    zip(
                        itertools.repeat(cleaner),
                        source_sentences_iter,
                        itertools.repeat(do_replacements),
                    ),
                    chunksize=chunksize,
                ),
            )

            if TQDM:
                source_cleaned_iter = typing.cast(
                    Iterator[SentencesAndMeta],
                    tqdm(source_cleaned_iter, desc="output", unit="docs"),
                )

            # write results
            if fmt_output_detected is FileFormats.SOURCE:

                def _sentences_to_doc(
                    doc: SentencesAndMeta,
                ) -> DocAndMeta:
                    return DocAndMeta(meta=doc.meta, content=("\n".join(doc.sentences)))

                converted_iter = map(_sentences_to_doc, source_cleaned_iter)

                lcc.io.write_source_docs_iter(fn_output, converted_iter)
            elif fmt_output_detected is FileFormats.WARC:

                def _sentences_to_doc(
                    doc: SentencesAndMeta,
                ) -> DocAndMeta:
                    return DocAndMeta(
                        meta=doc.meta,
                        content=("\n".join(doc.sentences) if doc.sentences else None),
                    )

                converted_iter = map(_sentences_to_doc, source_cleaned_iter)

                lcc.io.write_warc_docs_iter(fn_output, converted_iter)
            elif fmt_output_detected is FileFormats.MEDUSA:
                lcc.io.write_sentences_to_medusa_format_iter(
                    fn_output, source_cleaned_iter
                )
            else:
                raise RuntimeError(
                    f"Detected output format '{fmt_output_detected}' should be valid here!"
                )

        elif fmt_input_detected is FileFormats.MEDUSA:
            # load sentences
            input_sentences_iter = lcc.io.parse_sentences_from_medusa_format_iter(
                fn_input
            )

            if TQDM:
                input_sentences_iter = typing.cast(
                    Iterator[SentenceAndMeta],
                    tqdm(input_sentences_iter, desc="input", unit="sentences"),
                )

            # perform sentence cleaning
            sentences_cleaned_iter = typing.cast(
                Iterator[Optional[SentenceAndMeta]],
                pool.imap(
                    _single_sentence_cleaning_wrapper,
                    zip(
                        itertools.repeat(cleaner),
                        input_sentences_iter,
                        itertools.repeat(do_replacements),
                    ),
                    chunksize=chunksize,
                ),
            )

            # now filter out Nones
            sentences_cleaned_filtered_iter = typing.cast(
                Iterator[SentenceAndMeta],
                (sen for sen in sentences_cleaned_iter if sen is not None),
            )

            if TQDM:
                sentences_cleaned_filtered_iter = typing.cast(
                    Iterator[SentenceAndMeta],
                    tqdm(
                        sentences_cleaned_filtered_iter, desc="output", unit="sentences"
                    ),
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


clean_sentences_parallel = clean_sentences_parallel_poolimap


# ---------------------------------------------------------------------------
# tokenization


def _single_sentence_tokenization(
    tokenizer: AbstractWordTokenizer,
    sentence: SentenceAndMeta,
    encoding: str = ENCODING,
) -> SentenceAndMeta:
    sentence_tok = tokenizer.execute(sentence.sentence)
    return SentenceAndMeta(meta=sentence.meta, sentence=sentence_tok)


def _single_sentence_tokenization_wrapper(
    args: Union[
        Tuple[AbstractWordTokenizer, SentenceAndMeta, str],
        Tuple[AbstractWordTokenizer, SentenceAndMeta],
    ]
) -> SentenceAndMeta:
    return _single_sentence_tokenization(*args)


def tokenize_sentence_parallel_poolimap(
    fn_input: str,
    fn_output: str,
    fmt_input: Optional[Literal[FileFormats.MEDUSA]] = None,
    fmt_output: Optional[Literal[FileFormats.MEDUSA, FileFormats.JSONL]] = None,
    tokenizer: Optional[AbstractWordTokenizer] = None,
    dn_tokenizer_resources: Optional[str] = None,
    n_cpus: Optional[int] = None,
    chunksize: int = 10,
):
    # check CPUs
    n_cpus = _validate_ncpus(n_cpus)
    if n_cpus <= 0:
        LOGGER.debug("Parallelism disabled (n_cpus=%s), just run sequentially.", n_cpus)
        tokenize_sentence(
            fn_input=fn_input,
            fn_output=fn_output,
            fmt_input=fmt_input,
            fmt_output=fmt_output,
            tokenizer=tokenizer,
            dn_tokenizer_resources=dn_tokenizer_resources,
        )
        return

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

    with multiprocessing.Pool(processes=n_cpus) as pool:
        # load input
        input_iter: Iterator[lcc.io.SentenceAndMeta]
        if fmt_input_detected is FileFormats.MEDUSA:
            input_iter = lcc.io.parse_sentences_from_medusa_format_iter(fn_input)
        else:
            raise RuntimeError(
                f"Detected input format '{fmt_input_detected}' should be valid here!"
            )

        if TQDM:
            input_iter = typing.cast(
                Iterator[SentenceAndMeta],
                tqdm(input_iter, desc="input", unit="sentences"),
            )

        # perform tokenization
        tokenized_iter = typing.cast(
            Iterator[SentenceAndMeta],
            pool.imap(
                _single_sentence_tokenization_wrapper,
                zip(itertools.repeat(tokenizer), input_iter),
                chunksize=chunksize,
            ),
        )

        if TQDM:
            tokenized_iter = typing.cast(
                Iterator[SentenceAndMeta],
                tqdm(tokenized_iter, desc="output", unit="sentences"),
            )

        # write output
        if fmt_output_detected is FileFormats.MEDUSA:
            lcc.io.write_sentences_to_medusa_format_iter(fn_output, tokenized_iter)
        elif fmt_output_detected is FileFormats.JSONL:
            lcc.io.write_sentence_jsonl(fn_output, tokenized_iter)
        else:
            raise RuntimeError(
                f"Detected output format '{fmt_output_detected}' should be valid here!"
            )


tokenize_sentence_parallel = tokenize_sentence_parallel_poolimap


# ---------------------------------------------------------------------------
