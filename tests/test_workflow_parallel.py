import typing
from pathlib import Path
from typing import List
from typing import Optional

import pytest

if typing.TYPE_CHECKING:  # pragma: no cover
    from pytest_mock import MockerFixture

import lcc.io
import lcc.segmentizer
import lcc.tokenizer
import lcc.workflow
import lcc.workflow_parallel

# ---------------------------------------------------------------------------


# NOTE: needs to be local since it needs to be serializeable/pickleable for multiprocessing
class LineSegmentizer(lcc.segmentizer.AbstractSegmentizer):
    def init(self):
        pass

    def segmentize(self, text: str) -> List[str]:
        if not text:
            return []  # pragma: no cover
        return text.splitlines(keepends=False)


class NoOpTokenizer(lcc.tokenizer.AbstractWordTokenizer):
    def init(self):
        pass

    def execute(self, line: str) -> str:
        return line


class RemoveOddLenSentenceCleaner(lcc.cleaner.SentenceCleaner):
    def __init__(self) -> None:
        pass

    def filter_sentence(
        self, sentence: str, do_replacements: bool = True
    ) -> Optional[str]:
        if len(sentence) % 2 == 1:
            return None
        return sentence


@pytest.fixture(scope="module")
def mock_line_segmentizer() -> lcc.segmentizer.AbstractSegmentizer:
    """A really basic segmentizer that just splits on new line breaks."""
    return LineSegmentizer()


@pytest.fixture(scope="module")
def mock_sentence_tokenizer() -> lcc.tokenizer.AbstractWordTokenizer:
    """A NoOp tokenizer that returns the sentence as is."""
    return NoOpTokenizer()


@pytest.fixture(scope="module")
def sample_docs() -> List[lcc.io.DocAndMeta]:
    """A list of sample documents."""
    N_DOCS = 10

    docs: List[lcc.io.DocAndMeta] = []
    for i in range(N_DOCS):
        meta = lcc.io.DocMetadata(f"url://{i}", f"0000-00-{i:02d}")
        doc = lcc.io.DocAndMeta(
            meta=meta,
            content="".join(f"A demo sentence, nr. {i}.\n" for j in range(i + 1)),
        )
        docs.append(doc)

    return docs


@pytest.fixture(scope="function")
def sample_source_file(tmp_path: Path, sample_docs: List[lcc.io.DocAndMeta]) -> Path:
    """A generated sample *.source file."""
    fn_source = tmp_path / "sample.source"

    lcc.io.write_source_docs_iter(str(fn_source), sample_docs)

    return fn_source


@pytest.fixture(scope="module")
def sample_sentences() -> List[lcc.io.SentencesAndMeta]:
    """A list of sample documents."""
    N_DOCS = 10

    docs: List[lcc.io.SentencesAndMeta] = []
    for i in range(N_DOCS):
        meta = lcc.io.DocMetadata(f"url://{i}", f"0000-01-{i:02d}")
        doc = lcc.io.SentencesAndMeta(
            meta=meta,
            sentences=[f"A demo sentence, nr. {i}." for j in range(i + 1)],
        )
        docs.append(doc)

    return docs


@pytest.fixture(scope="function")
def sample_medusa_file(
    tmp_path: Path, sample_sentences: List[lcc.io.SentencesAndMeta]
) -> Path:
    """A generated sample *.medusa file."""
    fn_medusa = tmp_path / "sample.medusa"

    lcc.io.write_sentences_to_medusa_format_iter(str(fn_medusa), sample_sentences)

    return fn_medusa


# ---------------------------------------------------------------------------
# helpers


def test__validate_ncpus(mocker: "MockerFixture"):
    patched = mocker.patch("os.cpu_count", autospec=True)

    patched.return_value = 42
    assert lcc.workflow_parallel._validate_ncpus(None) == 41
    patched.assert_called_once()
    patched.reset_mock()

    assert lcc.workflow_parallel._validate_ncpus(2) == 2
    patched.assert_not_called()


# ---------------------------------------------------------------------------
# sentence segmentation


@pytest.mark.skip(reason="debug: just to test the mock-spy stuff")
def test_parse_source_docs_iter(
    mocker: "MockerFixture",
    sample_source_file: Path,
    sample_docs: List[lcc.io.DocAndMeta],
):  # pragma: no cover
    spy = mocker.spy(lcc.io, "parse_source_docs_iter")

    input_iter = lcc.io.parse_source_docs_iter(
        str(sample_source_file), add_content=True
    )
    print(input_iter)
    docs = list(input_iter)
    assert len(docs) == 10
    assert {d.meta.location for d in docs} == {d.meta.location for d in sample_docs}

    spy.assert_called_once_with(str(sample_source_file), add_content=True)
    assert input_iter == spy.spy_return


def test_sentence_segment_parallel_poolimap(
    mocker: "MockerFixture",
    tmp_path: Path,
    sample_source_file: Path,
    mock_line_segmentizer: lcc.segmentizer.AbstractSegmentizer,
):
    fn_output_seq = tmp_path / "output.seq.source"
    fn_output_par = tmp_path / "output.par.source"

    spy_vfp = mocker.spy(lcc.workflow, "_validate_file_params")
    spy_vff = mocker.spy(lcc.workflow, "_validate_file_format")
    spy_vsp = mocker.spy(lcc.workflow, "_validate_segmentizer_params")

    lcc.workflow.sentence_segment(
        str(sample_source_file), str(fn_output_seq), segmentizer=mock_line_segmentizer
    )
    assert spy_vfp.call_count == 1
    assert spy_vff.call_count == 2
    assert spy_vsp.call_count == 1

    # NOTE: need to mock/spy the imported symbol, not the original module/declaration
    spy_vfp_par = mocker.spy(lcc.workflow_parallel, "_validate_file_params")
    spy_vff_par = mocker.spy(lcc.workflow_parallel, "_validate_file_format")
    spy_vsp_par = mocker.spy(lcc.workflow_parallel, "_validate_segmentizer_params")

    spy_seq = mocker.spy(lcc.workflow_parallel, "sentence_segment")

    # be sure that we are a multi-cpu system
    mock_ncpus = mocker.patch("lcc.workflow_parallel._validate_ncpus")
    mock_ncpus.return_value = 1

    # well, we mock-spy the function with spy_seq that is used for its annotations
    # so we need to do this manually, to simulate the call results for our mock check
    mocker.stop(spy_vff)
    spy_vff_par.side_effect = [
        lcc.workflow._validate_file_format(*call.args) for call in spy_vff.mock_calls
    ]
    spy_vff = mocker.spy(lcc.workflow, "_validate_file_format")

    lcc.workflow_parallel.sentence_segment_parallel_poolimap(
        str(sample_source_file), str(fn_output_par), segmentizer=mock_line_segmentizer
    )

    # that is the important stuff
    assert fn_output_seq.read_bytes() == fn_output_par.read_bytes()

    # sequentiall call stuff
    spy_vff.assert_not_called()  # since we did reset, should not have been called
    assert spy_vfp.call_count == 1
    assert spy_vsp.call_count == 1
    # parallel call stuff
    assert spy_vfp_par.call_count == 1
    assert spy_vff_par.call_count == 2
    assert spy_vsp_par.call_count == 1
    mock_ncpus.assert_called_once_with(None)
    spy_seq.assert_not_called()  # we did not call the seq one from the parallel one

    # now simulate no multi core CPU, so sequential call
    mocker.stopall()
    mock_seq = mocker.patch("lcc.workflow_parallel.sentence_segment")
    mock_vfp_par = mocker.patch("lcc.workflow_parallel._validate_file_params")
    mock_ncpus = mocker.patch("lcc.workflow_parallel._validate_ncpus")
    mock_ncpus.return_value = 0  # so only a single core

    lcc.workflow_parallel.sentence_segment_parallel_poolimap(
        str(sample_source_file),
        str(fn_output_par),
        segmentizer=mock_line_segmentizer,
    )

    # call was redirected to sequential method
    mock_seq.assert_called_once_with(
        fn_input=str(sample_source_file),
        fn_output=str(fn_output_par),
        fmt_input=None,
        fmt_output=None,
        segmentizer=mock_line_segmentizer,
        dn_segmentizer_resources=None,
    )
    # and no further checks performed (that are done in the sequential one)
    mock_vfp_par.assert_not_called()
    mock_ncpus.assert_called_once_with(None)


# test logic is same as base parallel version 'sentence_segment_parallel_poolimap'
def test_sentence_segment_parallel_poolimap_buffered(
    mocker: "MockerFixture",
    tmp_path: Path,
    sample_source_file: Path,
    mock_line_segmentizer: lcc.segmentizer.AbstractSegmentizer,
):
    fn_output_seq = tmp_path / "output.seq.source"
    fn_output_par = tmp_path / "output.par.source"

    spy_vfp = mocker.spy(lcc.workflow, "_validate_file_params")
    spy_vff = mocker.spy(lcc.workflow, "_validate_file_format")
    spy_vsp = mocker.spy(lcc.workflow, "_validate_segmentizer_params")

    lcc.workflow.sentence_segment(
        str(sample_source_file), str(fn_output_seq), segmentizer=mock_line_segmentizer
    )
    assert spy_vfp.call_count == 1
    assert spy_vff.call_count == 2
    assert spy_vsp.call_count == 1

    # NOTE: need to mock/spy the imported symbol, not the original module/declaration
    spy_vfp_par = mocker.spy(lcc.workflow_parallel, "_validate_file_params")
    spy_vff_par = mocker.spy(lcc.workflow_parallel, "_validate_file_format")
    spy_vsp_par = mocker.spy(lcc.workflow_parallel, "_validate_segmentizer_params")

    spy_seq = mocker.spy(lcc.workflow_parallel, "sentence_segment")

    # be sure that we are a multi-cpu system
    mock_ncpus = mocker.patch("lcc.workflow_parallel._validate_ncpus")
    mock_ncpus.return_value = 1

    # well, we mock-spy the function with spy_seq that is used for its annotations
    # so we need to do this manually, to simulate the call results for our mock check
    mocker.stop(spy_vff)
    spy_vff_par.side_effect = [
        lcc.workflow._validate_file_format(*call.args) for call in spy_vff.mock_calls
    ]
    spy_vff = mocker.spy(lcc.workflow, "_validate_file_format")

    lcc.workflow_parallel.sentence_segment_parallel_poolimap_buffered(
        str(sample_source_file), str(fn_output_par), segmentizer=mock_line_segmentizer
    )

    # that is the important stuff
    assert fn_output_seq.read_bytes() == fn_output_par.read_bytes()

    # sequentiall call stuff
    spy_vff.assert_not_called()  # since we did reset, should not have been called
    assert spy_vfp.call_count == 1
    assert spy_vsp.call_count == 1
    # parallel call stuff
    assert spy_vfp_par.call_count == 1
    assert spy_vff_par.call_count == 2
    assert spy_vsp_par.call_count == 1
    mock_ncpus.assert_called_once_with(None)
    spy_seq.assert_not_called()  # we did not call the seq one from the parallel one

    # now simulate no multi core CPU, so sequential call
    mocker.stopall()
    mock_seq = mocker.patch("lcc.workflow_parallel.sentence_segment")
    mock_vfp_par = mocker.patch("lcc.workflow_parallel._validate_file_params")
    mock_ncpus = mocker.patch("lcc.workflow_parallel._validate_ncpus")
    mock_ncpus.return_value = 0  # so only a single core

    lcc.workflow_parallel.sentence_segment_parallel_poolimap_buffered(
        str(sample_source_file),
        str(fn_output_par),
        segmentizer=mock_line_segmentizer,
    )

    # call was redirected to sequential method
    mock_seq.assert_called_once_with(
        fn_input=str(sample_source_file),
        fn_output=str(fn_output_par),
        fmt_input=None,
        fmt_output=None,
        segmentizer=mock_line_segmentizer,
        dn_segmentizer_resources=None,
    )
    # and no further checks performed (that are done in the sequential one)
    mock_vfp_par.assert_not_called()
    mock_ncpus.assert_called_once_with(None)


def test_sentence_segment_parallel_poolexector(
    mocker: "MockerFixture",
    tmp_path: Path,
    sample_source_file: Path,
    mock_line_segmentizer: lcc.segmentizer.AbstractSegmentizer,
):
    fn_output_seq = tmp_path / "output.seq.source"
    fn_output_par = tmp_path / "output.par.source"

    spy_vfp = mocker.spy(lcc.workflow, "_validate_file_params")
    spy_vff = mocker.spy(lcc.workflow, "_validate_file_format")
    spy_vsp = mocker.spy(lcc.workflow, "_validate_segmentizer_params")

    lcc.workflow.sentence_segment(
        str(sample_source_file), str(fn_output_seq), segmentizer=mock_line_segmentizer
    )
    assert spy_vfp.call_count == 1
    assert spy_vff.call_count == 2
    assert spy_vsp.call_count == 1

    # NOTE: need to mock/spy the imported symbol, not the original module/declaration
    spy_vfp_par = mocker.spy(lcc.workflow_parallel, "_validate_file_params")
    spy_vff_par = mocker.spy(lcc.workflow_parallel, "_validate_file_format")
    spy_vsp_par = mocker.spy(lcc.workflow_parallel, "_validate_segmentizer_params")

    spy_seq = mocker.spy(lcc.workflow_parallel, "sentence_segment")

    # be sure that we are a multi-cpu system
    mock_ncpus = mocker.patch("lcc.workflow_parallel._validate_ncpus")
    mock_ncpus.return_value = 1

    # well, we mock-spy the function with spy_seq that is used for its annotations
    # so we need to do this manually, to simulate the call results for our mock check
    mocker.stop(spy_vff)
    spy_vff_par.side_effect = [
        lcc.workflow._validate_file_format(*call.args) for call in spy_vff.mock_calls
    ]
    spy_vff = mocker.spy(lcc.workflow, "_validate_file_format")

    lcc.workflow_parallel.sentence_segment_parallel_poolexector(
        str(sample_source_file), str(fn_output_par), segmentizer=mock_line_segmentizer
    )

    # that is the important stuff
    assert fn_output_seq.read_bytes() == fn_output_par.read_bytes()

    # sequentiall call stuff
    spy_vff.assert_not_called()  # since we did reset, should not have been called
    assert spy_vfp.call_count == 1
    assert spy_vsp.call_count == 1
    # parallel call stuff
    assert spy_vfp_par.call_count == 1
    assert spy_vff_par.call_count == 2
    assert spy_vsp_par.call_count == 1
    mock_ncpus.assert_called_once_with(None)
    spy_seq.assert_not_called()  # we did not call the seq one from the parallel one

    # now simulate no multi core CPU, so sequential call
    mocker.stopall()
    mock_seq = mocker.patch("lcc.workflow_parallel.sentence_segment")
    mock_vfp_par = mocker.patch("lcc.workflow_parallel._validate_file_params")
    mock_ncpus = mocker.patch("lcc.workflow_parallel._validate_ncpus")
    mock_ncpus.return_value = 0  # so only a single core

    lcc.workflow_parallel.sentence_segment_parallel_poolexector(
        str(sample_source_file),
        str(fn_output_par),
        segmentizer=mock_line_segmentizer,
    )

    # call was redirected to sequential method
    mock_seq.assert_called_once_with(
        fn_input=str(sample_source_file),
        fn_output=str(fn_output_par),
        fmt_input=None,
        fmt_output=None,
        segmentizer=mock_line_segmentizer,
        dn_segmentizer_resources=None,
    )
    # and no further checks performed (that are done in the sequential one)
    mock_vfp_par.assert_not_called()
    mock_ncpus.assert_called_once_with(None)


# ---------------------------------------------------------------------------
# sentence cleaning


def test_clean_sentences_parallel_poolimap_s2s(
    mocker: "MockerFixture", tmp_path: Path, sample_source_file: Path
):
    mock_cleaner: lcc.cleaner.SentenceCleaner = RemoveOddLenSentenceCleaner()

    fn_output_seq = tmp_path / "output.seq.source"
    fn_output_par = tmp_path / "output.par.source"

    spy_vfp = mocker.spy(lcc.workflow, "_validate_file_params")
    spy_vff = mocker.spy(lcc.workflow, "_validate_file_format")

    lcc.workflow.clean_sentences(
        str(sample_source_file), str(fn_output_seq), cleaner=mock_cleaner
    )
    assert spy_vfp.call_count == 1
    assert spy_vff.call_count == 2

    # NOTE: need to mock/spy the imported symbol, not the original module/declaration
    spy_vfp_par = mocker.spy(lcc.workflow_parallel, "_validate_file_params")
    spy_vff_par = mocker.spy(lcc.workflow_parallel, "_validate_file_format")

    spy_seq = mocker.spy(lcc.workflow_parallel, "clean_sentences")

    # be sure that we are a multi-cpu system
    mock_ncpus = mocker.patch("lcc.workflow_parallel._validate_ncpus")
    mock_ncpus.return_value = 1

    # well, we mock-spy the function with spy_seq that is used for its annotations
    # so we need to do this manually, to simulate the call results for our mock check
    mocker.stop(spy_vff)
    spy_vff_par.side_effect = [
        lcc.workflow._validate_file_format(*call.args) for call in spy_vff.mock_calls
    ]
    spy_vff = mocker.spy(lcc.workflow, "_validate_file_format")

    lcc.workflow_parallel.clean_sentences_parallel_poolimap(
        str(sample_source_file), str(fn_output_par), cleaner=mock_cleaner
    )

    # that is the important stuff
    assert fn_output_seq.read_bytes() == fn_output_par.read_bytes()

    # sequentiall call stuff
    spy_vff.assert_not_called()  # since we did reset, should not have been called
    assert spy_vfp.call_count == 1
    # parallel call stuff
    assert spy_vfp_par.call_count == 1
    assert spy_vff_par.call_count == 2
    mock_ncpus.assert_called_once_with(None)
    spy_seq.assert_not_called()  # we did not call the seq one from the parallel one

    # now simulate no multi core CPU, so sequential call
    mocker.stopall()
    mock_seq = mocker.patch("lcc.workflow_parallel.clean_sentences")
    mock_vfp_par = mocker.patch("lcc.workflow_parallel._validate_file_params")
    mock_ncpus = mocker.patch("lcc.workflow_parallel._validate_ncpus")
    mock_ncpus.return_value = 0  # so only a single core

    lcc.workflow_parallel.clean_sentences_parallel_poolimap(
        str(sample_source_file), str(fn_output_par), cleaner=mock_cleaner
    )

    # call was redirected to sequential method
    mock_seq.assert_called_once_with(
        fn_input=str(sample_source_file),
        fn_output=str(fn_output_par),
        fmt_input=None,
        fmt_output=None,
        cleaner=mock_cleaner,
        do_replacements=True,
    )
    # and no further checks performed (that are done in the sequential one)
    mock_vfp_par.assert_not_called()
    mock_ncpus.assert_called_once_with(None)


def test_clean_sentences_parallel_poolimap_x2x(
    mocker: "MockerFixture",
    tmp_path: Path,
    sample_source_file: Path,
    sample_medusa_file: Path,
):
    mock_cleaner: lcc.cleaner.SentenceCleaner = RemoveOddLenSentenceCleaner()

    fn_output_seq_m = tmp_path / "output.seq.medusa"
    fn_output_par_m = tmp_path / "output.par.medusa"
    fn_output_seq_s = tmp_path / "output.seq.source"
    fn_output_par_s = tmp_path / "output.par.source"
    fn_output_seq_sm = tmp_path / "output.seq.s2m.source"
    fn_output_par_sm = tmp_path / "output.par.s2m.source"

    spy_vff = mocker.spy(lcc.workflow, "_validate_file_format")

    # generate test data
    with pytest.raises(ValueError):
        lcc.workflow.clean_sentences(
            str(sample_medusa_file), str(fn_output_seq_s), cleaner=mock_cleaner
        )

    lcc.workflow.clean_sentences(
        str(sample_medusa_file), str(fn_output_seq_m), cleaner=mock_cleaner
    )
    lcc.workflow.clean_sentences(
        str(sample_source_file), str(fn_output_seq_s), cleaner=mock_cleaner
    )
    lcc.workflow.clean_sentences(
        str(sample_source_file), str(fn_output_seq_sm), cleaner=mock_cleaner
    )

    # well, we mock-spy the function with spy_seq that is used for its annotations
    # so we need to do this manually, to simulate the call results for our mock check
    # note that the format combinations for seq and par need to be the same!
    mocker.stop(spy_vff)
    spy_vff_par = mocker.spy(lcc.workflow_parallel, "_validate_file_format")
    spy_vff_par.side_effect = [
        lcc.workflow._validate_file_format(*call.args) for call in spy_vff.mock_calls
    ]
    spy_seq = mocker.spy(lcc.workflow_parallel, "clean_sentences")

    # be sure that we are a multi-cpu system
    mock_ncpus = mocker.patch("lcc.workflow_parallel._validate_ncpus")
    mock_ncpus.return_value = 1

    with pytest.raises(ValueError):
        lcc.workflow_parallel.clean_sentences_parallel_poolimap(
            str(sample_medusa_file), str(fn_output_par_s), cleaner=mock_cleaner
        )
    mock_ncpus.assert_called_once_with(None)
    mock_ncpus.reset_mock()

    lcc.workflow_parallel.clean_sentences_parallel_poolimap(
        str(sample_medusa_file), str(fn_output_par_m), cleaner=mock_cleaner
    )
    mock_ncpus.assert_called_once_with(None)
    mock_ncpus.reset_mock()
    lcc.workflow_parallel.clean_sentences_parallel_poolimap(
        str(sample_source_file), str(fn_output_par_s), cleaner=mock_cleaner
    )
    mock_ncpus.assert_called_once_with(None)
    mock_ncpus.reset_mock()
    lcc.workflow_parallel.clean_sentences_parallel_poolimap(
        str(sample_source_file), str(fn_output_par_sm), cleaner=mock_cleaner
    )
    mock_ncpus.assert_called_once_with(None)
    mock_ncpus.reset_mock()

    spy_seq.assert_not_called()  # we did not call the seq one from the parallel one

    # that is the important stuff
    assert fn_output_seq_m.read_bytes() == fn_output_par_m.read_bytes()
    assert fn_output_seq_s.read_bytes() == fn_output_par_s.read_bytes()
    assert fn_output_seq_sm.read_bytes() == fn_output_par_sm.read_bytes()

    # now simulate no multi core CPU, so sequential call
    mocker.stopall()
    mock_seq = mocker.patch("lcc.workflow_parallel.clean_sentences")
    mock_vfp_par = mocker.patch("lcc.workflow_parallel._validate_file_params")
    mock_ncpus = mocker.patch("lcc.workflow_parallel._validate_ncpus")
    mock_ncpus.return_value = 0  # so only a single core

    # call was redirected to sequential method

    lcc.workflow_parallel.clean_sentences_parallel_poolimap(
        str(sample_medusa_file), str(fn_output_par_m), cleaner=mock_cleaner
    )
    mock_seq.assert_called_once_with(
        fn_input=str(sample_medusa_file),
        fn_output=str(fn_output_par_m),
        fmt_input=None,
        fmt_output=None,
        cleaner=mock_cleaner,
        do_replacements=True,
    )
    # and no further checks performed (that are done in the sequential one)
    mock_vfp_par.assert_not_called()
    mock_ncpus.assert_called_once_with(None)
    mock_seq.reset_mock()
    mock_ncpus.reset_mock()
    lcc.workflow_parallel.clean_sentences_parallel_poolimap(
        str(sample_source_file), str(fn_output_par_s), cleaner=mock_cleaner
    )
    mock_seq.assert_called_once_with(
        fn_input=str(sample_source_file),
        fn_output=str(fn_output_par_s),
        fmt_input=None,
        fmt_output=None,
        cleaner=mock_cleaner,
        do_replacements=True,
    )
    # and no further checks performed (that are done in the sequential one)
    mock_vfp_par.assert_not_called()
    mock_ncpus.assert_called_once_with(None)
    mock_seq.reset_mock()
    mock_ncpus.reset_mock()
    lcc.workflow_parallel.clean_sentences_parallel_poolimap(
        str(sample_source_file), str(fn_output_par_sm), cleaner=mock_cleaner
    )
    mock_seq.assert_called_once_with(
        fn_input=str(sample_source_file),
        fn_output=str(fn_output_par_sm),
        fmt_input=None,
        fmt_output=None,
        cleaner=mock_cleaner,
        do_replacements=True,
    )
    # and no further checks performed (that are done in the sequential one)
    mock_vfp_par.assert_not_called()
    mock_ncpus.assert_called_once_with(None)


# TODO: warc parallel tests


# ---------------------------------------------------------------------------
# tokenization


def test_tokenize_sentence_parallel_poolimap(
    mocker: "MockerFixture",
    tmp_path: Path,
    sample_medusa_file: Path,
    mock_sentence_tokenizer: lcc.tokenizer.AbstractWordTokenizer,
):
    fn_output_seq = tmp_path / "output.seq.medusa"
    fn_output_par = tmp_path / "output.par.medusa"

    spy_vfp = mocker.spy(lcc.workflow, "_validate_file_params")
    spy_vff = mocker.spy(lcc.workflow, "_validate_file_format")
    spy_vsp = mocker.spy(lcc.workflow, "_validate_tokenizer_params")

    lcc.workflow.tokenize_sentence(
        str(sample_medusa_file), str(fn_output_seq), tokenizer=mock_sentence_tokenizer
    )
    assert spy_vfp.call_count == 1
    assert spy_vff.call_count == 2
    assert spy_vsp.call_count == 1

    # NOTE: need to mock/spy the imported symbol, not the original module/declaration
    spy_vfp_par = mocker.spy(lcc.workflow_parallel, "_validate_file_params")
    spy_vff_par = mocker.spy(lcc.workflow_parallel, "_validate_file_format")
    spy_vsp_par = mocker.spy(lcc.workflow_parallel, "_validate_tokenizer_params")

    spy_seq = mocker.spy(lcc.workflow_parallel, "tokenize_sentence")

    # be sure that we are a multi-cpu system
    mock_ncpus = mocker.patch("lcc.workflow_parallel._validate_ncpus")
    mock_ncpus.return_value = 1

    # well, we mock-spy the function with spy_seq that is used for its annotations
    # so we need to do this manually, to simulate the call results for our mock check
    mocker.stop(spy_vff)
    spy_vff_par.side_effect = [
        lcc.workflow._validate_file_format(*call.args) for call in spy_vff.mock_calls
    ]
    spy_vff = mocker.spy(lcc.workflow, "_validate_file_format")

    lcc.workflow_parallel.tokenize_sentence_parallel(
        str(sample_medusa_file), str(fn_output_par), tokenizer=mock_sentence_tokenizer
    )

    # that is the important stuff
    assert fn_output_seq.read_bytes() == fn_output_par.read_bytes()

    # sequentiall call stuff
    spy_vff.assert_not_called()  # since we did reset, should not have been called
    assert spy_vfp.call_count == 1
    assert spy_vsp.call_count == 1
    # parallel call stuff
    assert spy_vfp_par.call_count == 1
    assert spy_vff_par.call_count == 2
    assert spy_vsp_par.call_count == 1
    mock_ncpus.assert_called_once_with(None)
    spy_seq.assert_not_called()  # we did not call the seq one from the parallel one

    # now simulate no multi core CPU, so sequential call
    mocker.stopall()
    mock_seq = mocker.patch("lcc.workflow_parallel.tokenize_sentence")
    mock_vfp_par = mocker.patch("lcc.workflow_parallel._validate_file_params")
    mock_ncpus = mocker.patch("lcc.workflow_parallel._validate_ncpus")
    mock_ncpus.return_value = 0  # so only a single core

    lcc.workflow_parallel.tokenize_sentence_parallel(
        str(sample_medusa_file),
        str(fn_output_par),
        tokenizer=mock_sentence_tokenizer,
    )

    # call was redirected to sequential method
    mock_seq.assert_called_once_with(
        fn_input=str(sample_medusa_file),
        fn_output=str(fn_output_par),
        fmt_input=None,
        fmt_output=None,
        tokenizer=mock_sentence_tokenizer,
        dn_tokenizer_resources=None,
    )
    # and no further checks performed (that are done in the sequential one)
    mock_vfp_par.assert_not_called()
    mock_ncpus.assert_called_once_with(None)


# ---------------------------------------------------------------------------
