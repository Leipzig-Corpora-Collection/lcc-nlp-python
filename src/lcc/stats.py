import hashlib
import json
import logging
from dataclasses import asdict
from dataclasses import dataclass
from typing import Iterable
from typing import Optional
from typing import Union

import lcc.io

# ---------------------------------------------------------------------------


LOGGER = logging.getLogger(__name__)

ENCODING = "utf-8"


# ---------------------------------------------------------------------------
# source


@dataclass(frozen=True)
class DocStats:
    docs: int
    lines: int
    tokens: int
    chars: int
    source_dates: Optional[int] = None
    sources: Optional[int] = None
    dates: Optional[int] = None

    def to_tsv(self) -> str:
        return (
            f"{self.source_dates if self.source_dates else ''}\t"
            f"{self.sources if self.sources else ''}\t"
            f"{self.dates if self.dates else ''}\t"
            f"{self.docs}\t{self.lines}\t{self.tokens}\t{self.chars}"
        )

    def to_json(self, pretty: bool = False) -> str:
        return json.dumps(asdict(self), indent=2 if pretty else None)


def compute_docs_stats_heuristic(
    docs: Iterable[lcc.io.DocAndMeta],
    hasher_fn=hashlib.sha256,
    encoding: str = ENCODING,
) -> DocStats:
    num_docs = num_lines = num_tokens = num_chars = 0
    sources = set()
    dates = set()
    source_dates = set()
    for doc in docs:
        num_docs += 1
        if doc.content:
            num_lines += len(doc.content.splitlines())
            # sum(map(lambda x: len(x.split()), lines))
            num_tokens += len(doc.content.split())
            # sum(map(len, lines))
            num_chars += len(
                doc.content.decode(encoding)
                if isinstance(doc.content, (bytes, bytearray))
                else doc.content
            )

        if hasher_fn:
            m = hasher_fn((doc.meta.location or "").encode(encoding))
            sources.add(m.digest())
            m.update((doc.meta.date or "").encode(encoding))
            source_dates.add(m.digest())
            dates.add(doc.meta.date)

    return DocStats(
        lines=num_lines,
        tokens=num_tokens,
        chars=num_chars,
        docs=num_docs,
        sources=len(sources) if hasher_fn else None,
        dates=len(dates) if hasher_fn else None,
        source_dates=len(source_dates) if hasher_fn else None,
    )


# ---------------------------------------------------------------------------
# medusa


@dataclass(frozen=True)
class MedusaStats:
    lines: int
    tokens: int
    chars: int
    sources: Optional[int] = None

    def to_tsv(self) -> str:
        return f"{self.sources if self.sources is not None else ''}\t{self.lines}\t{self.tokens}\t{self.chars}"

    def to_json(self, pretty: bool = False) -> str:
        return json.dumps(asdict(self), indent=2 if pretty else None)


def compute_sentences_stats_heuristic(
    sent_iter: Union[
        Iterable[lcc.io.SentenceAndMeta], Iterable[lcc.io.SentencesAndMeta]
    ],
    count_sources: bool = True,
    hasher_fn=hashlib.sha256,
    encoding: str = ENCODING,
) -> MedusaStats:
    num_lines = num_tokens = num_chars = 0
    sources = set()
    for sent in sent_iter:
        if isinstance(sent, lcc.io.SentenceAndMeta):
            num_lines += 1
            num_tokens += len(sent.sentence.split())
            num_chars += len(sent.sentence)
        elif isinstance(sent, lcc.io.SentencesAndMeta):
            num_lines += len(sent.sentences)
            num_tokens += sum(len(sentence.split()) for sentence in sent.sentences)
            num_chars += sum(map(len, sent.sentences))
        else:
            LOGGER.warning(
                "Unsupported 'sent_iter' item of type %s. Skipped.", type(sent)
            )
            continue

        if count_sources:
            m = hasher_fn((sent.meta.location or "").encode(encoding))
            m.update((sent.meta.date or "").encode(encoding))
            sources.add(m.digest())

    return MedusaStats(
        lines=num_lines,
        tokens=num_tokens,
        chars=num_chars,
        sources=len(sources) if count_sources else None,
    )


# ---------------------------------------------------------------------------
