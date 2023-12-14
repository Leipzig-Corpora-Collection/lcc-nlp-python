try:
    import spacy
except ImportError:
    raise RuntimeError(
        "SpaCy integration requires the installation of the optional dependency 'spacy'!"
    )

from dataclasses import dataclass
from typing import Optional

from spacy.language import Language
from spacy.tokens import Doc
from spacy.tokens import Span

from lcc.cleaner import SentenceCleaner

# ---------------------------------------------------------------------------


@Language.factory(
    "sentencecleaner",
    default_config={
        "dn_rules": "resources/cleaner",
        "text_type": None,
        "lang_code": None,
        "show_reason": False,
    },
)
def create_sentencecleaner_component(
    nlp: Language,
    name: str,
    dn_rules: str,
    text_type: Optional[str] = None,
    lang_code: Optional[str] = None,
    show_reason: bool = False,
):
    return SentenceCleanerComponent(
        nlp,
        dn_rules=dn_rules,
        text_type=text_type,
        lang_code=lang_code,
        show_reason=show_reason,
    )


@dataclass
class FilterResult:
    id: int
    description: str
    filtered: bool


class SentenceCleanerComponent:
    def __init__(
        self,
        nlp: Language,
        dn_rules: str,
        text_type: Optional[str] = None,
        lang_code: Optional[str] = None,
        show_reason: Optional[bool] = False,
    ):
        self._dn_rules = dn_rules
        self._show_reason = show_reason

        self.lcc_cleaner = SentenceCleaner(
            dn_rules, text_type=text_type, lang_code=lang_code
        )

        if not Doc.has_extension("filtered"):
            Doc.set_extension("filtered", default=False)
        # if not Span.has_extension("filtered"):
        #     Span.set_extension("filtered", default=False)
        if show_reason:
            if not Doc.has_extension("filter_reasons"):
                Doc.set_extension("filter_reasons", default=None)
            # if not Span.has_extension("filter_reasons"):
            #     Span.set_extension("filter_reasons", default=None)

    def __call__(self, doc: Doc) -> Doc:
        if self._show_reason:
            doc_result = self.lcc_cleaner.filter_sentence_results(doc.text)
            filtered = any(doc_result.values())
            doc._.set("filtered", filtered)
            doc._.set(
                "filter_reasons",
                [
                    FilterResult(id=k.id_, description=k.description, filtered=True)
                    for k, v in doc_result.items()
                    if v
                ],
            )
        else:
            filtered = self.lcc_cleaner.filter_sentence(doc.text) is None
            doc._.set("filtered", filtered)

        return doc


# ---------------------------------------------------------------------------
