try:
    import spacy  # noqa: F401
except ImportError:
    raise RuntimeError(
        "SpaCy integration requires the installation of the optional dependency 'spacy'!"
    )

from typing import List
from typing import Optional

from spacy.language import Language
from spacy.tokens import Doc
from spacy.tokens import Span

from lcc.language.sentence import LanIKernel
from lcc.language.sentence import get_languages_to_check_for

# ---------------------------------------------------------------------------


@Language.factory(
    "lani",
    default_config={
        "dn_wordlists": "resources/jlani/wordlists/plain",
        "fn_filterlist": "resources/jlani/blacklist_utf8.txt",
        "languages": [],
        "expand_for_similar": False,
        "fn_lang_dist": "resources/jlani/wordlists/lang_dist.tsv",
    },
)
def create_lani_component(
    nlp: Language,
    name: str,
    dn_wordlists: str,
    fn_filterlist: Optional[str] = None,
    languages: Optional[List[str]] = None,
    expand_for_similar: bool = False,
    fn_lang_dist: Optional[str] = None,
):
    return LaniComponent(
        nlp,
        dn_wordlists=dn_wordlists,
        fn_filterlist=fn_filterlist,
        languages=languages,
        expand_for_similar=expand_for_similar,
        fn_lang_dist=fn_lang_dist,
    )


class LaniComponent:
    def __init__(
        self,
        nlp: Language,
        dn_wordlists: str,
        fn_filterlist: Optional[str] = None,
        languages: Optional[List[str]] = None,
        expand_for_similar: Optional[bool] = False,
        fn_lang_dist: Optional[str] = None,
    ):
        if languages and not isinstance(languages, (tuple, list, set)):
            raise TypeError("Parameter 'languages' must be of list/tuple/set type!")
        if expand_for_similar:
            if not languages:
                raise ValueError(
                    "Parameter 'expand_for_similar' must not be True if no 'languages' are provided!"
                )
            if not fn_lang_dist:
                raise ValueError(
                    "Parameter 'fn_lang_dist' is required if for similar languages should be checked!"
                )

        self._dn_wordlists = dn_wordlists
        self._fn_filterlist = fn_filterlist
        self._languages = set(languages) if languages else None
        self._expand_for_similar = expand_for_similar

        self.lcc_lani = LanIKernel(
            dn_wordlists=self._dn_wordlists, fn_filterlist=self._fn_filterlist
        )

        self._languages_to_check = set(self._languages) if self._languages else None
        if self._expand_for_similar and self._languages_to_check:
            for language in self._languages:
                new_languages = get_languages_to_check_for(language, fn_lang_dist)
                self._languages_to_check.update(new_languages)

        if not Doc.has_extension("language"):
            Doc.set_extension("language", default=None)
        # if not Span.has_extension("language"):
        #     Span.set_extension("language", default=None)

    def __call__(self, doc: Doc) -> Doc:
        result = self.lcc_lani.evaluate(doc.text, languages=self._languages_to_check)
        doc._.set("language", None)
        if not result:
            return doc

        lang_result = result.get_result()
        if not lang_result.is_known():
            return doc

        doc._.set("language", (lang_result.language, lang_result.probability / 100))

        return doc


# ---------------------------------------------------------------------------
