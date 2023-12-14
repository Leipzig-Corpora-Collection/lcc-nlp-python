import logging
import os.path
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

try:
    import regex as re
except ImportError:
    import re  # type: ignore[no-redef]

# ---------------------------------------------------------------------------

# src/main/java/de/uni_leipzig/asv/tools/jLanI/kernel/DataSource.java
# src/main/java/de/uni_leipzig/asv/tools/jLanI/kernel/DatasourceManager.java
# src/main/java/de/uni_leipzig/asv/tools/jLanI/kernel/LangResult.java
# src/main/java/de/uni_leipzig/asv/tools/jLanI/kernel/LanIKernel.java
# src/main/java/de/uni_leipzig/asv/tools/jLanI/kernel/Response.java

# ---------------------------------------------------------------------------


LOGGER = logging.getLogger(__name__)

ENCODING = "utf-8"
LANG_UNKNOWN = "unknown"

FN_PROPERTY = "lanikernel.ini"
EXT_WORDS = ".words"

LANGUAGE_MIN_SIMILARITY_THRESHOLD = 0.8
LANGUAGES_ALWAYS = ["ara", "eng", "deu", "fra", "spa", "por", "rus"]


# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LanguageResult:
    MINWORDCOUNT: int
    MINCOVERAGE: float

    language: str
    language_nextbest: str
    probability: int
    probability_nextbest: int
    wordcount: int
    coverage: float

    def __post_init__(self):
        if not self.is_known():
            object.__setattr__(self, "language", LANG_UNKNOWN)
            object.__setattr__(self, "language_nextbest", LANG_UNKNOWN)
            object.__setattr__(self, "probability", 0)
            object.__setattr__(self, "probability_nextbest", 0)

    def is_known(self) -> bool:
        if self.probability < 2 * self.probability_nextbest:
            return False

        # XXX: changes here to account for reduced evaluation mode
        if self.wordcount != 0 and self.wordcount < self.MINWORDCOUNT:
            return False
        if self.coverage != -1.0 and self.coverage < self.MINCOVERAGE:
            return False

        return True

    def to_str(self) -> str:
        if not self.is_known():
            return LANG_UNKNOWN
        return f"{self.language}\t{self.probability}\t{self.language_nextbest}\t{self.probability_nextbest}"

    def to_textout(self) -> str:
        if not self.is_known():
            return LANG_UNKNOWN
        return f"{self.language}:{self.probability} {self.language_nextbest}:{self.probability_nextbest}"


@dataclass
class EvaluationResult:
    sentence_length: int = -1
    result: Dict[str, float] = field(default_factory=dict)
    seen_words: Dict[str, List[str]] = field(default_factory=dict)
    coverage: Dict[str, float] = field(default_factory=dict)
    wordcount: Dict[str, int] = field(default_factory=dict)

    def get_language_coverage(self, language: str) -> float:
        if not self.coverage:
            LOGGER.error("Reduced result, no coverage information available!")
            return -1.0

        try:
            return self.coverage[language]
        except KeyError:
            LOGGER.error("Coverage data for language '%s' not found?!", language)
            return -1.0

    def get_language_wordcount(self, language: str) -> int:
        if not self.wordcount:
            LOGGER.error("Reduced result, no word count information available!")
            return 0

        try:
            return self.wordcount[language]
        except KeyError:
            LOGGER.error("Word count data for language '%s' not found?!", language)
            return 0

    def get_result(self, mincov: float = 0.15, mincount: int = 2) -> LanguageResult:
        # NOTE: assume that at least two languages were used at this point

        prob_sum: float = sum(self.result.values())
        top_two_results = sorted(self.result.items(), key=lambda x: x[1], reverse=True)[
            :2
        ]

        # TODO: faster sort for only top two
        # - get first two elements, switch so that first is highest
        # - go over all the remaining elements (3+)
        #   - for each check if higher than second (highest) element
        #     - if so, check if higher than first (highest) element
        #       - if so, then set second with first and update first with current
        #       - otherwise only update second (highest) with current element

        language = top_two_results[0][0]
        return LanguageResult(
            language=language,
            probability=round(top_two_results[0][1] / prob_sum * 100),
            language_nextbest=top_two_results[1][0],
            probability_nextbest=round(top_two_results[1][1] / prob_sum * 100),
            coverage=self.get_language_coverage(language),
            wordcount=self.get_language_wordcount(language),
            MINCOVERAGE=mincov,
            MINWORDCOUNT=mincount,
        )


@dataclass(frozen=True)
class DataSource:
    name: str
    wordlist: Dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.wordlist)

    def __getitem__(self, word: str) -> Optional[float]:
        return self.wordlist.get(word, None)

    def __contains__(self, word: str) -> bool:
        return word in self.wordlist


class DataSourceManager:
    datasources: Dict[str, DataSource]
    REST: float

    def __init__(self) -> None:
        self.datasources = dict()
        self.REST = 0.1

    @property
    def languages(self) -> Set[str]:
        return set(self.datasources.keys())

    def add_datasource(self, name: str, ds: DataSource) -> bool:
        if name in self.datasources:
            LOGGER.warning(
                "DataSourceManager already contains language '%s'! New DataSource will be ignored.",
                name,
            )
            return False

        self.datasources[name] = ds

        return True

    def add_datasource_from_file(self, name: str, fn_wordlist: str) -> bool:
        if not fn_wordlist:
            raise TypeError("Parameter 'fn_wordlist' must not be None!")
        if not os.path.exists(fn_wordlist) or not os.path.isfile(fn_wordlist):
            raise FileNotFoundError("File 'fn_wordlist' does not exist!")

        wordlist = self._parse_wordlist_file(fn_wordlist)
        ds = DataSource(name, wordlist)

        return self.add_datasource(name, ds)

    def _parse_wordlist_file(self, fn_wordlist: str, sep="\t") -> Dict[str, float]:
        wordmap: Dict[str, float] = dict()

        LOGGER.debug("Parse wordlist from '%s' ...", fn_wordlist)
        with open(fn_wordlist, "r", encoding=ENCODING) as fp:
            try:
                line = fp.readline().strip()
                token = int(line)
            except ValueError as ex:
                raise Exception("Invalid fileformat!") from ex

            if token <= 0:
                raise Exception(
                    f"Value of 'token' (size of reference corpus) cannot be {token}!"
                )
            LOGGER.debug("token (from %s): %s", fn_wordlist, token)

            for lno, line in enumerate(fp, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split(sep)
                if len(parts) != 2:
                    LOGGER.error(
                        "Line '%s' (at line %s) doesn't match required format '<int><sep><word>' (sep=%r)",
                        line,
                        lno,
                        sep,
                    )
                    # raise Exception(f"Wrong format in fn_wordlist in line '{line}'!")
                    continue

                wordcount = int(parts[0])
                word = parts[1]

                if word in wordmap:
                    LOGGER.warning(
                        "Duplicate entry in wordlist: '%s' (at line %s)! Keeping the first entry only.",
                        word,
                        lno,
                    )
                    continue

                wordmap[word] = wordcount / token

        return wordmap

    def evaluate(
        self, languages: Iterable[str], word: str, reduced: bool = False
    ) -> Tuple[Dict[str, Optional[float]], Optional[List[str]]]:
        languages = set(languages)
        if not languages.issubset(self.datasources.keys()):
            LOGGER.error(
                "Given languages '%s' is not a subset of '%s'!",
                languages,
                set(self.datasources.keys()),
            )
            raise ValueError(
                "One or some languages are not known by this DataSourceManager!"
            )

        failed: int = 0
        probSum: float = 0.0
        result: Dict[str, Optional[float]] = dict()
        found_languages: List[str] = list()

        for language in languages:
            prob = self.datasources[language][word]

            if prob is None:
                failed += 1
                result[language] = None
            else:
                probSum += prob
                result[language] = prob
                if not reduced:
                    found_languages.append(language)

        if failed == len(languages):
            rest = 1.0
        else:
            rest = self.REST / len(languages) * probSum

        for language in result:
            if result[language] is None:
                result[language] = rest
                probSum += rest

        for language in result:
            result[language] *= len(languages) / probSum

        return result, found_languages if not reduced else None

    def __contains__(self, language: str) -> bool:
        return language in self.datasources

    def __str__(self) -> str:
        return f"DataSourceManager contains languages {sorted(self.datasources.keys())} with {sum(map(len, self.datasources.values()))} words."


class LanIKernel:
    datasourcemngr: DataSourceManager
    filterlist: Set[str]
    specialChars: Optional[str] = None
    specialCharsPattern: Optional[re.Pattern] = None

    def __init__(
        self,
        dn_wordlists: str,
        specialChars: Optional[str] = None,
        fn_filterlist: Optional[str] = None,
    ) -> None:
        self._setup_languages(dn_wordlists)
        self._setup_special_chars(specialChars)
        self._setup_filterlist(fn_filterlist)

    @classmethod
    def from_prefs(cls, fn_prefs: str) -> "LanIKernel":
        if not fn_prefs:
            raise TypeError("Parameter 'fn_prefs' must not be None!")
        # allow for both directory (with default config name) or direct filepath
        if os.path.isdir(fn_prefs):
            fn_prefs = os.path.join(fn_prefs, FN_PROPERTY)
        prefs = cls._load_prefs(fn_prefs)

        # get base directory of config file for later relative paths
        dn_prefs = os.path.dirname(fn_prefs)

        dn_wordlists = prefs.get("WordlistDir", None)
        if dn_wordlists is None or not dn_wordlists.strip():
            raise ValueError("Missing required 'WordlistDir' configuration!")
        if not os.path.isabs(dn_wordlists):
            dn_wordlists = os.path.normpath(os.path.join(dn_prefs, dn_wordlists))

        specialChars = prefs.get("SpecialChars", None)

        fn_filterlist = prefs.get("BlacklistFile", None)
        if fn_filterlist and not os.path.isabs(fn_filterlist):
            fn_filterlist = os.path.normpath(os.path.join(dn_prefs, fn_filterlist))

        return cls(
            dn_wordlists=dn_wordlists,
            specialChars=specialChars,
            fn_filterlist=fn_filterlist,
        )

    @staticmethod
    def _load_prefs(fn_prefs):
        import configparser

        if not os.path.exists(fn_prefs):
            raise FileNotFoundError(f"Missing configuration file at '{fn_prefs}'!")

        config = configparser.ConfigParser(interpolation=None)

        with open(fn_prefs, "r", encoding=ENCODING) as fp:

            def add_section_header(fp, header_name):
                yield f"[{header_name}]\n"
                yield from fp

            config.read_file(add_section_header(fp, "dummy"), source=fn_prefs)

        return config["dummy"]

    def _setup_languages(self, dn_wordlists: str):
        self.datasourcemngr = DataSourceManager()

        if not dn_wordlists:
            raise ValueError("Invalid value of 'dn_wordlists'. Must not be None.")
        if not os.path.exists(dn_wordlists) or not os.path.isdir(dn_wordlists):
            raise FileNotFoundError(f"Directory '{dn_wordlists}' does not exist!")

        wl_files = os.listdir(dn_wordlists)
        if not wl_files:
            raise FileNotFoundError(f"No wordlist files found in '{dn_wordlists}'!")

        for wl_file in wl_files:
            if wl_file.endswith(".ser.gz"):
                raise RuntimeError(
                    f"Unsupported serialized worlist models! ({os.path.join(dn_wordlists, wl_file)})"
                )

            if wl_file.endswith(EXT_WORDS) and len(wl_file) > len(EXT_WORDS):
                LOGGER.debug(
                    "Found wordlist file: %s", os.path.join(dn_wordlists, wl_file)
                )

                language = wl_file[: -len(EXT_WORDS)]
                self.datasourcemngr.add_datasource_from_file(
                    language, os.path.join(dn_wordlists, wl_file)
                )

        LOGGER.debug("%s", self.datasourcemngr)

    def _setup_special_chars(self, specialChars: Optional[str] = None):
        self.specialChars = specialChars
        if not self.specialChars:
            LOGGER.debug("No special chars to remove found.")
            self.specialCharsPattern = None
            return

        self.specialCharsPattern = re.compile(self.specialChars)

    def _setup_filterlist(self, fn_filterlist: Optional[str] = None):
        self.filterlist = set()

        if fn_filterlist is None:
            LOGGER.debug("No filterlist file specified, that's not a real problem.")
            return

        if not os.path.exists(fn_filterlist):
            raise FileNotFoundError(f"Filterlist file '{fn_filterlist}' not found!")

        with open(fn_filterlist, "r", encoding=ENCODING) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                self.filterlist.add(line)

        LOGGER.debug("Read %s entries from filterlist file.", len(self.filterlist))

    def _clean_sentence(self, sentence: str) -> str:
        if not self.specialChars or not self.specialCharsPattern:
            return sentence

        return self.specialCharsPattern.sub("", sentence)

    def evaluate(
        self,
        sentence: str,
        languages: Optional[Iterable[str]] = None,
        reduced: bool = False,
        num_words_to_check: int = 0,
    ) -> Optional[EvaluationResult]:
        if not sentence:
            LOGGER.error("Parameter 'sentence' shouldn't be None!")
            return None

        if not languages:
            LOGGER.debug(
                "No 'languages' specified. Checking sentence against all languages available."
            )
            languages = self.datasourcemngr.languages
            LOGGER.debug("Available languages are: %s", languages)

        temp_result: Dict[str, float] = dict()
        seen_words: Dict[str, List[str]] = dict()
        for language in languages:
            temp_result[language] = 1.0
            if not reduced:
                seen_words[language] = list()

        sentence = self._clean_sentence(sentence)
        tokens = sentence.split(" ")

        # sample
        if num_words_to_check > 0:
            if len(tokens) > num_words_to_check:
                skip = len(tokens) // num_words_to_check
                tokens = [token for idx, token in enumerate(tokens) if idx % skip == 0]
                tokens = tokens[:num_words_to_check]

        n_tokens = 0  # sentenceLength
        for token in tokens:
            if token in self.filterlist:
                continue
            n_tokens += 1

            # TODO: maybe parallelize here?
            token_result, token_languages = self.datasourcemngr.evaluate(
                languages, token, reduced
            )
            if not reduced:
                for language in token_languages:
                    seen_words[language].append(token)

            for language in languages:
                temp_result[language] *= token_result[language]

        result: EvaluationResult = EvaluationResult(
            sentence_length=n_tokens, result=temp_result, seen_words=seen_words
        )

        if not reduced:
            coverage: Dict[str, float] = dict()
            wordcount: Dict[str, int] = dict()

            for language, tokens in seen_words.items():
                coverage[language] = len(tokens) / n_tokens if n_tokens > 0 else 0.0
                wordcount[language] = len(tokens)

            result.coverage = coverage
            result.wordcount = wordcount

        return result


# ---------------------------------------------------------------------------


def get_languages_similar_to(
    language: str,
    fn_lang_dist: str,
    min_similarity: float = LANGUAGE_MIN_SIMILARITY_THRESHOLD,
) -> Set[str]:
    results: Set[str] = set()
    with open(fn_lang_dist, "r", encoding=ENCODING) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 3:
                continue

            lang_1, lang_2, dist = parts[0], parts[1], float(parts[2])
            if language.startswith(lang_1) and dist < min_similarity:
                results.add(lang_2)

    return results


def get_languages_to_check_for(
    language,
    fn_lang_dist: str,
    min_similarity: float = LANGUAGE_MIN_SIMILARITY_THRESHOLD,
) -> Set[str]:
    languages = get_languages_similar_to(
        language=language, fn_lang_dist=fn_lang_dist, min_similarity=min_similarity
    )
    return set(LANGUAGES_ALWAYS) | languages | {language}


# ---------------------------------------------------------------------------
