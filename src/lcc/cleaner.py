import logging
import os.path
from abc import ABCMeta
from abc import abstractmethod
from binascii import hexlify
from typing import Dict
from typing import List
from typing import Optional

try:
    import regex as re
except ImportError:
    import re  # type: ignore[no-redef]

# ---------------------------------------------------------------------------

# src/main/java/de/uni_leipzig/asv/tools/sentencecleaner/replacements/StringReplacements.java
# src/main/java/de/uni_leipzig/asv/tools/sentencecleaner/RuleFileParser.java
# src/main/java/de/uni_leipzig/asv/tools/sentencecleaner/SentenceCleaner.java
# src/main/java/de/uni_leipzig/asv/tools/sentencecleaner/SentenceFilter.java
# src/main/java/de/uni_leipzig/asv/tools/sentencecleaner/SimpleSentenceFilter.java

# ---------------------------------------------------------------------------


LOGGER = logging.getLogger(__name__)

ENCODING = "utf-8"

DISABLED_REGEX = "$UNMATCHABLE"


# ---------------------------------------------------------------------------


class SentenceFilter(metaclass=ABCMeta):
    """Interface for a single sentence filter.

    author: Thomas Eckart"""

    #: ID of this filter
    id_: int = -1
    #: description string of this filter
    description: Optional[str] = None
    #: number of "hits" (=removed sentences by this filter)
    hits: int = 0

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def is_valid(self, sentence: str) -> bool:
        """Checks sentence if it is valid regarding this filter"""


class DropExistingSentenceFilter(SentenceFilter):
    """No-op sentence filter. Used to mark removal of existing filter with same `id_`."""

    def __init__(self, id_: int, description: Optional[str] = None):
        self.id_ = id_
        self.description = description

    def init(self):
        pass

    def is_valid(self, sentence: str) -> bool:
        return True


class SimpleSentenceFilter(SentenceFilter):
    """Simple sentence filter that looks for regexp patterns or length restrictions."""

    pattern_str: Optional[str] = None
    pattern: Optional[re.Pattern] = None
    pattern_hex_str: Optional[str] = None
    pattern_hex: Optional[re.Pattern] = None

    min_length: int = 0
    max_length: int = 0
    replace_character_str: Optional[str] = None
    replace_ratio: float = 0.0
    replace_count: int = 0

    def init(self):
        if self.pattern_str:
            self.pattern = re.compile(self.pattern_str)
        if self.pattern_hex_str:
            self.pattern_hex = re.compile(self.pattern_hex_str)

    def is_filter_valid(self) -> bool:
        if self.id_ == -1:
            return False

        # warnings if nonsensical filters or invalid condition combinations
        if (
            self.replace_character_str is not None
            and self.replace_ratio != 0.0
            and self.replace_count != 0
        ):
            LOGGER.warning(
                "Both REPLACE_RATIO and REPLACE_COUNT are set. Check will only consider REPLACE_RATIO!"
            )
        if (
            self.replace_character_str is not None
            and self.replace_ratio == 0.0
            and self.replace_count == 0
        ):
            LOGGER.warning(
                "Both REPLACE_RATIO and REPLACE_COUNT are not set. REPLACE_CHARS will be skipped!"
            )
        if self.replace_character_str is None and (
            self.replace_ratio != 0.0 or self.replace_count != 0
        ):
            LOGGER.warning(
                "REPLACE_CHARS is not set. REPLACE_RATIO and/or REPLACE_COUNT will be skipped!"
            )

        if (
            self.pattern_str is None
            and self.pattern_hex_str is None
            and self.min_length == 0
            and self.max_length == 0
            and self.replace_character_str is None
            and self.replace_ratio == 0.0
            and self.replace_count == 0
        ):
            return False
        return True

    def is_valid(self, sentence: str) -> bool:
        """Check sentence validity against each condition. All conditions
        must apply for a sentence to be considered invalid.

        That means, check for each enabled filter condition if it matches
        then the sentence may not be valid anymore. Aggregate all conditions
        and if all express failure (not valid) then the sentence is not valid.
        That means if any conditions does not match then the sentence can be
        considered valid."""
        valid = True

        # REGEXP
        if self.pattern:
            if self.pattern.search(sentence):
                valid = False
            else:
                return True

        # MINLENGTH
        if self.min_length != 0:
            if len(sentence) < self.min_length:
                valid = False
            else:
                return True

        # MAXLENGTH
        if self.max_length != 0:
            if len(sentence) > self.max_length:
                valid = False
            else:
                return True

        # REPLACE_CHARS + REPLACE_RATIO
        if self.replace_character_str is not None and (
            self.replace_ratio != 0.0 or self.replace_count != 0.0
        ):
            sentence_tmp = sentence
            for char in self.replace_character_str:
                sentence_tmp = sentence_tmp.replace(char, "")

            if self.replace_ratio != 0.0:
                ratio = len(sentence) / (len(sentence_tmp) + 1)
                if ratio > self.replace_ratio:
                    valid = False
                else:
                    return True
            elif self.replace_count:
                count = len(sentence) - len(sentence_tmp)
                if count >= self.replace_count:
                    valid = False
                else:
                    return True

        # HEX_REGEXP
        if self.pattern_hex:
            hex_failed = False
            for char in sentence:
                try:
                    char_hex = hexlify(char.encode(ENCODING)).decode(ENCODING)
                    if self.pattern_hex.search(char_hex):
                        hex_failed = True
                        break
                except Exception as ex:
                    LOGGER.debug(
                        "Error checking hex de-/encoded characters: '%s', %s", char, ex
                    )
            if hex_failed:
                valid = False
            else:
                return True

        if not valid:
            self.hits += 1

        return valid


# ---------------------------------------------------------------------------


class StringReplacements:
    fn_replacements: str
    replacements: Dict[str, str] = dict()

    def __init__(self, fn_replacements: str) -> None:
        self.fn_replacements = fn_replacements
        self.load_replacements(self.fn_replacements)

    def load_replacements(self, fn_replacements: str, encoding: str = ENCODING):
        try:
            with open(fn_replacements, "r", encoding=encoding) as fp:
                for line in fp:
                    line = line.rstrip()
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) != 2:
                        continue
                    self.replacements[parts[0]] = parts[1]

        except FileNotFoundError:
            LOGGER.warning("String replacements file '%s' not found!", fn_replacements)

    def replace(self, string: str) -> str:
        for r_from, r_to in self.replacements.items():
            string = string.replace(r_from, r_to)
        return string


# ---------------------------------------------------------------------------


def load_from_rulesfile(
    fn_rules: str, name: str, encoding: str = ENCODING
) -> List[SentenceFilter]:
    LOGGER.debug("Try to parse rules from file '%s'", fn_rules)
    filters: List[SentenceFilter] = list()
    try:
        with open(fn_rules, "r", encoding=encoding) as fp:
            filter_: SimpleSentenceFilter = SimpleSentenceFilter()
            for lno, line in enumerate(fp):
                if line.startswith("#") or line.startswith("//") or not line.strip():
                    continue

                line = line.rstrip("\r\n")
                line_upper = line.upper()
                if line_upper.startswith("RULE"):
                    if filter_.is_filter_valid():
                        filter_.init()
                        LOGGER.debug(
                            "   ...added rule: [%s] %s",
                            filter_.id_,
                            filter_.description or "nothing",
                        )
                        if filter_.pattern_str == DISABLED_REGEX:
                            filters.append(
                                DropExistingSentenceFilter(
                                    filter_.id_, filter_.description
                                )
                            )
                        else:
                            filters.append(filter_)

                    filter_ = SimpleSentenceFilter()
                    filter_.id_ = int(line[5:])

                elif line_upper.startswith("DESC:"):
                    filter_.description = line[5:]
                    if name:
                        filter_.description = f"{name} - {filter_.description}"

                elif line_upper.startswith("REGEXP:"):
                    filter_.pattern_str = line[7:]

                elif line_upper.startswith("MAXLENGTH:"):
                    filter_.max_length = int(line[10:])

                elif line_upper.startswith("MINLENGTH:"):
                    filter_.min_length = int(line[10:])

                # NOTE: may end with ":" or "="
                elif line_upper.startswith("REPLACE_CHARS"):
                    filter_.replace_character_str = line[14:]

                # NOTE: may end with ":" or "="
                elif line_upper.startswith("REPLACE_RATIO"):
                    filter_.replace_ratio = float(line[14:])

                # NOTE: may end with ":" or "="
                elif line_upper.startswith("REPLACE_COUNT"):
                    filter_.replace_count = int(line[14:])

                elif line_upper.startswith("HEX_REGEXP:"):
                    filter_.pattern_hex_str = line[11:]

                # XXX: add a disabled keyword
                elif line_upper.startswith("DISABLE"):
                    filter_.pattern_str = DISABLED_REGEX

                else:
                    LOGGER.warning(
                        "Ignore unknown rule line: '%s' in line %s of %s",
                        line,
                        lno,
                        fn_rules,
                    )

        # last rule
        if filter_.is_filter_valid():
            filter_.init()
            LOGGER.debug(
                "   ...added rule: [%s] %s",
                filter_.id_,
                filter_.description or "nothing",
            )
            if filter_.pattern_str == DISABLED_REGEX:
                filters.append(
                    DropExistingSentenceFilter(filter_.id_, filter_.description)
                )
            else:
                filters.append(filter_)

    except FileNotFoundError:
        LOGGER.warning("Sentence cleaner rules file '%s' not found!", fn_rules)

    return filters


# ---------------------------------------------------------------------------


class SentenceCleaner:
    filters: List[SentenceFilter]
    replacer: Optional[StringReplacements] = None

    def __init__(
        self,
        dn_rules: str,
        text_type: Optional[str] = None,
        lang_code: Optional[str] = None,
        fn_replacements: Optional[str] = "StringReplacements.list",
    ) -> None:
        self.load_rules(dn_rules, text_type=text_type, lang_code=lang_code)
        if fn_replacements:
            self.replacer = StringReplacements(os.path.join(dn_rules, fn_replacements))

    def filter_sentence(
        self, sentence: str, do_replacements: bool = True
    ) -> Optional[str]:
        """Filters a single sentence by checking it against SentenceFilters."""
        # String replacements
        if do_replacements and self.replacer:
            sentence = self.replacer.replace(sentence)

        # sequential filter checks
        for filter_ in self.filters:
            if not filter_.is_valid(sentence):
                LOGGER.debug(
                    'Sentence "%s" failed test: %s - %s',
                    sentence,
                    filter_.id_,
                    filter_.description or "nothing",
                )
                return None

        return sentence

    def load_rules(
        self,
        dn_rules: str,
        text_type: Optional[str] = None,
        lang_code: Optional[str] = None,
    ) -> bool:
        self.filters = list()

        # mapping of filename base to category name
        fns_rules = {"general": "General"}
        if text_type:
            fns_rules[f"texttype_{text_type}"] = text_type
        if lang_code:
            fns_rules[f"lang_{lang_code}"] = lang_code

        for fn_rules, name in fns_rules.items():
            fn_rules = os.path.join(dn_rules, f"{fn_rules}.rules")
            if not os.path.exists(fn_rules):
                LOGGER.debug("Rules file '%s' not found.", fn_rules)
                continue

            filter_tmp = load_from_rulesfile(fn_rules, name)
            LOGGER.debug(
                "Loaded %s filter rules for %s from %s", len(filter_tmp), name, fn_rules
            )

            # remove disabled filters
            drop_filter_ids = {
                filter_.id_
                for filter_ in filter_tmp
                if isinstance(filter_, DropExistingSentenceFilter)
            }
            if drop_filter_ids:
                # remove disabled filter (marker) since we do not want to add it to the final list
                filter_tmp = [
                    filter_
                    for filter_ in filter_tmp
                    if not isinstance(filter_, DropExistingSentenceFilter)
                ]
                # find to-be-disabled filters (if they exist)
                dropped_filters = [
                    filter_
                    for filter_ in self.filters
                    if filter_.id_ in drop_filter_ids
                ]
                if dropped_filters:
                    # remove disabled filter from existing filter list
                    self.filters = [
                        filter_
                        for filter_ in self.filters
                        if filter_.id_ not in drop_filter_ids
                    ]
                    if LOGGER.isEnabledFor(logging.DEBUG):
                        LOGGER.debug("Disabled the following filters:")
                        for filter_ in dropped_filters:
                            LOGGER.debug(
                                "   ...removed rule: [%s] %s",
                                filter_.id_,
                                filter_.description or "nothing",
                            )

            # check overridden filters
            new_filter_ids = {filter_.id_ for filter_ in filter_tmp}
            if new_filter_ids:
                filters_to_be_overriden = [
                    filter_ for filter_ in self.filters if filter_.id_ in new_filter_ids
                ]
                if filters_to_be_overriden:
                    # remove overridden filter from existing filter list
                    self.filters = [
                        filter_
                        for filter_ in self.filters
                        if filter_.id_ not in new_filter_ids
                    ]
                    if LOGGER.isEnabledFor(logging.DEBUG):
                        LOGGER.debug("Overrides the following filters:")
                        for filter_ in filters_to_be_overriden:
                            LOGGER.debug(
                                "   ...overridden rule: [%s] %s",
                                filter_.id_,
                                filter_.description or "nothing",
                            )

            # merge filter lists
            self.filters.extend(filter_tmp)

        return bool(self.filters)


# ---------------------------------------------------------------------------
