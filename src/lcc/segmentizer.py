import logging
import os.path
from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

try:
    import regex as re
except ImportError:
    import re  # type: ignore[no-redef]

# ---------------------------------------------------------------------------

# src/main/java/de/uni_leipzig/asv/tools/segmentizer/app/AsvSegmentizerImpl.java
# src/main/java/de/uni_leipzig/asv/tools/segmentizer/config/impl/SegmentizerConfigImpl.java
# src/main/java/de/uni_leipzig/asv/tools/segmentizer/file/FileReaderThread.java
# src/main/java/de/uni_leipzig/asv/tools/segmentizer/file/FileWriterThread.java
# src/main/java/de/uni_leipzig/asv/tools/segmentizer/processor/impl/util/BoundaryIndexAndLength.java
# src/main/java/de/uni_leipzig/asv/tools/segmentizer/processor/impl/util/BoundaryRuleHelper.java
# src/main/java/de/uni_leipzig/asv/tools/segmentizer/processor/impl/util/StringHash.java
# src/main/java/de/uni_leipzig/asv/tools/segmentizer/processor/impl/util/StringHelper.java
# src/main/java/de/uni_leipzig/asv/tools/segmentizer/processor/impl/PostBoundaryListProcessor.java
# src/main/java/de/uni_leipzig/asv/tools/segmentizer/processor/impl/PostBoundaryRulesProcessor.java
# src/main/java/de/uni_leipzig/asv/tools/segmentizer/processor/impl/PreBoundaryListProcessor.java
# src/main/java/de/uni_leipzig/asv/tools/segmentizer/processor/impl/PreBoundaryRulesProcessor.java
# src/main/java/de/uni_leipzig/asv/tools/segmentizer/processor/BoundaryProcessor.java

# Issues:
# regex stuff: \p{IsL}
# https://github.com/openjdk-mirror/jdk7u-jdk/blob/f4d80957e89a19a29bb9f9807d2a28351ed7f7df/src/share/classes/java/util/regex/Pattern.java#L5533
# https://docs.oracle.com/javase/8/docs/api/java/util/regex/Pattern.html

# ---------------------------------------------------------------------------


LOGGER = logging.getLogger(__name__)


POINT_CHAR = "."
ENCODING = "utf-8"
CR = "\n"  # XXX: it's a line feed to be exact
SPACE = " "
DOUBLE_SPACE = "  "


# ---------------------------------------------------------------------------


def str_hash(text: str) -> int:
    """Calculates the hash value for a given string.

    The algorithm uses every character up to strings with 15 chars. For
    larger strings the first 15 chars added by a maximum of 16 other chars
    equally distributed over the rest of the string are used. (algorithm was
    taken from JDK 1.1 but changed (T.Boehme))
    """
    #: hash value
    h = 0
    #: current character in string
    off = 0
    #: calculate hash for the first 15 characters
    tmpLen = min(len(text), 15)
    for _ in range(tmpLen, 0, -1):
        h = (h * 37) + ord(text[off])
        h = h & 0xFFFFFFFF
        h = (h ^ 0x80000000) - 0x80000000
        off += 1
    #: process the rest of the string
    if len(text) >= 16:
        #: only sample some characters from rest
        skip = len(text) // 16
        for _ in range(len(text), 15, -skip):
            h = (h * 39) + ord(text[off])
            h = h & 0xFFFFFFFF
            h = (h ^ 0x80000000) - 0x80000000
            off += skip
    return abs(h)


# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundaryIndexAndLength:
    index: int
    length: int


def load_patterns(fn_rules: str, encoding: str = "utf-8") -> Dict[re.Pattern, bool]:
    patterns: Dict[re.Pattern, bool] = dict()

    try:
        with open(fn_rules, "r", encoding=encoding) as fp:
            for line in fp:
                line = line.rstrip()
                if not line or line.startswith("#"):
                    continue
                if not line.startswith("- ") and not line.startswith("+ "):
                    LOGGER.warning(
                        "Unable to parse line '%s' in postBoundaryRules-file. Patterns need to specify a decision using '+ '/'- ' in front of the pattern.",
                        line,
                    )
                    continue
                try:
                    pattern = re.compile(line[2:])
                    is_decision = line.startswith("+ ")
                    patterns[pattern] = is_decision
                except re.error as ex:
                    LOGGER.error("Error compiling rule pattern for '%s': %s", line, ex)
    except FileNotFoundError:
        LOGGER.warning("Pattern rules file '%s' not found!", fn_rules)

    return patterns


def fetch_next_token_after_pos(
    text: str,
    boundary_candidate: BoundaryIndexAndLength,
    include_boundary: bool,
    remove_sentence_boundaries_at_end: bool,
    sentence_boundaries: List[str],
) -> Optional[str]:
    after_boundary_pos: int = boundary_candidate.index + boundary_candidate.length
    pos: int = text.find(" ", after_boundary_pos + 1)
    if pos == -1:
        return None
    start: int = boundary_candidate.index if include_boundary else after_boundary_pos
    post_boundary_token: str = text[start:pos]
    if remove_sentence_boundaries_at_end:  # XXX: added, or should this be run always?
        for boundary in sentence_boundaries:
            if post_boundary_token.endswith(boundary):
                return post_boundary_token[: -len(boundary)]
    return post_boundary_token


# ---------------------------------------------------------------------------


class BoundaryProcessor(metaclass=ABCMeta):
    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def is_no_sentence_boundary(
        self, text: str, boundary_candidate: BoundaryIndexAndLength
    ) -> bool:
        pass


class PreBoundaryRulesProcessor(BoundaryProcessor):
    patterns: Dict[re.Pattern, bool] = dict()

    fn_pre_boundary_rules: str
    encoding: str = ENCODING

    def __init__(self, fn_pre_boundary_rules: str, encoding: str = ENCODING):
        self.fn_pre_boundary_rules = fn_pre_boundary_rules
        self.encoding = encoding

    def init(self):
        self.load_rules(self.fn_pre_boundary_rules, self.encoding)

    def load_rules(self, fn_rules: str, encoding: str = ENCODING):
        LOGGER.debug(
            "Loading pre-boundary rules from: %s using encoding %s", fn_rules, encoding
        )
        self.patterns.update(load_patterns(fn_rules, encoding))
        LOGGER.debug("Done loading pre-boundary rules from: %s", fn_rules)

    def is_no_sentence_boundary(
        self, text: str, boundary_candidate: BoundaryIndexAndLength
    ) -> bool:
        rev_str = text[: boundary_candidate.index + boundary_candidate.length :][::-1]
        pos = rev_str.find(" ", boundary_candidate.length)
        end = pos if pos != -1 and pos - boundary_candidate.length > 0 else len(rev_str)
        token = rev_str[boundary_candidate.length : end].strip()[::-1]

        for pattern, is_decision in self.patterns.items():
            if pattern.fullmatch(token):
                return not is_decision

        return False


class PostBoundaryRulesProcessor(BoundaryProcessor):
    patterns: Dict[re.Pattern, bool] = dict()

    fn_post_boundary_rules: str
    sentence_boundaries: List[str]
    encoding: str = ENCODING

    def __init__(
        self,
        fn_post_boundary_rules: str,
        sentence_boundaries: List[str],
        encoding: str = ENCODING,
    ):
        self.fn_post_boundary_rules = fn_post_boundary_rules
        self.sentence_boundaries = sentence_boundaries
        self.encoding = encoding

    def init(self):
        self.load_rules(self.fn_post_boundary_rules, self.encoding)

    def load_rules(self, fn_rules: str, encoding: str = ENCODING):
        LOGGER.debug(
            "Loading post-boundary rules from: %s using encoding %s", fn_rules, encoding
        )
        self.patterns.update(load_patterns(fn_rules, encoding))
        LOGGER.debug("Done loading post-boundary rules from: %s", fn_rules)

    def is_no_sentence_boundary(
        self, text: str, boundary_candidate: BoundaryIndexAndLength
    ) -> bool:
        post_boundary_token: Optional[str] = fetch_next_token_after_pos(
            text, boundary_candidate, False, False, self.sentence_boundaries
        )

        if post_boundary_token is None:
            return False

        for pattern, is_decision in self.patterns.items():
            if pattern.fullmatch(post_boundary_token):
                return not is_decision

        return False


class PreBoundaryListProcessor(BoundaryProcessor):
    abbreviations_rev: Set[str] = set()
    longest_abbreviation: int = 0

    fn_pre_boundaries_list: str
    encoding: str = ENCODING
    is_auto_uppercase_first_letter_pre_list: bool = True

    def __init__(
        self,
        fn_pre_boundaries_list: str,
        encoding: str = ENCODING,
        is_auto_uppercase_first_letter_pre_list: bool = True,
    ) -> None:
        self.fn_pre_boundaries_list = fn_pre_boundaries_list
        self.encoding = encoding
        self.is_auto_uppercase_first_letter_pre_list = (
            is_auto_uppercase_first_letter_pre_list
        )

    def init(self):
        self.load_abbreviations(self.fn_pre_boundaries_list, self.encoding)

    def load_abbreviations(self, fn_boundaries_list: str, encoding: str = ENCODING):
        LOGGER.debug(
            "Loading pre-boundary list from: %s using encoding %s",
            fn_boundaries_list,
            encoding,
        )

        try:
            with open(fn_boundaries_list, "r", encoding=encoding) as fp:
                for line in fp:
                    line = line.rstrip()
                    if not line:
                        continue
                    if len(line) > self.longest_abbreviation:
                        self.longest_abbreviation = len(line)

                    self.abbreviations_rev.add(line[::-1])

                    if (
                        self.is_auto_uppercase_first_letter_pre_list
                        and not line[0].isupper()
                    ):
                        self.abbreviations_rev.add(f"{line[0].upper()}{line[1:]}"[::-1])

        except FileNotFoundError:
            LOGGER.warning(
                "Abbreviations list file '%s' not found!", fn_boundaries_list
            )

        LOGGER.debug("Done loading pre-boundary list from: %s", fn_boundaries_list)

    def is_no_sentence_boundary(
        self, text: str, boundary_candidate: BoundaryIndexAndLength
    ) -> bool:
        first_pos: int = max(0, boundary_candidate.index - self.longest_abbreviation)
        rev_str = text[
            first_pos : boundary_candidate.index + boundary_candidate.length
        ][::-1]
        pos: int = rev_str.find(" ", boundary_candidate.length)

        rev_abbrev_candidate = rev_str[:pos] if pos != -1 else rev_str
        if rev_abbrev_candidate.strip() in self.abbreviations_rev:
            return True

        return False


class PostBoundaryListProcessor(BoundaryProcessor):
    post_boundaries: Set[str] = set()
    longest_post_boundary: int = 0

    fn_post_boundaries_list: str
    sentence_boundaries: List[str]
    encoding: str = ENCODING

    def __init__(
        self,
        fn_post_boundaries_list: str,
        sentence_boundaries: List[str],
        encoding: str = ENCODING,
    ) -> None:
        self.fn_post_boundaries_list = fn_post_boundaries_list
        self.sentence_boundaries = sentence_boundaries
        self.encoding = encoding

    def init(self):
        self.load_post_boundaries(self.fn_post_boundaries_list, self.encoding)

    def load_post_boundaries(self, fn_boundaries_list: str, encoding: str = ENCODING):
        LOGGER.debug(
            "Loading post-boundary list from: %s using encoding %s",
            fn_boundaries_list,
            encoding,
        )

        try:
            with open(fn_boundaries_list, "r", encoding=encoding) as fp:
                for line in fp:
                    line = line.rstrip()
                    if not line:
                        continue
                    if len(line) > self.longest_post_boundary:
                        self.longest_post_boundary = len(line)

                    self.post_boundaries.add(line)
        except FileNotFoundError:
            LOGGER.warning(
                "Post-boundary list file '%s' not found!", fn_boundaries_list
            )

        LOGGER.debug("Done loading post-boundary list from: %s", fn_boundaries_list)

    def is_no_sentence_boundary(
        self, text: str, boundary_candidate: BoundaryIndexAndLength
    ) -> bool:
        post_boundary_token = fetch_next_token_after_pos(
            text, boundary_candidate, False, True, self.sentence_boundaries
        )

        if post_boundary_token is None:
            return False

        if post_boundary_token.strip() in self.post_boundaries:  # XXX: added .strip()
            return True

        return False


# ---------------------------------------------------------------------------


class AbstractSegmentizer(metaclass=ABCMeta):
    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def segmentize(self, text: str) -> List[str]:
        pass


class Segmentizer(AbstractSegmentizer):
    fn_sentence_boundaries: Optional[str] = None
    fn_pre_boundary_rules: Optional[str] = None
    fn_pre_boundaries_list: Optional[str] = None
    fn_post_boundary_rules: Optional[str] = None
    fn_post_boundaries_list: Optional[str] = None

    encoding: str = ENCODING

    is_auto_uppercase_first_letter_pre_list: bool = True

    sentence_boundaries: List[str] = list()
    boundary_processors: List[BoundaryProcessor] = list()

    def __init__(
        self,
        fn_sentence_boundaries: Optional[str] = None,
        fn_pre_boundary_rules: Optional[str] = None,
        fn_pre_boundaries_list: Optional[str] = None,
        fn_post_boundary_rules: Optional[str] = None,
        fn_post_boundaries_list: Optional[str] = None,
        encoding: str = ENCODING,
        is_auto_uppercase_first_letter_pre_list: bool = True,
    ):
        super().__init__()

        self.fn_sentence_boundaries = fn_sentence_boundaries
        self.fn_pre_boundary_rules = fn_pre_boundary_rules
        self.fn_pre_boundaries_list = fn_pre_boundaries_list
        self.fn_post_boundary_rules = fn_post_boundary_rules
        self.fn_post_boundaries_list = fn_post_boundaries_list

        self.encoding = encoding

        self.is_auto_uppercase_first_letter_pre_list = (
            is_auto_uppercase_first_letter_pre_list
        )

        self.init()

    @classmethod
    def create_default(cls, dn_resources: str = "resources") -> "Segmentizer":
        fn_sentence_boundaries = os.path.join(dn_resources, "boundariesFile.txt")
        fn_pre_boundary_rules = os.path.join(dn_resources, "preRules.txt")
        fn_pre_boundaries_list = os.path.join(dn_resources, "preList.txt")
        fn_post_boundary_rules = os.path.join(dn_resources, "postRules.txt")
        fn_post_boundaries_list = os.path.join(dn_resources, "postList.txt")

        segmentizer = cls(
            fn_sentence_boundaries=fn_sentence_boundaries,
            fn_pre_boundary_rules=fn_pre_boundary_rules,
            fn_pre_boundaries_list=fn_pre_boundaries_list,
            fn_post_boundary_rules=fn_post_boundary_rules,
            fn_post_boundaries_list=fn_post_boundaries_list,
        )

        return segmentizer

    def init(self):
        self._init_sentence_boundaries()
        self._init_boundary_processors()

    def _init_sentence_boundaries(self):
        self.sentence_boundaries = list()

        if self.fn_sentence_boundaries:
            try:
                LOGGER.debug(
                    "Loading boundary candiates from: %s using encoding %s",
                    self.fn_sentence_boundaries,
                    self.encoding,
                )
                with open(
                    self.fn_sentence_boundaries, "r", encoding=self.encoding
                ) as fp:
                    for line in fp:
                        line = line.rstrip()
                        if not line:
                            continue
                        self.sentence_boundaries.append(line)
                LOGGER.debug(
                    "Done loading boundary candidates from: %s",
                    self.fn_sentence_boundaries,
                )
            except FileNotFoundError:
                LOGGER.warning(
                    "Sentence boundary file '%s' not found!",
                    self.fn_sentence_boundaries,
                )

        if not self.sentence_boundaries:
            LOGGER.debug("No/Empty boundary candidates file found. Using defaults.")
            self.sentence_boundaries.append(".")
            self.sentence_boundaries.append("!")
            self.sentence_boundaries.append("?")

        # new line boundaries
        self.sentence_boundaries.extend([CR])

    def _init_boundary_processors(self):
        self.boundary_processors = list()

        self.boundary_processors.append(
            PreBoundaryListProcessor(
                self.fn_pre_boundaries_list,
                encoding=self.encoding,
                is_auto_uppercase_first_letter_pre_list=self.is_auto_uppercase_first_letter_pre_list,
            )
        )
        self.boundary_processors.append(
            PostBoundaryListProcessor(
                self.fn_post_boundaries_list,
                sentence_boundaries=self.sentence_boundaries,
                encoding=self.encoding,
            )
        )
        self.boundary_processors.append(
            PreBoundaryRulesProcessor(
                self.fn_pre_boundary_rules, encoding=self.encoding
            )
        )
        self.boundary_processors.append(
            PostBoundaryRulesProcessor(
                self.fn_post_boundary_rules,
                sentence_boundaries=self.sentence_boundaries,
                encoding=self.encoding,
            )
        )

        for boundary_processor in self.boundary_processors:
            boundary_processor.init()

    def segmentize(self, text: str) -> List[str]:
        sentences: List[str] = list()
        offset = 0
        while True:
            boundary_candidate = self.find_next_sentence_boundary(text, offset)
            if boundary_candidate.index != -1:
                cut_idx = boundary_candidate.index + boundary_candidate.length
                if self.is_sentence_boundary(text, boundary_candidate):
                    sentences.append(text[:cut_idx])
                    text = text[cut_idx:]
                    offset = 0
                else:
                    offset = cut_idx
            else:
                sentences.append(text)
                break
        return sentences

    def find_next_sentence_boundary(
        self, text: str, offset: int = 0
    ) -> BoundaryIndexAndLength:
        closestIndex: BoundaryIndexAndLength = BoundaryIndexAndLength(-1, -1)

        for sentence_boundary in self.sentence_boundaries:
            foundIndex: int = text.find(sentence_boundary, offset)
            if closestIndex.index == -1 or (
                foundIndex != -1 and foundIndex < closestIndex.index
            ):
                closestIndex = BoundaryIndexAndLength(
                    foundIndex, len(sentence_boundary)
                )
                if foundIndex != -1:
                    text = text[:foundIndex]

        return closestIndex

    def is_sentence_boundary(
        self, text: str, boundary_candidate: BoundaryIndexAndLength
    ) -> bool:
        if (
            text[
                boundary_candidate.index : boundary_candidate.index
                + boundary_candidate.length
            ]
            == CR
        ):
            return True

        for boundary_processor in self.boundary_processors:
            if boundary_processor.is_no_sentence_boundary(text, boundary_candidate):
                return False

        return True


class LineAwareSegmentizer(Segmentizer):
    is_trim_mode: bool = True
    """whether to fold all sequences of spaces into a single whitespace"""

    use_carriage_return_as_boundary: bool = True
    """preprocess text and add carriage returns (line feeds to be exact `\\n`) between subsequent lines or concat all together with a single whitespace"""

    use_empty_line_as_boundary: bool = True
    """preprocess text and if `True` all empty lines are considered sentence separators using carriage returns (line feeds to be exact `\\n`) or if `False` a single space to concatenate text pieces together (e.g. sentences that may have been split across empty lines)"""

    def __init__(
        self,
        fn_sentence_boundaries: Optional[str] = None,
        fn_pre_boundary_rules: Optional[str] = None,
        fn_pre_boundaries_list: Optional[str] = None,
        fn_post_boundary_rules: Optional[str] = None,
        fn_post_boundaries_list: Optional[str] = None,
        encoding: str = ENCODING,
        is_auto_uppercase_first_letter_pre_list: bool = True,
        is_trim_mode: bool = True,
        use_carriage_return_as_boundary: bool = True,
        use_empty_line_as_boundary: bool = True,
    ):
        super().__init__(
            fn_sentence_boundaries,
            fn_pre_boundary_rules,
            fn_pre_boundaries_list,
            fn_post_boundary_rules,
            fn_post_boundaries_list,
            encoding,
            is_auto_uppercase_first_letter_pre_list,
        )

        self.is_trim_mode = is_trim_mode
        self.use_carriage_return_as_boundary = use_carriage_return_as_boundary
        self.use_empty_line_as_boundary = use_empty_line_as_boundary

    # def __reduce__(self):
    #     args = (
    #         self.fn_sentence_boundaries,
    #         self.fn_pre_boundary_rules,
    #         self.fn_pre_boundaries_list,
    #         self.fn_post_boundary_rules,
    #         self.fn_post_boundaries_list,
    #         self.encoding,
    #         self.is_auto_uppercase_first_letter_pre_list,
    #         self.is_trim_mode,
    #         self.use_carriage_return_as_boundary,
    #         self.use_empty_line_as_boundary,
    #     )
    #     return (self.__class__, args, None, None)

    def _prepare_text(self, text: str) -> str:
        lines = text.splitlines(keepends=False)

        lines_processed = list()
        for line in lines:
            if not line or line.isspace():
                if self.use_empty_line_as_boundary:
                    lines_processed.append(CR)
                else:
                    lines_processed.append(f"{line}{SPACE}")
            else:
                if self.use_carriage_return_as_boundary:
                    lines_processed.append(f"{line}{CR}")
                else:
                    lines_processed.append(f"{line}{SPACE}")

        return "".join(lines_processed)

    def segmentize(self, text: str) -> List[str]:
        if not text:
            return []

        text = self._prepare_text(text)

        sentences = super().segmentize(text)

        if self.is_trim_mode:
            # collapse spaces
            sentences = [SPACE.join(sentence.split()).strip() for sentence in sentences]

        # remove empty lines
        sentences = [sentence for sentence in sentences if sentence]

        return sentences


# ---------------------------------------------------------------------------
