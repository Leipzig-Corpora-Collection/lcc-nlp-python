import logging
import os.path
import sys
import unicodedata
from abc import ABCMeta
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

from lcc.util import memoized_method

# ---------------------------------------------------------------------------

# src/main/java/de/uni_leipzig/asv/medusa/input/Tokenizer.java
# src/main/java/de/uni_leipzig/asv/medusa/input/AbstractWordTokenizer.java
# src/main/java/de/uni_leipzig/asv/medusa/input/CharacterBasedWordTokenizerImprovedImpl.java

# src/main/java/de/uni_leipzig/asv/medusa/input/AbstractInput.java

# src/main/java/de/uni_leipzig/asv/medusa/config/ClassConfig.java
# src/main/java/de/uni_leipzig/asv/medusa/config/ConfigurationContainer.java

# ---------------------------------------------------------------------------


LOGGER = logging.getLogger(__name__)


# fmt: off
DEFAULT_EXPLIZIT_CUT = [
    "(", ")", "{", "}", "[", "]", '"', ":", "!", "?", "¿", ";",
    "'", ",", "#", "«", "»", "^", "`", "´", "¨", "‹", "›", "“",
    "”", "„", "‘", "‚", "’", "$", "€", "「", "」", "『", "』",
    "〈", "〉", "《", "》", "₩", "¢", "฿", "₫", "₤", "₦", "元",
    "₪", "₱", "₨", "-", "\\",
]
# fmt: on

TOKEN_SENT_START = "%^%"
TOKEN_SENT_END = "%$%"
TOKEN_NUMBER = "%N%"
TOKEN_TAB = "%TAB%"

# from https://stackoverflow.com/a/14246025/9360161
UNICODE_CATEGORY = defaultdict(list)
for c in map(chr, range(sys.maxunicode + 1)):
    UNICODE_CATEGORY[unicodedata.category(c)].append(c)

# https://www.compart.com/en/unicode/category/Po
UNICODE_CATEGORY_PUNCTUATION_OTHER = "Po"
UNICODE_PUNCTUATION_OTHER = UNICODE_CATEGORY[UNICODE_CATEGORY_PUNCTUATION_OTHER]


# ---------------------------------------------------------------------------


class ConfigBase:
    strConfigFile: Optional[str] = None

    def config(self):
        if not self.strConfigFile:
            LOGGER.debug(
                "Property 'strConfigFile' not set. Automatic configuration disabled."
            )
            return
        if not os.path.exists(self.strConfigFile):
            LOGGER.debug(
                "Configuration file 'strConfigFile' not found. Automatic configuration disabled."
            )
            return

        LOGGER.debug(
            "Configuring class '%s' with '%s'.",
            f"{self.__class__.__module__}.{self.__class__.__name__}",
            self.strConfigFile,
        )

        import inspect
        import typing

        def is_optional_type(type_annot):
            return typing.get_origin(type_annot) is typing.Union and type(
                None
            ) in typing.get_args(type_annot)

        def is_base_type(type_annot):
            return type_annot in (int, float, str)

        def extract_optional_inner_type(type_annot):
            if is_base_type(type_annot):
                return type_annot
            if not is_optional_type(type_annot):
                return None
            return list(set(typing.get_args(type_annot)) - {type(None)})[0]

        # get all classes that can be configured
        clazzes = []
        for clazz in inspect.getmro(self.__class__):
            if clazz == ConfigBase:
                break
            if clazz == object:
                break
            clazzes.append(clazz)

        # now start from the parents to the child in case someone overwrites some attribute ...
        for clazz in reversed(clazzes):
            for field, ftype in inspect.get_annotations(clazz).items():
                rtype = extract_optional_inner_type(ftype)
                # LOGGER.debug(
                #     "%s: %s is %s (base?=%s, optional?=%s) -> %s",
                #     f"{clazz.__module__}.{clazz.__name__}",
                #     field,
                #     ftype,
                #     is_base_type(ftype),
                #     is_optional_type(ftype),
                #     f"{rtype.__module__}.{rtype.__name__}",
                # )
                if not rtype:
                    continue

                # TODO: configure from parsing `strConfigFile`
                LOGGER.warning(
                    "Can not configure field '%s' of '%s' with type %s !",
                    field,
                    f"{clazz.__module__}.{clazz.__name__}",
                    f"{rtype.__module__}.{rtype.__name__}",
                )


"""
import logging; logging.basicConfig(level=logging.DEBUG)
import inspect
import lcc.tokenizer
import typing

# inspect.get_annotations(lcc.tokenizer.CharacterBasedWordTokenizerImproved.__base__)

class A(lcc.tokenizer.ConfigBase):
  e: str = 1
  a: typing.Optional[float] = None
  b: int
  c = "abc"
  def x(self): pass

class B(A):
  b: float
  f: int = 3

A().config()
b = B()
b.strConfigFile = "../resources/Medusa/medusa_config.xml"
b.config()
"""


# ---------------------------------------------------------------------------


class AbstractWordTokenizer(ConfigBase, metaclass=ABCMeta):
    # abbrev/default.abbrev
    strAbbrevListFile: Optional[str] = None
    objAbbrevList: Set[str]
    # Medusa/wordlists/100-wn-all.txt
    TOKENISATION_CHARACTERS_FILE_NAME: Optional[str] = None
    objTokenisationCharacters: Dict[str, int]
    EXPLIZIT_CUT: List[str]

    def init(self):
        self.config()

        self._load_abbreviations()
        self._load_tokenization_characters()

    def _load_abbreviations(self):
        self.objAbbrevList = set()
        if not self.strAbbrevListFile:
            # raise ValueError("Property 'strAbbrevListFile' not set.")
            LOGGER.warning("Property 'strAbbrevListFile' not set.")
        elif not os.path.exists(self.strAbbrevListFile):
            LOGGER.warning(
                "Could not load abbreviation list from file: %s", self.strAbbrevListFile
            )
        else:
            with open(self.strAbbrevListFile, "r", encoding="utf-8") as fp:
                for line in fp:
                    self.objAbbrevList.add(f"{line.strip()}.")

    def _load_tokenization_characters(self):
        if not self.TOKENISATION_CHARACTERS_FILE_NAME:
            LOGGER.warning(
                "TOKENISATION_CHARACTERS_FILE_NAME not set. Using default characters."
            )
            self.EXPLIZIT_CUT = DEFAULT_EXPLIZIT_CUT
        elif not os.path.exists(self.TOKENISATION_CHARACTERS_FILE_NAME):
            LOGGER.warning(
                "TOKENISATION_CHARACTERS_FILE_NAME '%s' not found. Using default characters.",
                self.TOKENISATION_CHARACTERS_FILE_NAME,
            )
            self.EXPLIZIT_CUT = DEFAULT_EXPLIZIT_CUT
        else:
            self.objTokenisationCharacters = dict()
            with open(
                self.TOKENISATION_CHARACTERS_FILE_NAME, "r", encoding="utf-8"
            ) as fp:
                for line in fp:
                    parts = line.rstrip().split("\t")
                    if len(parts) == 2:
                        char = parts[1].strip()
                        self.objTokenisationCharacters[char] = len(char)
                    else:
                        LOGGER.warning(
                            "Ingnoring line (reading tokenisation characters): %s", line
                        )

            # remove some elements of this list
            for char in [
                TOKEN_SENT_START,
                TOKEN_SENT_END,
                TOKEN_NUMBER,
                TOKEN_TAB,  # XXX: added here
                "_TAB_",
                "_",
            ]:
                self.objTokenisationCharacters.pop(char, None)
            # remove, just for Bible work
            # self.objTokenisationCharacters.pop("-", None)
            self.objTokenisationCharacters.pop(".", None)

            self.EXPLIZIT_CUT = list(self.objTokenisationCharacters.keys())

    @abstractmethod
    def execute(self, line: str) -> str:
        pass

    def get_char_len(self, char: str) -> int:
        if char is None:
            return 1
        return max(self.objTokenisationCharacters.get(char, 0), 0)

    # ----------------------------------------------------

    @staticmethod
    def get_whitespace_before_dot(pos_dot: int, fragment: str) -> int:
        pos = pos_dot - 1
        while pos > 0:
            if fragment[pos].isspace():
                return pos
            pos -= 1
        return 0

    @staticmethod
    def get_whitespace_positions(line: str) -> List[int]:
        # ArrayList<int[]>
        tokens = []

        for pos, char in enumerate(line):
            if char.isspace():
                tokens.append(pos)
        tokens.append(len(line))

        return tokens

    @staticmethod
    def get_tokens(line: str, whitespace_positions: List[int]) -> List[str]:
        if len(whitespace_positions) == 0 or whitespace_positions[0] == len(line):
            return [line]

        tokens = []
        start = 0
        for end in whitespace_positions:
            token = line[start:end].strip()
            if token:
                tokens.append(token)
            start = end + 1

        token = line[start:].strip()
        if token:
            tokens.append(token)

        return tokens


# ---------------------------------------------------------------------------


class MWUWordTokenizerMixin:
    # MWUs/<lang>.mwu
    MWU_FILE_NAME: Optional[str] = None
    knownNumbers2: Optional[Set[str]] = None
    MWUMap: Optional[Dict[str, str]] = None
    MWUIgnored: Optional[Set[str]] = None

    def load_MWU(self, tokenizer: AbstractWordTokenizer):
        LOGGER.debug("MWU_FILE_NAME: %s", self.MWU_FILE_NAME)
        if not self.MWU_FILE_NAME:
            return

        LOGGER.info(
            "Using multi word units (mwu) in %s",
            os.path.abspath(self.MWU_FILE_NAME),
        )
        self.knownNumbers2 = set()
        if not os.path.exists(self.MWU_FILE_NAME):
            LOGGER.warning("Couldn't read mwu file: %s", self.MWU_FILE_NAME)
            return

        self.MWUMap = dict()
        self.MWUIgnored = set()
        with open(self.MWU_FILE_NAME, "r", encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                mwu_original: str = line.strip().split("\t")[0]

                # removing backslashes
                if "\\" in mwu_original:
                    mwu_original_temp = mwu_original.replace("\\", " ").strip()
                    LOGGER.debug(
                        "correcting mwu from >>>%s<<< to >>>%s<<<",
                        mwu_original,
                        mwu_original_temp,
                    )
                    mwu_original = mwu_original_temp

                if " " in mwu_original and "   " not in mwu_original:
                    mwu_tokenized = tokenizer.execute(mwu_original).strip()
                    self.knownNumbers2.add(mwu_tokenized)
                    self.MWUMap[mwu_tokenized] = mwu_original
                else:
                    self.MWUIgnored.add(line)

        LOGGER.info("%s different mwu's found and loaded.", len(self.knownNumbers2))

    def get_MWU_tokens(self, line: str, whitespace_positions: List[int]) -> List[str]:
        # single word (Einzelwortzeile)
        if len(whitespace_positions) == 0 or whitespace_positions[0] == len(line):
            return []

        if not self.knownNumbers2:
            return []

        # remove last position if it is the line length
        if whitespace_positions[-1] == len(line):
            whitespace_positions = whitespace_positions[:-1]

        # two words (Zweiwortzeile), and line consists of one MWU
        if len(whitespace_positions) == 1:
            if line.strip() in self.knownNumbers2:
                return [line]
            else:
                return []

        end = len(whitespace_positions) - 2
        line_strip = line.strip()
        tokens: List[str] = list()

        # detecting MWU at the beginning of a sentence
        possible_MUW_prefix = line[: whitespace_positions[1]]
        LOGGER.debug('MWU-Begin: "%s"', possible_MUW_prefix)
        possible_MWUs: Set[str] = {
            mwu
            for mwu in self.knownNumbers2
            if possible_MUW_prefix <= mwu < f"{possible_MUW_prefix}?"
        }
        for mwu_defined in possible_MWUs:
            if line_strip == mwu_defined:
                tokens.append(mwu_defined)
            elif line_strip.startswith(f"{mwu_defined} "):
                tokens.append(mwu_defined)

        # detecting MWUs inside a sentence
        for idx in range(end):
            possible_MUW_prefix = line[
                whitespace_positions[idx] + 1 : whitespace_positions[idx + 2]
            ]
            LOGGER.debug('MWU: "%s"', possible_MUW_prefix)
            possible_MWUs = {
                mwu
                for mwu in self.knownNumbers2
                if possible_MUW_prefix <= mwu < f"{possible_MUW_prefix}?"
            }
            for mwu_defined in possible_MWUs:
                LOGGER.debug("POSSIBLE MWU: %s", mwu_defined)
                if line_strip.startswith(
                    f"{mwu_defined} ", whitespace_positions[idx] + 1
                ) or line.endswith(mwu_defined):
                    LOGGER.debug('MWU FOUND: "%s"', mwu_defined)
                    tokens.append(mwu_defined)

        possible_MUW_prefix = line[whitespace_positions[end] + 1]
        LOGGER.debug('MWU-END: "%s"', possible_MUW_prefix)
        possible_MWUs = {
            mwu
            for mwu in self.knownNumbers2
            if possible_MUW_prefix <= mwu < f"{possible_MUW_prefix}z"
        }
        for mwu_defined in possible_MWUs:
            LOGGER.debug("POSSIBLE MWU: %s", mwu_defined)
            if line.endswith(mwu_defined):
                LOGGER.debug('MWU-END FOUND: "%s"', mwu_defined)
                tokens.append(mwu_defined)

        return tokens

    @staticmethod
    def make_word(tokens: List[str], start: int, end: int) -> str:
        return " ".join(tokens[start:end]).strip()

    def get_tokens_with_MWU(self, line_split: List[str]) -> List[str]:
        """Known words can include word groups, so eventually tokens have to
        be merged back after splitting them at spaces."""
        # TODO: is this method ever used?
        tokens = list()

        for idx in range(len(line_split)):
            largest_seen_word = line_split[idx]
            if self.knownNumbers2:
                # add largestSeenWord as next token
                tokens.append(largest_seen_word)
                for jdx in range(len(line_split), idx, -1):
                    # for jdx in range(idx + 1, len(line_split)):
                    cur_attempt = self.make_word(line_split, idx, jdx)

                    if cur_attempt in self.knownNumbers2:
                        largest_seen_word = cur_attempt
                        # add largestSeenWord as next token
                        tokens.append(largest_seen_word)
                        break

            else:
                # XXX: moved this behind else, otherwise add this token/MWU twice?
                # add largestSeenWord as next token
                tokens.append(largest_seen_word)

        return tokens


# ---------------------------------------------------------------------------


class CharacterBasedWordTokenizerImproved(AbstractWordTokenizer):
    # Medusa/wordlists/tokenization_character_actions.txt
    strCharacterActionsFile: Optional[str] = None
    charActions: Dict[str, List[int]]  # TreeMap<String, Integer[]>
    characters: List[str]
    # Medusa/wordlists/fixed_tokens.txt
    fixedTokensFile: Optional[str] = None
    objFixedTokens: Set[str]
    # whether to replace any occurence of numbers
    boolReplaceNumbers: bool = False

    evilSentenceEndCharacters = ['"', "'", "“", "”", "„", ",", "«", "»"]
    punctuationCharacters = [".", "!", "?", ";", "-", ":"]

    def __init__(
        self,
        strAbbrevListFile: Optional[str] = None,
        TOKENISATION_CHARACTERS_FILE_NAME: Optional[str] = None,
        strCharacterActionsFile: Optional[str] = None,
        fixedTokensFile: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.strAbbrevListFile = strAbbrevListFile
        self.TOKENISATION_CHARACTERS_FILE_NAME = TOKENISATION_CHARACTERS_FILE_NAME
        self.strCharacterActionsFile = strCharacterActionsFile
        self.fixedTokensFile = fixedTokensFile

        self.init()

    @classmethod
    def create_default(
        cls, dn_resources: str = "resources"
    ) -> "CharacterBasedWordTokenizerImproved":
        strAbbrevListFile = os.path.join(dn_resources, "default.abbrev")
        TOKENISATION_CHARACTERS_FILE_NAME = os.path.join(dn_resources, "100-wn-all.txt")
        strCharacterActionsFile = os.path.join(
            dn_resources, "tokenization_character_actions.txt"
        )
        fixedTokensFile = os.path.join(dn_resources, "fixed_tokens.txt")

        tokenizer = cls(
            strAbbrevListFile=strAbbrevListFile,
            TOKENISATION_CHARACTERS_FILE_NAME=TOKENISATION_CHARACTERS_FILE_NAME,
            strCharacterActionsFile=strCharacterActionsFile,
            fixedTokensFile=fixedTokensFile,
        )

        return tokenizer

    # def __reduce__(self):
    #     args = (
    #         self.strAbbrevListFile,
    #         self.TOKENISATION_CHARACTERS_FILE_NAME,
    #         self.strCharacterActionsFile,
    #         self.fixedTokensFile,
    #     )
    #     return (self.__class__, args, None, None)

    def init(self):
        super().init()

        self._load_fixed_tokens()
        self._load_tokenization_character_actions()

    def _load_fixed_tokens(self):
        # UnTok
        self.objFixedTokens = set()
        if not self.fixedTokensFile:
            LOGGER.warning("Fixed tokens file not set.")
        elif not os.path.exists(self.fixedTokensFile):
            LOGGER.warning("Fixed tokens file not found: %s", self.fixedTokensFile)
        else:
            with open(self.fixedTokensFile, "r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    if " " in line:
                        LOGGER.warning("Fixed token with space: '%s'", line)
                    self.objFixedTokens.add(line)

    def _load_tokenization_character_actions(self):
        # tokenization character actions
        self.charActions = dict()
        if not self.strCharacterActionsFile:
            LOGGER.warning("Character action file not set.")
        elif not os.path.exists(self.strCharacterActionsFile):
            LOGGER.warning(
                "Character action file not found: %s", self.strCharacterActionsFile
            )
        else:
            LOGGER.debug("Character action file: %s", self.strCharacterActionsFile)

            actionsMap = {"nothing": 0, "whitespace": 1, "delete": 2}

            with open(self.strCharacterActionsFile, "r", encoding="utf-8") as fp:
                # first line of document contains table headers, skip it
                fp.readline()
                for lno, line in enumerate(fp):
                    line = line.strip()
                    if line:
                        parts = line.split("\t")
                        if len(parts) != 6:
                            LOGGER.error(
                                "Character actions file malformatted. (lineno. %s)", lno
                            )
                            continue
                        _, char, *actions = parts
                        self.charActions[char] = [
                            actionsMap.get(action, 3) for action in actions
                        ]

        self.characters = list(self.charActions.keys())

    def execute(self, line: str) -> str:
        if (
            not hasattr(self, "charActions")
            and not hasattr(self, "characters")
            and not hasattr(self, "objFixedTokens")
        ):
            LOGGER.warning("Forgot to call .init()? Calling ...")
            self.init()

        line = self.fix_evil_sentence_end(line)

        start = 0
        is_sentence_end = False
        objBuffer = []

        for end in self.get_whitespace_positions(line):
            if end - start > 0:
                word = line[start:end]

                if word in [TOKEN_SENT_START, TOKEN_SENT_END, TOKEN_NUMBER, TOKEN_TAB]:
                    objBuffer.append(word)

                else:
                    is_sentence_end = end == len(line)
                    LOGGER.debug(
                        "process prefix ON %s sentenceEND %s", word, is_sentence_end
                    )
                    tokenized_word = self.process_prefix(word, is_sentence_end)

                    while word != tokenized_word:
                        word = tokenized_word
                        tokenized_word = self.execute(word)

                    tokenized_word = self.process_infix(word)

                    while word != tokenized_word:
                        word = tokenized_word
                        tokenized_word = self.execute(word)

                    objBuffer.append(tokenized_word)

            start = end + 1

        return " ".join(objBuffer).strip()

    def fix_evil_sentence_end(self, line: str) -> str:
        for char in self.evilSentenceEndCharacters:
            for punct in self.punctuationCharacters:
                end = f"{punct}{char}"
                if line.endswith(end):
                    return f"{line[:-len(end)]} {end}"
        return line

    def process_prefix(self, word: str, is_sentence_end: bool) -> str:
        LOGGER.debug("processPrefix: strWord %s", word)
        if not word:
            return ""
        if len(word) == 1:
            return self.process_single_number(word)

        objBuffer = []
        offset = idx = 0
        # for every character to tokenize by
        while idx < len(self.characters):
            # XXX: check fixed tokens here? but is possibly partial match
            # ahocorasick?

            char = self.characters[idx]
            if word.startswith(char, offset):
                charlen = self.get_char_len(char)

                desiredAction: int
                if offset == 0:
                    # use word start action
                    desiredAction = self.charActions[char][0]
                else:
                    # use word mid action
                    desiredAction = self.charActions[char][1]

                if desiredAction == 1:
                    # whitespace
                    objBuffer.append(word[offset : offset + charlen])
                    objBuffer.append(" ")
                elif desiredAction in (0, 3):
                    # do nothing
                    objBuffer.append(word[offset : offset + charlen])
                elif desiredAction == 2:
                    # delete the character
                    pass
                else:
                    # NOTE: this should not happen. Append anyway to not lose something.
                    objBuffer.append(word[offset : offset + charlen])

                # next position of string
                offset += charlen
                # do it again
                idx = 0

            idx += 1

        objBuffer.append(self.process_suffix(word[offset:], is_sentence_end))
        return "".join(objBuffer).strip()

    def process_suffix(self, word: str, is_sentence_end: bool) -> str:
        LOGGER.debug("processSuffix: strWord %s", word)
        # if not word:
        #     return ""
        if len(word) == 1:
            return self.process_single_number(word)

        objBuffer = []
        result = ""
        idx = 0

        is_sentence_end_tmp = is_sentence_end
        word_tmp = word

        while idx < len(self.characters):
            char = self.characters[idx]

            if word_tmp.endswith(char):
                charlen = self.get_char_len(char)

                whatToDo: int
                if not is_sentence_end_tmp:
                    whatToDo = 2
                else:
                    whatToDo = 3
                    is_sentence_end_tmp = False

                desiredAction: int = self.charActions[char][whatToDo]

                if desiredAction == 1:
                    # whitespace before char
                    result = f" {word_tmp[-charlen:]}{result}"
                    word_tmp = word_tmp[:-charlen]
                elif desiredAction in (0, 3):
                    # do nothing
                    result = f"{word_tmp[-charlen:]}{result}"
                    word_tmp = word_tmp[:-charlen]
                elif desiredAction == 2:
                    # delete the character
                    word_tmp = word_tmp[-charlen:]
                else:
                    # NOTE: this should not happen. Append anyway to not lose something
                    result = f"{word_tmp[-charlen:]}{result}"
                    word_tmp = word_tmp[:-charlen]

            idx += 1

        # by now, we should not have any dots in it anymore. Why do we do this?
        objBuffer.append(self.process_dot_as_prefix(word_tmp, is_sentence_end))
        objBuffer.append(result)
        return "".join(objBuffer).strip()

    def process_infix(self, line: str) -> str:
        start = 0
        objBuffer = []

        for end in self.get_whitespace_positions(line):
            if end - start > 0:
                word = line[start:end]

                if word in [TOKEN_SENT_START, TOKEN_SENT_END, TOKEN_NUMBER, TOKEN_TAB]:
                    objBuffer.append(word)

                else:
                    tokenized_word = self.process_infix1(word)

                    while word != tokenized_word:
                        word = tokenized_word
                        tokenized_word = self.process_infix(word)

                    objBuffer.append(tokenized_word)

            start = end + 1

        return " ".join(objBuffer).strip()

    def process_infix1(self, word: str) -> str:
        if len(word.strip()) == 1:
            return word
        if word in self.objFixedTokens:
            return word

        word_tmp = word
        objBuffer: List[str] = []
        idx = 0
        # for every character to tokenize by
        while idx < len(self.characters):
            char = self.characters[idx]
            charpos = word_tmp.find(char)

            if charpos >= 0:
                if charpos in (0, len(word_tmp) - 1):
                    strBuffer = "".join(objBuffer)
                    if char not in strBuffer:
                        # wordEnd already been handled by process_suffix
                        idx += 1
                        continue
                    else:
                        # objBuffer contains first part of word with multiple instances of a found character
                        objBuffer.append(word)
                        return "".join(objBuffer)

                charlen = self.get_char_len(char)
                # use word mid action
                desiredAction: int = self.charActions[char][1]

                if desiredAction == 1:
                    # whitespace
                    LOGGER.debug(
                        "processInfix: char %s found at: %s length: %s tmpwordlength %s %s",
                        char,
                        charpos,
                        charlen,
                        len(word_tmp),
                        word_tmp,
                    )
                    word_tmp = word_tmp[charpos + charlen :]
                    LOGGER.debug(
                        "processInfix: strTMPWord (whitespace) %s append: %s %s",
                        word_tmp,
                        word[:charpos],
                        char,
                    )
                    objBuffer.append(word[:charpos])
                    objBuffer.append(" ")
                    objBuffer.append(char)
                    objBuffer.append(" ")
                elif desiredAction in (0, 3):
                    # do nothing
                    word_tmp = word_tmp[charpos + charlen :]
                    objBuffer.append(word[: charpos + charlen])
                elif desiredAction == 2:
                    # delete the character
                    word_tmp = word_tmp[charpos + charlen :]
                    objBuffer.append(word[:charpos])
                else:
                    # NOTE: this should not happen. Append anyway to not lose something.
                    word_tmp = word_tmp[charpos + charlen :]
                    objBuffer.append(word[: charpos + charlen])

                if char in word_tmp:
                    LOGGER.debug("another %s detected.", char)
                    word = word_tmp
                    continue
                else:
                    objBuffer.append(word_tmp)
                    word_tmp = "".join(objBuffer)
                    word = word_tmp
                    objBuffer[:] = []

            idx += 1

        return word

    def process_single_number(self, word: str) -> str:
        if not word:
            return ""
        if self.boolReplaceNumbers and word[0].isdigit():
            return TOKEN_NUMBER
        return word

    def process_number(self, word: str) -> str:
        if self.boolReplaceNumbers and self.is_number(word):
            return TOKEN_NUMBER
        return word

    def process_dot_as_prefix(self, word: str, is_sentence_end: bool) -> str:
        if len(word) == 1 or self.is_sequence_of_dot(word):
            return self.process_single_number(word)

        if not word.startswith("."):
            return self.process_dot_as_suffix(word, is_sentence_end).strip()

        objBuffer = []
        if "." in self.charActions:
            # get wordstart action
            action = self.charActions["."][0]
            if action == 1:
                # whitespace
                objBuffer.append(".")
                objBuffer.append(" ")
            elif action in (0, 3):
                # do nothing
                objBuffer.append(".")
            elif action == 2:
                # delete char
                pass
            else:
                # NOTE: this should not happen. Append anyway to not lose something.
                objBuffer.append(".")
        else:
            # fallback to simple append
            objBuffer.append(".")  # XXX: missing space append after?

        objBuffer.append(self.process_dot_as_suffix(word[1:], is_sentence_end))
        return "".join(objBuffer).strip()

    def process_dot_as_suffix(self, word: str, is_sentence_end: bool) -> str:
        if len(word) == 1 or word in self.objAbbrevList:
            LOGGER.debug("isAbbrev or not relevant: %s", word)
            return self.process_single_number(word)

        # word not ends with "." (dot)
        if not word.endswith("."):
            LOGGER.debug("!!isAbbrev: %s", word)
            return self.process_infix_apostrophe(word)

        # word ends with "." (dot)
        cnt_dots = self.detect_sequence_of_dot(word)
        if cnt_dots > 1:
            LOGGER.debug("isDotSeq: %s", word[:-cnt_dots])
            return " ".join(
                [self.process_infix_apostrophe(word[:-cnt_dots]), "." * cnt_dots]
            ).strip()

        LOGGER.debug("!!!!!!!!!!!!!isAbbr: %s", word[:-1])

        objBuffer = []
        objBuffer.append(self.process_infix_apostrophe(word[:-1]))

        if "." in self.charActions:
            whatToDo: int
            if not is_sentence_end:
                # get wordend action
                whatToDo = 2
            else:
                whatToDo = 3

            action = self.charActions["."][whatToDo]
            if action == 1:
                # whitespace
                objBuffer.append(". ")
            elif action in (0, 3):
                # do nothing
                objBuffer.append(".")
            elif action == 2:
                # delete the character
                pass
            else:
                # NOTE: this should not happen. Append anyway to not lose something.
                objBuffer.append(".")

        else:
            # fallback to simple append
            objBuffer.append(" .")  # XXX: ?

        return "".join(objBuffer)

    def process_infix_apostrophe(self, word: str) -> str:
        return self.process_number(word)

    # TODO: ignores TOKENISATION_CHARACTERS_FILE_NAME config
    def get_char_len(self, char: str) -> int:
        if char is None:
            return 1
        return len(char)

    @staticmethod
    def needs_split(word: str) -> bool:
        for char in word:
            if not (
                char.isalpha()
                or unicodedata.category(char) == UNICODE_CATEGORY_PUNCTUATION_OTHER
            ):
                return False
        return True

    @staticmethod
    def needs_number_split(word: str) -> bool:
        # if words ends with . and any OTHER_PUNCTUATION like %S
        if (
            word.endswith(".")
            and unicodedata.category(word[-2]) == UNICODE_CATEGORY_PUNCTUATION_OTHER
        ):
            return True

        for char in word:
            if not (
                char.isdigit()
                or unicodedata.category(char) == UNICODE_CATEGORY_PUNCTUATION_OTHER
            ):
                return False

        # if any time value followed by . so seperate the last .
        if word.endswith(".") and word.find(",") != -1 and word.find(":") != -1:
            return True

        return False

    @staticmethod
    def is_number(word: str) -> bool:
        for char in word:
            if not (
                char.isdigit()
                or unicodedata.category(char) == UNICODE_CATEGORY_PUNCTUATION_OTHER
                or char == "-"
            ):
                return False
        return True

    @staticmethod
    def is_sequence_of_dot(word: str) -> bool:
        for char in word:
            if char != ".":
                return False
        return True

    @staticmethod
    def detect_sequence_of_dot(word: str) -> int:
        cnt = 0
        for char in word[::-1]:
            if char != ".":
                return cnt
            cnt += 1
        # XXX: is this return correct, shouldn't that be cnt? original might be wrong?
        return 0


# ---------------------------------------------------------------------------


# TODO: this needs to be a bit improved, it shouldn't really do much (anything) at this moment ...


class CharacterBasedWordTokenizerImprovedWithMWUSupport(
    MWUWordTokenizerMixin, CharacterBasedWordTokenizerImproved
):
    def __init__(
        self,
        strAbbrevListFile: Optional[str] = None,
        TOKENISATION_CHARACTERS_FILE_NAME: Optional[str] = None,
        strCharacterActionsFile: Optional[str] = None,
        fixedTokensFile: Optional[str] = None,
        MWU_FILE_NAME: Optional[str] = None,
    ) -> None:
        super().__init__(
            strAbbrevListFile=strAbbrevListFile,
            TOKENISATION_CHARACTERS_FILE_NAME=TOKENISATION_CHARACTERS_FILE_NAME,
            strCharacterActionsFile=strCharacterActionsFile,
            fixedTokensFile=fixedTokensFile,
        )

        self.MWU_FILE_NAME = MWU_FILE_NAME

        self.init()

    @classmethod
    def create_default(
        cls,
        dn_resources: str = "resources",
        MWU_FILE_NAME: Optional[str] = None,
    ) -> "CharacterBasedWordTokenizerImprovedWithMWUSupport":
        strAbbrevListFile = os.path.join(dn_resources, "default.abbrev")
        TOKENISATION_CHARACTERS_FILE_NAME = os.path.join(dn_resources, "100-wn-all.txt")
        strCharacterActionsFile = os.path.join(
            dn_resources, "tokenization_character_actions.txt"
        )
        fixedTokensFile = os.path.join(dn_resources, "fixed_tokens.txt")

        tokenizer = cls(
            strAbbrevListFile=strAbbrevListFile,
            TOKENISATION_CHARACTERS_FILE_NAME=TOKENISATION_CHARACTERS_FILE_NAME,
            strCharacterActionsFile=strCharacterActionsFile,
            fixedTokensFile=fixedTokensFile,
            MWU_FILE_NAME=MWU_FILE_NAME,
        )

        return tokenizer

    def init(self):
        super().init()

        self._load_mwus()

    def _load_mwus(self):
        self.load_MWU(self)
        if self.knownNumbers2:
            self.objFixedTokens.update(self.knownNumbers2)

    # TODO: handle tokenization (infix) differently? (merge tokens together afterwards?)


# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignedSentence:
    raw: str
    tokenized: str

    def __str__(self) -> str:
        return self.raw

    @memoized_method(maxsize=1)
    def tokens(self) -> List[str]:
        return self.tokenized.split(" ")

    @memoized_method(maxsize=1)
    def alignment_indices(self) -> List[Tuple[int, int]]:
        indices: List[Tuple[int, int]] = []
        start: int = 0
        for token in self.tokens():
            start = self.raw.index(token, start)
            indices.append((start, start + len(token)))
            start = start + len(token)
        # NOTE: strict check if valid mapping
        if len(self.raw) != start:
            raise IndexError(
                "Mapping of 'tokenized' to 'raw' might be invalid "
                "since 'raw' seems to have more content than 'tokenized'?"
            )
        return indices

    @memoized_method(maxsize=1)
    def tokens_glue(self) -> List[bool]:
        glues: List[bool] = []
        last: int = -1
        for start, end in self.alignment_indices():
            glues.append(last == start)
            last = end
        return glues[1:]


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
