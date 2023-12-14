try:
    import spacy  # noqa: F401
except ImportError:
    raise RuntimeError(
        "SpaCy integration requires the installation of the optional dependency 'spacy'!"
    )

from itertools import chain
from typing import Iterable
from typing import Iterator
from typing import Optional

import srsly
from spacy import util
from spacy.tokens import Doc
from spacy.vocab import Vocab

from lcc.segmentizer import LineAwareSegmentizer
from lcc.tokenizer import AlignedSentence
from lcc.tokenizer import CharacterBasedWordTokenizerImproved

# ---------------------------------------------------------------------------


class Tokenizer:
    def __init__(
        self,
        vocab: Vocab,
        dn_tokenizer_resources: str,
        dn_segmentizer_resources: Optional[str] = None,
    ):
        """Create a `Tokenizer`, to create `Doc` objects given unicode text.

        vocab (Vocab): A storage container for lexical types.
        dn_tokenizer_resources (str): Path to LCC tokenizer resources.
        dn_segmentizer_resources (str): Path to LCC sentence segmentizer resources. Optional.

        EXAMPLE:
            >>> tokenizer = Tokenizer(nlp.vocab, "resources/tokenizer")

        DOCS: https://spacy.io/api/tokenizer#init
        """
        self.vocab = vocab

        self._dn_tokenizer_resources = dn_tokenizer_resources
        self._dn_segmentizer_resources = dn_segmentizer_resources

        self.lcc_tokenizer: CharacterBasedWordTokenizerImproved
        self.lcc_segmentizer: Optional[LineAwareSegmentizer] = None

        self._init()

    def _init(self):
        self.lcc_tokenizer = CharacterBasedWordTokenizerImproved.create_default(
            self._dn_tokenizer_resources
        )

        self.lcc_segmentizer = None
        if self._dn_segmentizer_resources:
            self.lcc_segmentizer = LineAwareSegmentizer.create_default(
                self._dn_segmentizer_resources
            )

    def __call__(self, text: str) -> Doc:
        """Tokenize a string.

        string (str): The string to tokenize.
        RETURNS (Doc): A container for linguistic annotations.

        DOCS: https://spacy.io/api/tokenizer#call
        """

        # sentence segment
        if self.lcc_segmentizer:
            return self.segment_and_tokenize(text)

        # plain tokenize
        return self.tokenize(text)

    def segment_and_tokenize(self, text: str) -> Doc:
        if not self.lcc_segmentizer:
            # TODO: warn that no segmentation is possible!
            return self.tokenize(text)

        sentences = self.lcc_segmentizer.segmentize(text)
        sentences_tokenized = list(map(self.lcc_tokenizer.execute, sentences))
        sentences_aligned = [
            AlignedSentence(sentence, tokenized)
            for sentence, tokenized in zip(sentences, sentences_tokenized)
        ]

        words = [sentence_aligned.tokens() for sentence_aligned in sentences_aligned]
        spaces = [
            [not glue for glue in sentence_aligned.tokens_glue()] + [True]
            for sentence_aligned in sentences_aligned
        ]
        sent_starts = [
            [True] + ([False] * (len(sentence_words) - 1)) for sentence_words in words
        ]

        return Doc(
            self.vocab,
            words=list(chain.from_iterable(words)),
            spaces=list(chain.from_iterable(spaces)),
            sent_starts=list(chain.from_iterable(sent_starts)),
        )

    def tokenize(self, text: str) -> Doc:
        tokenized = self.lcc_tokenizer.execute(text)
        sentence_aligned = AlignedSentence(text, tokenized)

        words = sentence_aligned.tokens()
        spaces = [not glue for glue in sentence_aligned.tokens_glue()]
        # add space after sentence end
        spaces += [True]

        return Doc(self.vocab, words=words, spaces=spaces)

    # ----------------------------------------------------
    # mostly based on `spacy/tokenizer.pyx`

    def pipe(self, texts: Iterable[str], batch_size=1000) -> Iterator[Doc]:
        """Tokenize a stream of texts.

        texts: A sequence of unicode texts.
        batch_size (int): Number of texts to accumulate in an internal buffer.
        Defaults to 1000.
        YIELDS (Doc): A sequence of Doc objects, in order.

        DOCS: https://spacy.io/api/tokenizer#pipe
        """
        for text in texts:
            yield self(text)

    def to_disk(self, path, **kwargs):
        """Save the current state to a directory.

        path (str / Path): A path to a directory, which will be created if
            it doesn't exist.
        exclude (list): String names of serialization fields to exclude.

        DOCS: https://spacy.io/api/tokenizer#to_disk
        """
        path = util.ensure_path(path)
        with path.open("wb") as file_:
            file_.write(self.to_bytes(**kwargs))

    def from_disk(self, path, *, exclude=tuple()):
        """Loads state from a directory. Modifies the object in place and
        returns it.

        path (str / Path): A path to a directory.
        exclude (list): String names of serialization fields to exclude.
        RETURNS (Tokenizer): The modified `Tokenizer` object.

        DOCS: https://spacy.io/api/tokenizer#from_disk
        """
        path = util.ensure_path(path)
        with path.open("rb") as file_:
            bytes_data = file_.read()
        self.from_bytes(bytes_data, exclude=exclude)
        return self

    def to_bytes(self, *, exclude=tuple()):
        """Serialize the current state to a binary string.

        exclude (list): String names of serialization fields to exclude.
        RETURNS (bytes): The serialized form of the `Tokenizer` object.

        DOCS: https://spacy.io/api/tokenizer#to_bytes
        """
        serializers = {
            "vocab": lambda: self.vocab.to_bytes(exclude=exclude),
            "dn_tokenizer_resources": lambda: self._dn_tokenizer_resources,
            "dn_segmentizer_resources": lambda: self._dn_segmentizer_resources,
            "lcc_tokenizer": lambda: srsly.pickle_dumps(self.lcc_tokenizer, -1),
            "lcc_segmentizer": lambda: srsly.pickle_dumps(self.lcc_segmentizer, -1),
        }
        return util.to_bytes(serializers, exclude)

    def from_bytes(self, bytes_data, *, exclude=tuple()):
        """Load state from a binary string.

        bytes_data (bytes): The data to load from.
        exclude (list): String names of serialization fields to exclude.
        RETURNS (Tokenizer): The `Tokenizer` object.

        DOCS: https://spacy.io/api/tokenizer#from_bytes
        """
        data = {}
        deserializers = {
            "vocab": lambda b: self.vocab.from_bytes(b, exclude=exclude),
            "dn_tokenizer_resources": lambda b: data.setdefault(
                "dn_tokenizer_resources", b
            ),
            "dn_segmentizer_resources": lambda b: data.setdefault(
                "dn_segmentizer_resources", b
            ),
            "lcc_tokenizer": lambda b: data.setdefault(
                "lcc_tokenizer", srsly.pickle_loads(b)
            ),
            "lcc_segmentizer": lambda b: data.setdefault(
                "lcc_segmentizer", srsly.pickle_loads(b)
            ),
        }

        # reset all properties
        self.lcc_tokenizer = None
        self.lcc_segmentizer = None

        util.from_bytes(bytes_data, deserializers, exclude)

        if "dn_tokenizer_resources" in data and isinstance(
            data["dn_tokenizer_resources"], str
        ):
            self._dn_tokenizer_resources = data["dn_tokenizer_resources"]
        if "dn_segmentizer_resources" in data and isinstance(
            data["dn_segmentizer_resources"], str
        ):
            self._dn_segmentizer_resources = data["dn_segmentizer_resources"]

        if "lcc_tokenizer" in data and isinstance(
            data["lcc_tokenizer"], CharacterBasedWordTokenizerImproved
        ):
            self.lcc_tokenizer = data["lcc_tokenizer"]
        else:
            raise RuntimeError(
                "Missing/invalid 'lcc_tokenizer' field when deserializing ..."
            )

        if "lcc_segmentizer" in data and isinstance(
            data["lcc_segmentizer"], LineAwareSegmentizer
        ):
            self.lcc_segmentizer = data["lcc_segmentizer"]
        return self


# ---------------------------------------------------------------------------
