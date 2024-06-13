# Leipzig Corpora Collection (LCC) Natural Language Processing (NLP) Tools

The LCC-NLP tools for now only include a **sentence segmentizer**, **sentence cleaner**, **sentence language classifier** and **word tokenizer**. Additionally, it provides some methods to work with SOURCE and MEDUSA file formats that are in use at LCC.
This library can be used embedded, with [spaCy](https://spacy.io/) or as CLI tool.

[Installation](#installation) | [Configuration](#configuration-and-resources) | [CLI Usage](#run-cli) | [spaCy Integration](#integration-with-spacy) | [Demos](#demos) | [Development](#development)

Licensed under [_GNU Lesser General Public License v3 or later (LGPLv3+)_](LICENSE).

## Installation

```bash
# local (dev)
python3 -m pip install -e .

# from package
# pip install lcc-nlp
```

Optional WARC support can be enabled with the extra `warc` dependency group:

```bash
# local (dev)
python3 -m pip install -e .[warc]

# from package
# pip install lcc-nlp[warc]
```

TQDM progressbars are also included for parallel processing but they are for now disabled in code. See the `TQDM` constant in `src/lcc/workflow_parallel.py`. If the optional dependency `tqdm` is not installed, then there will simply be no progressbars but it causes no errors.

## Configuration and Resources

Example configuration and resources can be found in [`resources/`](resources). See the section [_Run CLI_](#run-cli) for how to use them.
They are _NOT_ included in the distribution packages but can easily be downloaded from the git repository.

## Run CLI

Show full help:

```bash
# short help
lcc-nlp -h

# longer help, prints all help messages for each sub command (MODE)
lcc-nlp
```

If run on a multi-core processor, multiprocessing will be used to speed up the work. By default all CPUs minus one will be used but the number can be restricted further using `--cpus N`. This effects the modes `tokenize`, `segmentize` and `cleaner`. File conversion etc. is performed sequentially.

### Logging

Logs will be written to `lcc-nlp.log`. Configured in `src/lcc/cli.py` at `LOGFILE` and can be disabled by setting to `None`.

Logging to stdout is enabled with `LOGGING_ECHO` in `src/lcc/cli.py`. Some basic colorization is implemented.

### Sentence Segmentizer

Run interactively in REPL loop for testing:

```bash
lcc-nlp segmentize-loop \
    --pre-boundary-list-file resources/segmentizer/preList.txt \
    --post-boundary-list-file resources/segmentizer/postList.txt \
    --pre-boundary-rules-file resources/segmentizer/preRules.txt \
    --post-boundary-rules-file resources/segmentizer/postRules.txt \
    --boundaries-file resources/segmentizer/boundariesFile.txt
```

Or use input and output files. The format will be detected based on file extension but can also be specified using `--input-format`/`--output-format`.

Additional segmentation parameters can be listed using `lcc-nlp segmentize -h`.

```bash
lcc-nlp segmentize \
    --pre-boundary-list-file resources/segmentizer/preList.txt \
    --post-boundary-list-file resources/segmentizer/postList.txt \
    --pre-boundary-rules-file resources/segmentizer/preRules.txt \
    --post-boundary-rules-file resources/segmentizer/postRules.txt \
    --boundaries-file resources/segmentizer/boundariesFile.txt \
    input.source output.source

# or output.medusa
```

### Sentence Cleaner

Run interactively in REPL loop for testing:

```bash
lcc-nlp cleaner-loop \
    --dn-rules resources/cleaner/
```

Or use input and output files. The format will be detected based on file extension but can also be specified using `--input-format`/`--output-format`.

Additional rule files can be loaded with `--text-type` (`texttype_<TYPE>.rules`) and `--lang-code` (`lang_<LANG>.rules`).
String replacements (for special entities) is configured using `--fn-replacements`.

```bash
lcc-nlp cleaner \
    --dn-rules resources/cleaner/ \
    input.source output.source

lcc-nlp cleaner \
    --dn-rules resources/cleaner/ \
    input.source output.medusa

lcc-nlp cleaner \
    --dn-rules resources/cleaner/ \
    input.medusa output.medusa
```

### Sentence Language Identification

Run interactively in REPL loop for testing:

```bash
lcc-nlp lani-loop \
    --dn-wordlists resources/jlani/wordlists \
    [ --special-chars '[°!=_+-/)(,."&%$§#]?' ] \
    [ --fn-filterlist resources/jlani/blacklist_utf8.txt ] \
    [ --max-words 0 ] \
    [ --reduced ] \
    [ --language eng [ --language deu [ ... ]]]
```

### Word Tokenizer

Run interactively in REPL loop for testing:

```bash
lcc-nlp tokenize-loop \
    --char-actions resources/tokenizer/tokenization_character_actions.txt \
    --fixed-token resources/tokenizer/fixed_tokens.txt \
    --token-chars resources/tokenizer/100-wn-all.txt \
    --abbrev-list resources/tokenizer/default.abbrev
```

Or use input and output files. The format will be detected based on file extension but can also be specified using `--input-format`/`--output-format`.

```bash
lcc-nlp tokenize-loop \
    --char-actions resources/tokenizer/tokenization_character_actions.txt \
    --fixed-token resources/tokenizer/fixed_tokens.txt \
    --token-chars resources/tokenizer/100-wn-all.txt \
    --abbrev-list resources/tokenizer/default.abbrev \
    input.medusa output.medusa
```

### File format conversion

- `convert-source2jsonl`: Convert from SOURCE to JSONL format.
- `convert-medusa2jsonl`: Convert from MEDUSA to JSONL format.
- `convert-warc2jsonl`: Convert from WARC (warc/wet) to JSONL format.
- `convert-source2warc`: Convert from SOURCE to WARC (warc/wet) format.
- `split-source`: Split a single SOURCE file into multiple based on byte and/or document count limits.

```bash
lcc-nlp convert-source2jsonl input.source source.jsonl
lcc-nlp convert-medusa2jsonl input.medusa medusa.jsonl
lcc-nlp convert-warc2jsonl input.warc warc.jsonl
lcc-nlp convert-source2warc input.source source.wet
```

```bash
lcc-nlp split-source \
    --max-documents 1000 \
    --max-bytes 1000000 \
    input.source split.source
# -> split_001.source, split_002.source, ...
```

## Integration with `spaCy`

Note that by default the various integrations and components use various resources (rules, language files, etc.) should be located relative in the working directory at `resources/`. Custom resource paths can be configured using the `config={}` parameter for the pipeline components and with the paths parameters for the tokenizer.

### Tokenizer

```python
>>> import spacy
>>> import lcc.integrations.spacy

>>> # first create your default spaCy NLP model
>>> nlp = spacy.load("en_core_web_sm", disable=["parser", "senter"])
>>> # or choose other models/languages as required
>>> # optionally disabled "parser" and "senter" to use our sentence segmentation annotations in 'Token.is_sent_start'

>>> # create the LCC tokenizer with the NLP Vocab, set path to tokenizer resources
>>> #   and optionally sentence segmentizer resources for sentence start annotations
>>> tokenizer = lcc.integrations.spacy.Tokenizer(nlp.vocab, "resources/tokenizer", "resources/segmentizer")
>>> # substitute the default tokenizer with our own
>>> nlp.tokenizer = tokenizer

>>> # use NLP pipeline like usual
>>> doc = nlp("This is a text.")
>>> list(doc)
[This, is, a, text, .]

>>> # or with multiple sentences
>>> doc = nlp("This is a text. And now another sentence!?")
>>> doc
This is a text. And now another sentence! ?
>>> [(tok.text_with_ws, tok.is_sent_start) for tok in doc]
[('This ', True), ('is ', False), ('a ', False), ('text', False), ('. ', False), ('And ', True), ('now ', False), ('another ', False), ('sentence', False), ('! ', False), ('? ', True)]
```

### Annotations: Sentence Quality / Cleaner

```python
>>> import spacy
>>> import lcc.integrations.spacy
>>> # from lcc.integrations.spacy import SentenceCleanerComponent

>>> # first create your default spaCy NLP model
>>> nlp = spacy.load("en_core_web_sm")

>>> # now add our component
>>> cleaner = nlp.add_pipe("sentencecleaner", config={"show_reason": True})
>>> cleaner
<lcc.integrations.spacy.cleaner.SentenceCleanerComponent object at 0x7ff2e0cb0af0>
>>> # 'show_reason' to add annotations about which filters were triggered

>>> doc = nlp("This is a text. And now another sentence!?")
>>> # this input was filtered/marked due to one rule about some quality criterium
>>> doc._.filtered
True
>>> # Doc._.filter_reasons is only available if the component is configured with 'show_reason=True'!
>>> doc._.filter_reasons
[FilterResult(id=18, description="General - Sätze, die mehrere aufeinanderfolgende '!', '?' besitzen", filtered=True)]

>>> doc = nlp("This is a text.")
>>> doc._.filtered
False
>>> doc._.filter_reasons
[]
>>> # alternatively, access component directly
>>> doc_results = cleaner.lcc_cleaner.filter_sentence_results(doc.text)
>>> any(doc_result.values())  # filtered?
False
```

### Annotations: Language Identification

```python
>>> import spacy
>>> import lcc.integrations.spacy

>>> # first create your default spaCy NLP model
>>> nlp = spacy.load("en_core_web_sm")

>>> # now add our component
>>> lani = nlp.add_pipe("lani", config={})
>>> lani
<lcc.integrations.spacy.lani.LaniComponent object at 0x7fe72e7fdae0>

>>> doc = nlp("This is a short example text in English.")
>>> doc._.language
('eng', 1.0)
>>> doc = nlp("Aber das ist ein Text in German.")
>>> doc._.language
('deu', 1.0)
```

## Demos

### Gradio

Gradio demo applications can be found in [`examples/gradio/`](examples/gradio/). There is also a [`Dockerfile`](examples/gradio/Dockerfile) to allow easy deployment.

## Development

Optional dependencies:

- `test` (pytest, tox)
- `style` (black, isort, flake8, mypy)
- `docs` (sphinx + myst)
- `build` (packaging stuff)

Run code and style checks:

```bash
isort src
black src
flake8 src
mypy src
```

Run tests:

```bash
pytest
```

Run CI tests and checks with:

```bash
tox

# or for a single environment
tox -e check
tox -e check-bugs
```

## Notes

New additions or changes compared to the original Java source code are marked with `# XXX: `. There might still be some minor rewrites (and optimizations) but it is mostly a translation of the Java version. The focus is on the essential classes that are in use at LCC.
