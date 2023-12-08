# Leipzig Corpora Collection (LCC) Natural Language Processing (NLP) Tools

The LCC-NLP tools for now only include a **sentence segmentizer**, **sentence cleaner**, **sentence language classifier** and **word tokenizer**. Additionally, it provides some methods to work with SOURCE and MEDUSA file formats that are in use at LCC.
This library can be used embedded or as CLI tool.

A spaCy integration is planned.

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

Example configuration and resources can be found in [`resources/`](resources). The section [_Run CLI_](#run-cli) for how to use them.
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