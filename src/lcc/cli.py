import datetime
import logging
import os
import re
import signal
import sys
import time
from argparse import ArgumentDefaultsHelpFormatter
from argparse import ArgumentParser
from typing import Callable
from typing import Dict
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple

# ---------------------------------------------------------------------------


LOGGER = logging.getLogger(__name__)

NAME = "lcc-nlp"

PAT_ANSI = re.compile(r"\x1b\[[^m]+m")

LOGFILE = f"{NAME}.log"
LOGGING_ECHO = True
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------


def format_time(time_diff: float) -> datetime.timedelta:
    time_diff = int(time_diff)
    return datetime.timedelta(seconds=time_diff)


def is_no_color_set() -> bool:
    return "NO_COLOR" in os.environ


def strip_ansi_sequences(val: str) -> str:
    return PAT_ANSI.sub("", val)


def terminate(signal, frame, msg: str, exit_code: int) -> NoReturn:
    # Never (3.11+)
    if exit_code > 0:
        LOGGER.critical(msg)
    else:
        LOGGER.info(msg)
    logging.shutdown()
    sys.exit(exit_code)


# ---------------------------------------------------------------------------


def setup_logger(echo: bool = True, file: Optional[str] = None):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    for handler in logger.handlers:
        logger.removeHandler(handler)

    if echo:
        # some simple colorization and
        class HeaderFormatter(logging.Formatter):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.no_color = is_no_color_set()

            def formatMessage(self, record):
                # # just format the message, keep rest normal
                # color = None
                # if record.levelname == "DEBUG":  # levelno=10
                #     color = "\033[2m"
                # elif record.levelname == "WARNING":  # levelno=30
                #     color = "\033[33m"
                # elif record.levelname == "ERROR":  # levelno=40
                #     color = "\033[31m"
                # if getattr(record, "header", False):
                #     color = "\033[1m" if not color else "\033[1m" + color
                # if color:
                #     record.message = "{color}{m}\033[0m".format(
                #         color=color, m=record.message
                #     )

                m = self._style.format(record)

                if not getattr(record, "header", False):
                    if self.no_color:
                        return m

                    if record.levelname == "DEBUG":  # levelno=10
                        m = "\033[2m{m}\033[0m".format(m=m)
                    elif record.levelname == "WARNING":  # levelno=30
                        m = "\033[33m{m}\033[0m".format(m=m)
                    elif record.levelname == "ERROR":  # levelno=40
                        m = "\033[31m{m}\033[0m".format(m=m)

                    return m

                _message = record.message
                record.message = ""
                m_pre = self._style.format(record)
                record.message = _message

                # ld = 78 - len(m_pre)
                ld = len(m) - len(m_pre)
                msg = "\n\033[1m{m_pre}{m_h}\n{m}\n{m_pre}{m_h}\033[0m".format(
                    m=m, m_pre=m_pre, m_h=("=" * ld)
                )
                if self.no_color:
                    msg = strip_ansi_sequences(msg)
                return msg

        console_formatter = HeaderFormatter(
            ">>> {asctime} [{levelname[0]}] - {message}",
            datefmt=TIMESTAMP_FORMAT,
            style="{",
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    if file:
        file_formatter = logging.Formatter(
            "{asctime} - {levelname} - {message}",
            datefmt=TIMESTAMP_FORMAT,
            style="{",
        )
        file_handler = logging.FileHandler(file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # quiet specific loggers
    logging.getLogger("lcc.tokenizer").setLevel(logging.INFO)
    # vendor
    # logging.getLogger("filelock").setLevel(logging.INFO)


# ---------------------------------------------------------------------------


def interactive_shell():
    import code
    import sys

    banner = "Python {} on {}\nContext: -\n" "Ctrl+D to quit"
    banner = banner.format(sys.version, sys.platform)
    ctx = dict()
    ctx.update(locals())

    # Try to enable tab completion
    try:
        # readline module is only available on unix systems
        import readline
    except ImportError:
        pass
    else:
        import rlcompleter

        readline.set_completer(rlcompleter.Completer(ctx).complete)
        readline.parse_and_bind("tab: complete")

    code.interact(banner=banner, local=ctx)


# ---------------------------------------------------------------------------


def make_cliparser() -> Tuple[ArgumentParser, Dict[str, ArgumentParser]]:
    import lcc.workflow
    from lcc.io import FileFormats
    from lcc.io import _extract_file_formats

    parser = ArgumentParser(
        prog=NAME,
        description="Wortschatz NLP Tools",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(title="Mode", dest="mode", metavar="MODE")

    # ------------------------------------
    # - shared options

    # - general
    shared_general_args = ArgumentParser(add_help=False)
    # encoding
    shared_general_args.add_argument(
        "--encoding",
        dest="encoding",
        required=False,
        default="utf-8",
        help="Specifies the encoding used in input/output.",
    )

    # - parallel processing
    shared_parallel_args = ArgumentParser(add_help=False)
    # cpus
    n_cpus = (os.cpu_count() or 1) - 1
    shared_parallel_args.add_argument(
        "--cpus",
        dest="n_cpus",
        required=False,
        type=int,
        default=n_cpus,
        help="Number of CPUs to use for parallel processing."
        f" If None then use all CPUs ({n_cpus})."
        " If 0 or negative, then disabled, so normal sequential processing.",
    )

    # - tokenizer

    shared_tokenize_config_args = ArgumentParser(add_help=False)
    # strAbbrevListFile
    shared_tokenize_config_args.add_argument(
        "--abbrev-list",
        dest="abbrev_list_file",
        required=False,  # TODO: should this be required?
        help="Abbreviations list file.",
    )
    # TOKENISATION_CHARACTERS_FILE_NAME
    shared_tokenize_config_args.add_argument(
        "--token-chars",
        dest="token_chars_file",
        required=False,
        help="Tokenization characters file.",
    )
    # fixedTokensFile
    shared_tokenize_config_args.add_argument(
        "--fixed-tokens",
        dest="fixed_tokens_file",
        required=False,
        help="Fixed tokens file.",
    )
    # strCharacterActionsFile
    shared_tokenize_config_args.add_argument(
        "--char-actions",
        dest="char_actions_file",
        required=False,
        help="Character actions file.",
    )

    # - cleaner

    shared_cleaner_config_args = ArgumentParser(add_help=False)
    # dn_rules
    shared_cleaner_config_args.add_argument(
        "--dn-rules",
        dest="dn_rules",
        required=True,
        help="Folder containing *.rules files. (and StringReplacements.list)",
    )
    shared_cleaner_config_args.add_argument(
        "--text-type",
        dest="text_type",
        required=False,
        default=None,
        help="Text type.",
    )
    shared_cleaner_config_args.add_argument(
        "--lang-code",
        dest="lang_code",
        required=False,
        default=None,
        help="ISO639 language code.",
    )
    shared_cleaner_config_args.add_argument(
        "--fn-replacements",
        dest="fn_replacements",
        required=False,
        default=None,
        help="Filename for replacements, relative to --dn-rules folder. Enables replacements.",
    )

    # - lani - language identification

    shared_lani_config_args = ArgumentParser(add_help=False)
    # init args
    shared_lani_config_args.add_argument(
        "--dn-wordlists",
        dest="dn_wordlists",
        required=True,
        help="Folder containing language wordlist files.",
    )
    shared_lani_config_args.add_argument(
        "--special-chars",
        dest="special_chars",
        required=False,
        default=None,
        help="Regular expression for characters that will be deleted from the input.",
    )
    shared_lani_config_args.add_argument(
        "--fn-filterlist",
        dest="fn_filterlist",
        required=False,
        default=None,
        help="File with words that will be filtered out before language identification.",
    )
    # runtime args
    shared_lani_config_args.add_argument(
        "--max-words",
        dest="max_words",
        required=False,
        default=0,
        help="Number of words to check at most for language identification."
        " Will intelligently skip in input.",
    )
    shared_lani_config_args.add_argument(
        "--reduced",
        dest="reduced",
        required=False,
        action="store_true",
        default=False,
        help="Compute only reduced set of information (no coverage/wordcount stats).",
    )
    shared_lani_config_args.add_argument(
        "--language",
        dest="languages",
        required=False,
        action="append",
        help="List of languages to check against."
        " NOTE: if only one language is provided, it is assumed this is the target language"
        " and similar languages for it will be chosen to compare against.",
    )

    # - segmentizer

    shared_segmentize_config_args = ArgumentParser(add_help=False)
    # boundariesFile.txt
    shared_segmentize_config_args.add_argument(
        "--boundaries-file",
        dest="boundaries_file",
        required=False,
        help="The file specifying the sentence boundary candidates to be used. If not specified '.?!' are used.",
    )
    # preList.txt
    shared_segmentize_config_args.add_argument(
        "--pre-boundary-list-file",
        dest="pre_boundary_list_file",
        required=True,
        help="The file containing those tokens that should never be present directly in front of a sentence boundary"
        " (includes the sentence boundary chars; example: abbreviations like 'Dr.').",
    )
    # postList.txt
    shared_segmentize_config_args.add_argument(
        "--post-boundary-list-file",
        dest="post_boundary_list_file",
        required=True,
        help="The file containing those tokens that should never be present directly after a sentence boundary.",
    )
    # preRules.txt
    shared_segmentize_config_args.add_argument(
        "--pre-boundary-rules-file",
        dest="pre_boundary_rules_file",
        required=True,
        help="The file containing the patterns that should/should not be recognized directly in front of a sentence boundary.",
    )
    # postRules.txt (cp1250)
    shared_segmentize_config_args.add_argument(
        "--post-boundary-rules-file",
        dest="post_boundary_rules_file",
        required=True,
        help="The file containing the patterns that should/should not be recognized directly after a sentence boundary.",
    )
    # encoding in general-args
    # flags
    shared_segmentize_config_args.add_argument(
        "--no-uppercase-pre-list-first-letter",
        dest="uppercase_pre_list_first_letter",
        required=False,
        action="store_false",
        default=True,
        help="Also use an additional version of each entry in preList with first letter in uppercase.",
    )
    shared_segmentize_config_args.add_argument(
        "--no-carriage-return",
        dest="carriage_return",
        required=False,
        action="store_false",
        default=True,
        help="Interpret carriage returns as sentences boundary.",
    )
    shared_segmentize_config_args.add_argument(
        "--empty-line",
        dest="empty_line",
        required=False,
        action="store_false",
        default=True,
        help="Interpret empty lines as sentence boundary.",
    )
    shared_segmentize_config_args.add_argument(
        "--trim",
        dest="trim",
        required=False,
        action="store_false",
        default=True,
        help="Trims all sentences (remove whitespaces at begin/end) before writing to the output.",
    )

    # ------------------------------------
    # tool usage

    def _add_input_output_and_format_params(
        parser: ArgumentParser,
        function: Callable,
        param_name_input: str = "fmt_input",
        param_name_output: str = "fmt_output",
    ):
        parser.add_argument(
            "input_file",
            metavar="input",
            type=str,
            help="Input text file."
            " Format will be detected by extension if not specified using '--input-format'.",
        )
        parser.add_argument(
            "output_file",
            metavar="output",
            type=str,
            help="Output text file."
            " Format will be detected by extension if not specified using '--output-format'.",
        )

        fmt_input_choices = _extract_file_formats(function, param_name_input)
        fmt_output_choices = _extract_file_formats(function, param_name_output)
        parser.add_argument(
            "--input-format",
            dest="fmt_input",
            default=None,
            type=FileFormats,
            choices=fmt_input_choices,
            help="Input text file format."
            " If not specified, will try to detect from 'input' argument file extension.",
        )
        parser.add_argument(
            "--output-format",
            dest="fmt_output",
            default=None,
            type=FileFormats,
            choices=fmt_output_choices,
            help="Output text file format."
            " If not specified, will try to detect from 'output' argument file extension.",
        )

    parser_tokenize = subparsers.add_parser(
        "tokenize",
        help="Tokenize sentences.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        parents=[
            shared_tokenize_config_args,
            shared_general_args,
            shared_parallel_args,
        ],
    )
    _add_input_output_and_format_params(parser_tokenize, lcc.workflow.tokenize_sentence)

    parser_tokenize_loop = subparsers.add_parser(
        "tokenize-loop",
        help="Tokenize sentences in interactive input loop.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        parents=[shared_tokenize_config_args],
    )

    parser_cleaner = subparsers.add_parser(
        "cleaner",
        help="Clean sentences.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        parents=[shared_cleaner_config_args, shared_general_args, shared_parallel_args],
    )
    _add_input_output_and_format_params(parser_cleaner, lcc.workflow.clean_sentences)

    parser_cleaner_loop = subparsers.add_parser(
        "cleaner-loop",
        help="Clean sentences in interactive input loop.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        parents=[shared_cleaner_config_args],
    )

    parser_lani = subparsers.add_parser(
        "lani",
        help="Sentence language identification.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        parents=[shared_lani_config_args, shared_general_args, shared_parallel_args],
    )
    # _add_input_output_and_format_params(parser_lani, lcc.workflow.lani_sentences)

    parser_lani_loop = subparsers.add_parser(
        "lani-loop",
        help="Sentence language identification in interactive input loop.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        parents=[shared_lani_config_args],
    )

    parser_segmentize = subparsers.add_parser(
        "segmentize",
        help="Segmentize text into sentences.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        parents=[
            shared_segmentize_config_args,
            shared_general_args,
            shared_parallel_args,
        ],
    )
    _add_input_output_and_format_params(
        parser_segmentize, lcc.workflow.sentence_segment
    )

    parser_segmentize_loop = subparsers.add_parser(
        "segmentize-loop",
        help="Segmentize text into sentences in interactive input loop.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        parents=[shared_segmentize_config_args, shared_general_args],
    )

    # ------------------------------------
    # format converter

    parser_convert_source2jsonl = subparsers.add_parser(
        "convert-source2jsonl",
        help="Convert from SOURCE to JSONL format.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        parents=[shared_general_args],
    )
    parser_convert_source2jsonl.add_argument(
        "input_file", metavar="input", help="Input SOURCE file."
    )
    parser_convert_source2jsonl.add_argument(
        "output_file", metavar="output", help="Output JSONL file."
    )

    parser_convert_medusa2jsonl = subparsers.add_parser(
        "convert-medusa2jsonl",
        help="Convert from MEDUSA to JSONL format.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        parents=[shared_general_args],
    )
    parser_convert_medusa2jsonl.add_argument(
        "input_file", metavar="input", help="Input MEDUSA file."
    )
    parser_convert_medusa2jsonl.add_argument(
        "output_file", metavar="output", help="Output JSONL file."
    )

    parser_convert_warc2jsonl = subparsers.add_parser(
        "convert-warc2jsonl",
        help="Convert from WARC (warc/wet) to JSONL format.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        parents=[shared_general_args],
    )
    parser_convert_warc2jsonl.add_argument(
        "input_file", metavar="input", help="Input WARC file."
    )
    parser_convert_warc2jsonl.add_argument(
        "output_file", metavar="output", help="Output JSONL file."
    )

    parser_convert_source2warc = subparsers.add_parser(
        "convert-source2warc",
        help="Convert from SOURCE to WARC (warc/wet) format.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        parents=[shared_general_args],
    )
    parser_convert_source2warc.add_argument(
        "input_file", metavar="input", help="Input SOURCE file."
    )
    parser_convert_source2warc.add_argument(
        "output_file", metavar="output", help="Output WARC (warc/wet) file."
    )

    parser_split_source = subparsers.add_parser(
        "split-source",
        help="Split a single SOURCE file into multiple based on byte and/or document count limits.",
        epilog="NOTE: Both byte and document count limits can be combined."
        " At least one document will always be in a file."
        " No limits means no splitting, so nothing will be done.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        parents=[shared_general_args],
    )
    parser_split_source.add_argument(
        "input_file", metavar="input", help="Input SOURCE file."
    )
    parser_split_source.add_argument(
        "output_file",
        metavar="output",
        help="Output SOURCE file base name. A counter will be added.",
    )
    parser_split_source.add_argument(
        "--max-bytes",
        dest="maxbytes",
        type=int,
        default=None,
        help="Maximum bytes per file."
        " NOTE that if a single document exceeds this limit it will still be written to its own file."
        " So for multiple documents this is the upper bound but a single document may exceed it!",
    )
    parser_split_source.add_argument(
        "--max-documents",
        dest="maxdocs",
        type=int,
        default=None,
        help="Total number of documents per file.",
    )

    parser_slice_source = subparsers.add_parser(
        "slice-source",
        help="Extract a slice of a SOURCE file.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        parents=[shared_general_args],
    )
    parser_slice_source.add_argument(
        "input_file", metavar="input", help="Input SOURCE file."
    )
    parser_slice_source.add_argument(
        "output_file",
        metavar="output",
        help="Output SOURCE file.",
    )
    parser_slice_source.add_argument(
        "--start",
        dest="start",
        type=int,
        default=None,
        help="Start index (starts at 1) for the first document. If not set then start from the beginning.",
    )
    parser_slice_source.add_argument(
        "--stop",
        dest="stop",
        type=int,
        default=None,
        help="Stop index (starts at 1, inclusive) for the last document. If not set then until end.",
    )

    # NOTE: merge-source? can also be done using just 'cat'

    # ------------------------------------
    # debugging, interactive console

    parser_shell = subparsers.add_parser(  # noqa: F841
        "shell",
        help=(
            "Starts an interactive Code Console (REPL); " "Ctrl+D or SIGTERM to quit."
        ),
    )

    # ------------------------------------
    # - all subparser for help msg dump

    # NOTE: map exists in `subparsers._name_parser_map` = { name => parser}

    lookup_subparsers = dict()
    lookup_subparsers["tokenize"] = parser_tokenize
    lookup_subparsers["tokenize-loop"] = parser_tokenize_loop
    lookup_subparsers["cleaner"] = parser_cleaner
    lookup_subparsers["cleaner-loop"] = parser_cleaner_loop
    lookup_subparsers["lani"] = parser_lani
    lookup_subparsers["lani-loop"] = parser_lani_loop
    lookup_subparsers["segmentize"] = parser_segmentize
    lookup_subparsers["segmentize-loop"] = parser_segmentize_loop
    lookup_subparsers["convert-source2jsonl"] = parser_convert_source2jsonl
    lookup_subparsers["convert-medusa2jsonl"] = parser_convert_medusa2jsonl
    lookup_subparsers["convert-warc2jsonl"] = parser_convert_warc2jsonl
    lookup_subparsers["convert-source2warc"] = parser_convert_source2warc
    lookup_subparsers["split-source"] = parser_split_source
    lookup_subparsers["slice-source"] = parser_slice_source
    lookup_subparsers["shell"] = parser_shell

    # ------------------------------------

    return parser, lookup_subparsers


# ---------------------------------------------------------------------------


def main(args: Optional[Sequence[str]] = None):
    parser, lookup_subparsers = make_cliparser()
    args_parsed = parser.parse_args(args=args)

    if args_parsed.mode is None:
        parser.print_help()
        for _, subparser in lookup_subparsers.items():
            print("-" * 78)
            subparser.print_help()
        parser.exit(message="Please specify a MODE!\n")

    # ------------------------------------

    LOGGER.info(
        f"Started {NAME} in '%s' with: %s", args_parsed.mode, " ".join(sys.argv[2:])
    )
    time_start = time.time()

    if args_parsed.mode in ("tokenize", "tokenize-loop"):
        import lcc.tokenizer

        tokenizer = lcc.tokenizer.CharacterBasedWordTokenizerImproved(
            strAbbrevListFile=args_parsed.abbrev_list_file,
            TOKENISATION_CHARACTERS_FILE_NAME=args_parsed.token_chars_file,
            fixedTokensFile=args_parsed.fixed_tokens_file,
            strCharacterActionsFile=args_parsed.char_actions_file,
        )

        if args_parsed.mode == "tokenize-loop":
            try:
                print("Exit with Ctrl+D.")
                while True:
                    line = input(" Input: ")
                    if not line:
                        continue
                    try:
                        line_tokenized = tokenizer.execute(line)
                    except Exception as ex:
                        print(f" Error: {ex}")
                        continue
                    print(f"Output: {line_tokenized}\n")
            except EOFError:
                print()

        elif args_parsed.mode == "tokenize":
            import lcc.workflow_parallel

            lcc.workflow_parallel.tokenize_sentence_parallel(
                args_parsed.input_file,
                args_parsed.output_file,
                fmt_input=args_parsed.fmt_input,
                fmt_output=args_parsed.fmt_output,
                tokenizer=tokenizer,
                n_cpus=args_parsed.n_cpus,
            )

    elif args_parsed.mode in ("cleaner", "cleaner-loop"):
        import lcc.cleaner

        cleaner = lcc.cleaner.SentenceCleaner(
            args_parsed.dn_rules,
            text_type=args_parsed.text_type,
            lang_code=args_parsed.lang_code,
            fn_replacements=args_parsed.fn_replacements,
        )
        do_replacements = args_parsed.fn_replacements is not None

        if args_parsed.mode == "cleaner-loop":
            try:
                print("Exit with Ctrl+D.")
                while True:
                    line = input(" Input: ")
                    if not line:
                        continue
                    try:
                        line_cleaned = cleaner.filter_sentence(
                            line, do_replacements=do_replacements
                        )
                    except Exception as ex:
                        print(f" Error: {ex}")
                        continue
                    if line_cleaned is None:
                        print("Filtered.")
                    else:
                        print(f"Output: {line_cleaned}\n")
            except EOFError:
                print()

        elif args_parsed.mode == "cleaner":
            import lcc.workflow_parallel

            lcc.workflow_parallel.clean_sentences_parallel(
                args_parsed.input_file,
                args_parsed.output_file,
                cleaner=cleaner,
                fmt_input=args_parsed.fmt_input,
                fmt_output=args_parsed.fmt_output,
                do_replacements=do_replacements,
                n_cpus=args_parsed.n_cpus,
            )

    elif args_parsed.mode in ("lani", "lani-loop"):
        import lcc.language.sentence

        lani = lcc.language.sentence.LanIKernel(
            args_parsed.dn_wordlists,
            specialChars=args_parsed.special_chars,
            fn_filterlist=args_parsed.fn_filterlist,
        )
        fn_lang_dist = os.path.join(args_parsed.dn_wordlists, "..", "lang_dist.tsv")

        languages = args_parsed.languages
        if languages and len(languages) == 1:
            LOGGER.debug("Main language to check for is '%s' ...", languages[0])
            # find similar languages + defaults
            languages = lcc.language.sentence.get_languages_to_check_for(
                languages[0], fn_lang_dist
            )
            LOGGER.debug("Selected languages to check against: %s", languages)

        if languages:
            # but trim back to only known languages
            languages = lani.datasourcemngr.languages & set(languages)
            LOGGER.debug("Keeping known languages to check against: %s", languages)

        if args_parsed.mode == "lani-loop":
            try:
                print("Exit with Ctrl+D.")
                while True:
                    line = input(" Input: ")
                    if not line:
                        continue
                    try:
                        result = lani.evaluate(
                            line,
                            languages=languages,
                            reduced=args_parsed.reduced,
                            num_words_to_check=args_parsed.max_words,
                        )
                        lang_result = result.get_result()
                    except Exception as ex:
                        print(f" Error: {ex}")
                        continue
                    LOGGER.debug("result: %s", lang_result)
                    if lang_result.is_known():
                        print(
                            f"Result: Detected language is '{lang_result.language}' "
                            f"(prob={lang_result.probability}%, tokens={lang_result.wordcount},"
                            f" coverage={round(lang_result.coverage*100):d}%).\n"
                        )
                    else:
                        print(
                            f"Result: No language could be detected with high confidence!"
                        )
                        LOGGER.debug("details: %s", result)
            except EOFError:
                print()

        elif args_parsed.mode == "lani":
            raise NotImplementedError("Lani workflow not implemented yet!")

    elif args_parsed.mode in ("segmentize", "segmentize-loop"):
        import lcc.segmentizer

        segmentizer = lcc.segmentizer.LineAwareSegmentizer(
            fn_sentence_boundaries=args_parsed.boundaries_file,
            fn_pre_boundary_rules=args_parsed.pre_boundary_rules_file,
            fn_pre_boundaries_list=args_parsed.pre_boundary_list_file,
            fn_post_boundary_rules=args_parsed.post_boundary_rules_file,
            fn_post_boundaries_list=args_parsed.post_boundary_list_file,
            encoding=args_parsed.encoding,
            is_auto_uppercase_first_letter_pre_list=args_parsed.uppercase_pre_list_first_letter,
            use_carriage_return_as_boundary=args_parsed.carriage_return,
            use_empty_line_as_boundary=args_parsed.empty_line,
            is_trim_mode=args_parsed.trim,
        )

        if args_parsed.mode == "segmentize-loop":
            try:
                while True:
                    texts = []
                    try:
                        print("Finish current input with Ctrl+D, exit with Ctrl+C.")
                        while True:
                            text = input(" Text: ")
                            texts.append(text)
                    except EOFError:
                        print()
                    try:
                        sentences = segmentizer.segmentize("\n".join(texts))
                    except Exception as ex:
                        print(f"Error: {ex}")
                        continue
                    print("       SNO [   HASH   ] SENTENCE")
                    for sno, sentence in enumerate(sentences, 1):
                        print(
                            f" Sent: {sno:03} [{lcc.segmentizer.str_hash(sentence):>10}] {sentence}"
                        )
            except KeyboardInterrupt:
                print()

        elif args_parsed.mode == "segmentize":
            import lcc.workflow_parallel

            lcc.workflow_parallel.sentence_segment_parallel(
                args_parsed.input_file,
                args_parsed.output_file,
                fmt_input=args_parsed.fmt_input,
                fmt_output=args_parsed.fmt_output,
                segmentizer=segmentizer,
                n_cpus=args_parsed.n_cpus,
            )

    elif args_parsed.mode == "convert-source2jsonl":
        import lcc.workflow

        lcc.workflow.convert_source_to_jsonl(
            args_parsed.input_file,
            args_parsed.output_file,
            encoding=args_parsed.encoding,
        )

    elif args_parsed.mode == "convert-medusa2jsonl":
        import lcc.workflow

        lcc.workflow.convert_medusa_to_jsonl(
            args_parsed.input_file,
            args_parsed.output_file,
            encoding=args_parsed.encoding,
        )

    elif args_parsed.mode == "convert-warc2jsonl":
        import lcc.workflow

        lcc.workflow.convert_warc_to_jsonl(
            args_parsed.input_file,
            args_parsed.output_file,
            encoding=args_parsed.encoding,
        )

    elif args_parsed.mode == "convert-source2warc":
        import lcc.workflow

        lcc.workflow.convert_source_to_warc(
            args_parsed.input_file,
            args_parsed.output_file,
            encoding=args_parsed.encoding,
        )

    elif args_parsed.mode == "split-source":
        import lcc.workflow

        lcc.workflow.split_source_file(
            args_parsed.input_file,
            args_parsed.output_file,
            maxdocs=args_parsed.maxdocs,
            maxbytes=args_parsed.maxbytes,
            encoding=args_parsed.encoding,
        )

    elif args_parsed.mode == "slice-source":
        import lcc.workflow

        lcc.workflow.slice_source_file(
            args_parsed.input_file,
            args_parsed.output_file,
            start=args_parsed.start,
            stop=args_parsed.stop,
            encoding=args_parsed.encoding,
        )

    elif args_parsed.mode == "shell":
        interactive_shell()

    time_diff = time.time() - time_start
    LOGGER.info("Done. (time: {})".format(format_time(time_diff)))


# ---------------------------------------------------------------------------


def cli_main(args: Optional[Sequence[str]] = None):
    try:
        setup_logger(echo=LOGGING_ECHO, file=LOGFILE)
        signal.signal(
            signal.SIGTERM, lambda s, f: terminate(s, f, "Halted (SIGTERM).", 0)
        )
        main(args=args)
        logging.shutdown()
    except KeyboardInterrupt:
        terminate(None, None, "Halted (KeyboardInterrupt).", 0)
    except Exception as ex:
        LOGGER.exception(ex)
        terminate(None, None, "Terminating because of an exception.", 1)


# ---------------------------------------------------------------------------
