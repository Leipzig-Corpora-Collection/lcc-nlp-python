import os.path
from typing import List
from typing import Optional

import gradio as gr

import lcc.language.sentence

# ---------------------------------------------------------------------------


READONLY_PATHS = os.getenv("READONLY_PATHS", "True").lower() == "true"


# ---------------------------------------------------------------------------


with gr.Blocks() as lani:
    dn_resources = "resources/jlani"

    # config
    with gr.Accordion("⚙️ LanI Configuration", open=False):
        with gr.Accordion(
            "General LanI Kernel Configuration", visible=not READONLY_PATHS
        ):
            dn_wordlists = gr.Textbox(
                value=os.path.join(dn_resources, "wordlists/plain"),
                label="Folder with wordlist files",
                max_lines=1,
                interactive=not READONLY_PATHS,
                visible=not READONLY_PATHS,
            )
            fn_filterlist = gr.Textbox(
                value=os.path.join(dn_resources, "blacklist_utf8.txt"),
                label="File with word ignore list",
                max_lines=1,
                interactive=not READONLY_PATHS,
                visible=not READONLY_PATHS,
            )
            specialChars = gr.Textbox(
                value='[°!=_+-/)(,."&%$§#]?',
                label="Preprocessing: delete all matches",
                max_lines=1,
                interactive=not READONLY_PATHS,
            )

        num_words_to_check = gr.Number(
            value=0,
            label="Number of words to check ('0' for all)",
            minimum=0,
            interactive=True,
        )

    # config: language selection
    with gr.Accordion("Language Candidate Selection", open=True):

        def update_language_dropdown(dn_wordlists: str):
            from lcc.language.sentence import EXT_WORDS

            languages_lst = []

            cwd = os.getcwd()
            if os.path.relpath(os.path.realpath(dn_wordlists), cwd).startswith("../"):
                raise gr.Error("Invalid 'dn_wordlists' path!")
            elif os.path.isdir(dn_wordlists):
                for wl_file in os.listdir(dn_wordlists):
                    if wl_file.endswith(EXT_WORDS) and len(wl_file) > len(EXT_WORDS):
                        lang = wl_file[: -len(EXT_WORDS)]
                        languages_lst.append(lang)

            languages = gr.Dropdown(
                choices=languages_lst,
                label="Languages to check against",
                multiselect=True,
                interactive=True,
            )

            return languages

        # trigger initial load
        languages = update_language_dropdown(dn_wordlists.value)
        # and check on path change
        dn_wordlists.change(
            fn=update_language_dropdown, inputs=dn_wordlists, outputs=languages
        )

        gr.Markdown(
            "If only a single language has been selected, then the language"
            " set can be expanded to include similar languages and some base"
            " languages for the language identification process."
        )

        fn_lang_dist = gr.Textbox(
            value=os.path.join(dn_resources, "wordlists/lang_dist.tsv"),
            label="File list language similarity",
            max_lines=1,
            interactive=not READONLY_PATHS,
            visible=not READONLY_PATHS,
        )

        filter_most_likely_chk = gr.Checkbox(
            value=False,
            label="Expand single language to likely language set",
            interactive=True,
        )
        min_similarity_nbr = gr.Number(
            value=lcc.language.sentence.LANGUAGE_MIN_SIMILARITY_THRESHOLD,
            label="Threshold for language similarity",
            minimum=0.0,
            maximum=1.0,
            step=0.1,
            interactive=True,
        )

        # selected languages preview
        preview_languages_expanded_dataset = gr.Dataset(
            components=[gr.Textbox(visible=False)],
            label="Expanded set of languages for language identification",
            samples=[],
        )

        # update function
        def preview_language_selection(
            languages: List[str],
            fn_lang_dist: str,
            filter_most_likely_chk: bool,
            min_similarity: float,
        ):
            cwd = os.getcwd()
            if os.path.relpath(os.path.realpath(fn_lang_dist), cwd).startswith("../"):
                raise gr.Error("Invalid 'fn_lang_dist' path!")
            if not os.path.isfile(fn_lang_dist):
                gr.Warning("File 'fn_lang_dist' not found!")

            if not languages:
                return []

            languages = set(languages)
            if filter_most_likely_chk and len(languages) == 1:
                languages |= lcc.language.sentence.get_languages_to_check_for(
                    list(languages)[0], fn_lang_dist, min_similarity=min_similarity
                )

            return [[language, "a"] for language in sorted(languages)]

        # change listener
        fn_lang_dist.change(
            fn=preview_language_selection,
            inputs=[
                languages,
                fn_lang_dist,
                filter_most_likely_chk,
                min_similarity_nbr,
            ],
            outputs=preview_languages_expanded_dataset,
        )
        min_similarity_nbr.change(
            fn=preview_language_selection,
            inputs=[
                languages,
                fn_lang_dist,
                filter_most_likely_chk,
                min_similarity_nbr,
            ],
            outputs=preview_languages_expanded_dataset,
        )
        filter_most_likely_chk.change(
            fn=preview_language_selection,
            inputs=[
                languages,
                fn_lang_dist,
                filter_most_likely_chk,
                min_similarity_nbr,
            ],
            outputs=preview_languages_expanded_dataset,
        )
        languages.change(
            fn=preview_language_selection,
            inputs=[
                languages,
                fn_lang_dist,
                filter_most_likely_chk,
                min_similarity_nbr,
            ],
            outputs=preview_languages_expanded_dataset,
        )

    # input
    document_text = gr.Textbox(
        lines=3, label="Text", placeholder="Enter you text here..."
    )

    # action buttons
    lani_btn = gr.Button("Identify Language", variant="primary")

    # output
    with gr.Row():
        language_text = gr.Textbox(label="Language")
        language_prob_number = gr.Number(label="Probability")

    with gr.Accordion("Result Details", open=False):
        result_candidate_json = gr.JSON(label="Best language candidate details")
        result_json = gr.JSON(label="Full LanI result")

    # worker function
    def detect_language(
        text: str,
        dn_wordlists: Optional[str] = None,
        fn_filterlist: Optional[str] = None,
        specialChars: Optional[str] = None,
        fn_lang_dist: Optional[str] = None,
        languages: Optional[List[str]] = None,
        filter_most_likely_chk: Optional[bool] = None,
        min_similarity: Optional[
            float
        ] = lcc.language.sentence.LANGUAGE_MIN_SIMILARITY_THRESHOLD,
        num_words_to_check: Optional[int] = None,
    ) -> str:
        cwd = os.getcwd()
        if os.path.relpath(os.path.realpath(dn_wordlists), cwd).startswith("../"):
            raise gr.Error("Invalid 'dn_wordlists' path!")
        if os.path.relpath(os.path.realpath(fn_filterlist), cwd).startswith("../"):
            raise gr.Error("Invalid 'fn_filterlist' path!")
        if os.path.relpath(os.path.realpath(fn_lang_dist), cwd).startswith("../"):
            raise gr.Error("Invalid 'fn_lang_dist' path!")

        if not os.path.isdir(dn_wordlists):
            gr.Warning("Folder 'dn_wordlists' not found!")
        if not os.path.isfile(fn_filterlist):
            gr.Warning("File 'fn_filterlist' not found!")
        if not os.path.isfile(fn_lang_dist):
            gr.Warning("File 'fn_lang_dist' not found!")

        if languages and len(set(languages)) == 1 and not filter_most_likely_chk:
            raise gr.Error(
                "Only a single language was selected. "
                "Then the result is clear, no need to check!"
            )

        lcc_lani = lcc.language.sentence.LanIKernel(
            dn_wordlists=dn_wordlists,
            specialChars=specialChars,
            fn_filterlist=fn_filterlist,
        )

        if filter_most_likely_chk and languages and len(set(languages)) == 1:
            languages = set(
                languages
            ) | lcc.language.sentence.get_languages_to_check_for(
                list(languages)[0], fn_lang_dist, min_similarity=min_similarity
            )

            languages = languages & lcc_lani.datasourcemngr.languages

        if not num_words_to_check:
            num_words_to_check = 0

        result = lcc_lani.evaluate(
            text, languages=languages, num_words_to_check=num_words_to_check
        )
        if not result:
            raise gr.Error("No LanI result?!")

        lang_result = result.get_result()
        if not lang_result.is_known():
            gr.Warning("No primary language could be detected!")

        return {
            result_json: result,
            result_candidate_json: lang_result,
            language_text: lang_result.language if lang_result.is_known() else None,
            language_prob_number: lang_result.probability
            if lang_result.is_known()
            else None,
        }

    # action buttons event handler
    lani_btn.click(
        fn=detect_language,
        inputs=[
            document_text,
            dn_wordlists,
            fn_filterlist,
            specialChars,
            fn_lang_dist,
            languages,
            filter_most_likely_chk,
            min_similarity_nbr,
            num_words_to_check,
        ],
        outputs=[
            result_json,
            result_candidate_json,
            language_text,
            language_prob_number,
        ],
        api_name="lani",
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lani.launch(show_api=False, share=False)
