import os.path
from typing import Optional

import gradio as gr

import lcc.cleaner

# ---------------------------------------------------------------------------


READONLY_PATHS = os.getenv("READONLY_PATHS", "True").lower() == "true"


# ---------------------------------------------------------------------------


with gr.Blocks() as cleaner:
    dn_resources = "resources/cleaner"

    # configuration
    with gr.Accordion("⚙️ Sentence Cleaner Configuration", open=False):
        dn_rules = gr.Textbox(
            value=dn_resources,
            label="Folder with cleaner rule files",
            max_lines=1,
            interactive=not READONLY_PATHS,
            visible=not READONLY_PATHS,
        )
        fn_replacements = gr.Textbox(
            value="StringReplacements.list",
            label="String replacement mapping file (filename in rules folder)",
            max_lines=1,
            interactive=not READONLY_PATHS,
            visible=not READONLY_PATHS,
        )

        with gr.Row():
            text_type = gr.Textbox(
                label="Text type, e.g., 'web', 'news'/'newscrawl', 'wikipedia', ...",
                max_lines=1,
                interactive=True,
            )
            lang_code = gr.Textbox(
                label="Language of text, e.g., 'deu', 'eng', ...",
                max_lines=1,
                interactive=True,
            )

        do_replacements = gr.Checkbox(
            value=True,
            label="Whether to apply text replacements before checking sentence quality",
            interactive=True,
        )

    # inputs
    document_text = gr.Textbox(
        lines=3, label="Text", placeholder="Enter a single sentence"
    )

    # action buttons
    cleaner_btn = gr.Button("Sanitize Sentences", variant="primary")

    # outputs
    with gr.Group():
        status_text = gr.HTML(label="Sentence Quality Status")
        replaced_text = gr.Textbox(
            label="Text with replacements", show_copy_button=True
        )
        filter_details_json = gr.JSON(label="Filter rules that flagged this sentence")

    # worker function
    def clean_text(
        text: str,
        dn_rules: Optional[str],
        text_type: Optional[str] = None,
        lang_code: Optional[str] = None,
        fn_replacements: Optional[str] = None,
        do_replacements: Optional[bool] = True,
    ):
        cwd = os.getcwd()
        if os.path.relpath(os.path.realpath(dn_rules), cwd).startswith("../"):
            raise gr.Error("Invalid 'dn_rules' path!")
        if fn_replacements and os.path.relpath(
            os.path.realpath(os.path.join(dn_rules, fn_replacements)), cwd
        ).startswith("../"):
            raise gr.Error("Invalid 'fn_replacements' filename!")

        if not os.path.isdir(dn_rules):
            gr.Warning("Folder 'dn_rules' not found!")
        if not os.path.isfile(os.path.join(dn_rules, fn_replacements)):
            gr.Warning("File 'fn_replacements' not found!")

        lcc_cleaner = lcc.cleaner.SentenceCleaner(
            dn_rules=dn_rules,
            text_type=text_type,
            lang_code=lang_code,
            fn_replacements=fn_replacements,
        )

        replaced = lcc_cleaner.replacer.replace(text) if do_replacements else None
        results = lcc_cleaner.filter_sentence_results(
            text, do_replacements=do_replacements
        )
        filter_details = [
            {"rule": filter.id_, "description": filter.description, "hit": hit}
            for filter, hit in results.items()
            if hit
        ]

        is_ok = not filter_details

        status = "✅ Sentence is ok." if is_ok else "❎ Sentence is bad!"
        new_status_text = gr.HTML(
            value=f"<p style='margin: 1rem; text-align: center; font-weight: bold;'>{status}</p>"
        )

        return {
            replaced_text: replaced,
            filter_details_json: filter_details if filter_details else None,
            status_text: new_status_text,
        }

    # action buttons event handler
    cleaner_btn.click(
        fn=clean_text,
        inputs=[
            document_text,
            dn_rules,
            text_type,
            lang_code,
            fn_replacements,
            do_replacements,
        ],
        outputs=[replaced_text, filter_details_json, status_text],
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cleaner.launch(show_api=False, share=False)
