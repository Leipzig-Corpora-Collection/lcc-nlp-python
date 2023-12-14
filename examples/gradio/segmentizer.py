import os
import os.path
from typing import Optional

import gradio as gr

import lcc.segmentizer

# ---------------------------------------------------------------------------


READONLY_PATHS = os.getenv("READONLY_PATHS", "True").lower() == "true"


# ---------------------------------------------------------------------------


with gr.Blocks() as segmentizer:
    dn_resources = "resources/segmentizer"

    # configuration
    with gr.Accordion("⚙️ Segmentizer Configuration", open=False):
        # files
        with gr.Accordion("Rule Files", visible=not READONLY_PATHS):
            fn_sentence_boundaries = gr.Textbox(
                value=os.path.join(dn_resources, "boundariesFile.txt"),
                label="Sentence boundaries file",
                max_lines=1,
                interactive=not READONLY_PATHS,
                visible=not READONLY_PATHS,
            )

            with gr.Row():
                fn_pre_boundary_rules = gr.Textbox(
                    value=os.path.join(dn_resources, "preRules.txt"),
                    label="Pre-boundary rules",
                    max_lines=1,
                    interactive=not READONLY_PATHS,
                    visible=not READONLY_PATHS,
                )
                fn_post_boundary_rules = gr.Textbox(
                    value=os.path.join(dn_resources, "postRules.txt"),
                    label="Post-boundary rules",
                    max_lines=1,
                    interactive=not READONLY_PATHS,
                    visible=not READONLY_PATHS,
                )

            with gr.Row():
                fn_pre_boundaries_list = gr.Textbox(
                    value=os.path.join(dn_resources, "preList.txt"),
                    label="Pre-boundary abbreviation list",
                    interactive=not READONLY_PATHS,
                    visible=not READONLY_PATHS,
                )
                fn_post_boundaries_list = gr.Textbox(
                    value=os.path.join(dn_resources, "postList.txt"),
                    label="Post-boundary token list",
                    max_lines=1,
                    interactive=not READONLY_PATHS,
                    visible=not READONLY_PATHS,
                )

        # boolean toggles
        with gr.Accordion("Other Input/Output Options"):
            is_auto_uppercase_first_letter_pre_list = gr.Checkbox(
                value=True,
                label="Auto add pre-abbreviation forms with upper first letter",
                interactive=not READONLY_PATHS,
                visible=not READONLY_PATHS,
            )
            is_trim_mode = gr.Checkbox(
                value=True, label="Collapse multiple whitespaces", interactive=True
            )

            with gr.Row():
                use_carriage_return_as_boundary = gr.Checkbox(
                    value=True,
                    label="Line break is sentence boundary",
                    interactive=True,
                )
                use_empty_line_as_boundary = gr.Checkbox(
                    value=True,
                    label="Empty line is sentence boundary",
                    interactive=True,
                )

    # input
    document_text = gr.Textbox(
        lines=3, label="Text", placeholder="Enter you document text here..."
    )

    # action button
    segmentize_btn = gr.Button("Segmentize", variant="primary")

    # outputs
    sentences_text = gr.Textbox(
        lines=3,
        label="Sentences",
        placeholder="Sentences (one per line)",
        show_copy_button=True,
    )
    sentences_dataset = gr.Dataset(
        components=[gr.Textbox(visible=False)], label="Sentences", samples=[]
    )

    # worker function
    def segmentize_text(
        text: str,
        fn_sentence_boundaries: Optional[str] = None,
        fn_pre_boundary_rules: Optional[str] = None,
        fn_pre_boundaries_list: Optional[str] = None,
        fn_post_boundary_rules: Optional[str] = None,
        fn_post_boundaries_list: Optional[str] = None,
        is_auto_uppercase_first_letter_pre_list: bool = True,
        is_trim_mode: bool = True,
        use_carriage_return_as_boundary: bool = True,
        use_empty_line_as_boundary: bool = True,
        encoding: str = "utf-8",
    ) -> str:
        cwd = os.getcwd()
        if os.path.relpath(os.path.realpath(fn_sentence_boundaries), cwd).startswith(
            "../"
        ):
            raise gr.Error("Invalid 'fn_sentence_boundaries' path!")
        if os.path.relpath(os.path.realpath(fn_pre_boundary_rules), cwd).startswith(
            "../"
        ):
            raise gr.Error("Invalid 'fn_pre_boundary_rules' path!")
        if os.path.relpath(os.path.realpath(fn_pre_boundaries_list), cwd).startswith(
            "../"
        ):
            raise gr.Error("Invalid 'fn_pre_boundaries_list' path!")
        if os.path.relpath(os.path.realpath(fn_post_boundary_rules), cwd).startswith(
            "../"
        ):
            raise gr.Error("Invalid 'fn_post_boundary_rules' path!")
        if os.path.relpath(os.path.realpath(fn_post_boundaries_list), cwd).startswith(
            "../"
        ):
            raise gr.Error("Invalid 'fn_post_boundaries_list' path!")

        if not os.path.isfile(fn_sentence_boundaries):
            gr.Warning("File 'fn_sentence_boundaries' not found!")
        if not os.path.isfile(fn_pre_boundary_rules):
            gr.Warning("File 'fn_pre_boundary_rules' not found!")
        if not os.path.isfile(fn_pre_boundaries_list):
            gr.Warning("File 'fn_pre_boundaries_list' not found!")
        if not os.path.isfile(fn_post_boundary_rules):
            gr.Warning("File 'fn_post_boundary_rules' not found!")
        if not os.path.isfile(fn_post_boundaries_list):
            gr.Warning("File 'fn_post_boundaries_list' not found!")

        lcc_segmentizer = lcc.segmentizer.LineAwareSegmentizer(
            fn_sentence_boundaries=fn_sentence_boundaries,
            fn_pre_boundary_rules=fn_pre_boundary_rules,
            fn_pre_boundaries_list=fn_pre_boundaries_list,
            fn_post_boundary_rules=fn_post_boundary_rules,
            fn_post_boundaries_list=fn_post_boundaries_list,
            encoding=encoding,
            is_auto_uppercase_first_letter_pre_list=is_auto_uppercase_first_letter_pre_list,
            is_trim_mode=is_trim_mode,
            use_carriage_return_as_boundary=use_carriage_return_as_boundary,
            use_empty_line_as_boundary=use_empty_line_as_boundary,
        )

        sentences = lcc_segmentizer.segmentize(text)
        return {
            sentences_text: "\n".join(sentences),
            sentences_dataset: [
                [sentence] for sentence in sentences if sentence.strip()
            ],
        }

    # action buttons event handler
    segmentize_btn.click(
        fn=segmentize_text,
        inputs=[
            document_text,
            fn_sentence_boundaries,
            fn_pre_boundary_rules,
            fn_pre_boundaries_list,
            fn_post_boundary_rules,
            fn_post_boundaries_list,
            is_auto_uppercase_first_letter_pre_list,
            is_trim_mode,
            use_carriage_return_as_boundary,
            use_empty_line_as_boundary,
        ],
        outputs=[sentences_text, sentences_dataset],
        api_name="segmentize",
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    segmentizer.launch(show_api=False, share=False)
