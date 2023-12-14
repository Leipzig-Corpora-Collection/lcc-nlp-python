import os.path
from typing import Optional

import gradio as gr

import lcc.tokenizer

# ---------------------------------------------------------------------------


READONLY_PATHS = os.getenv("READONLY_PATHS", "True").lower() == "true"


# ---------------------------------------------------------------------------


with gr.Blocks() as tokenizer:
    dn_resources = "resources/tokenizer"

    # configuration
    with gr.Accordion(
        "⚙️ Tokenizer Configuration", open=False, visible=not READONLY_PATHS
    ):
        strAbbrevListFile = gr.Textbox(
            value=os.path.join(dn_resources, "default.abbrev"),
            label="Abbreviations list file",
            max_lines=1,
            interactive=not READONLY_PATHS,
            visible=not READONLY_PATHS,
        )
        TOKENISATION_CHARACTERS_FILE_NAME = gr.Textbox(
            value=os.path.join(dn_resources, "100-wn-all.txt"),
            label="Fixed word number to token file",
            max_lines=1,
            interactive=not READONLY_PATHS,
            visible=not READONLY_PATHS,
        )
        strCharacterActionsFile = gr.Textbox(
            value=os.path.join(dn_resources, "tokenization_character_actions.txt"),
            label="Tokenization actions file",
            max_lines=1,
            interactive=not READONLY_PATHS,
            visible=not READONLY_PATHS,
        )
        fixedTokensFile = gr.Textbox(
            value=os.path.join(dn_resources, "fixed_tokens.txt"),
            label="Fixed tokens file",
            max_lines=1,
            interactive=not READONLY_PATHS,
            visible=not READONLY_PATHS,
        )

    # input
    sentence_text = gr.Textbox(label="Sentence", placeholder="Sentence")

    # action button
    tokenize_btn = gr.Button("Tokenize", variant="primary")

    # outputs
    tokenized_text = gr.Textbox(
        label="Tokenized", placeholder="Tokenized sentence", show_copy_button=True
    )
    tokenized_highlighted = gr.HighlightedText(
        label="Highlight split tokens", combine_adjacent=True
    )

    # worker function
    def tokenize_text(
        text: str,
        strAbbrevListFile: Optional[str] = None,
        TOKENISATION_CHARACTERS_FILE_NAME: Optional[str] = None,
        strCharacterActionsFile: Optional[str] = None,
        fixedTokensFile: Optional[str] = None,
    ) -> str:
        cwd = os.getcwd()
        if os.path.relpath(os.path.realpath(strAbbrevListFile), cwd).startswith("../"):
            raise gr.Error("Invalid 'strAbbrevListFile' path!")
        if os.path.relpath(
            os.path.realpath(TOKENISATION_CHARACTERS_FILE_NAME), cwd
        ).startswith("../"):
            raise gr.Error("Invalid 'TOKENISATION_CHARACTERS_FILE_NAME' path!")
        if os.path.relpath(os.path.realpath(strCharacterActionsFile), cwd).startswith(
            "../"
        ):
            raise gr.Error("Invalid 'strCharacterActionsFile' path!")
        if os.path.relpath(os.path.realpath(fixedTokensFile), cwd).startswith("../"):
            raise gr.Error("Invalid 'fixedTokensFile' path!")

        if not os.path.isfile(strAbbrevListFile):
            gr.Warning("File 'strAbbrevListFile' not found!")
        if not os.path.isfile(TOKENISATION_CHARACTERS_FILE_NAME):
            gr.Warning("File 'TOKENISATION_CHARACTERS_FILE_NAME' not found!")
        if not os.path.isfile(strCharacterActionsFile):
            gr.Warning("File 'strCharacterActionsFile' not found!")
        if not os.path.isfile(fixedTokensFile):
            gr.Warning("File 'fixedTokensFile' not found!")

        lcc_tokenizer = lcc.tokenizer.CharacterBasedWordTokenizerImproved(
            strAbbrevListFile=strAbbrevListFile,
            TOKENISATION_CHARACTERS_FILE_NAME=TOKENISATION_CHARACTERS_FILE_NAME,
            strCharacterActionsFile=strCharacterActionsFile,
            fixedTokensFile=fixedTokensFile,
        )

        tokenized = lcc_tokenizer.execute(text)
        return tokenized

    # output transformer function
    def align_tokens_to_input(text: str, tokens: str):
        aligned = lcc.tokenizer.AlignedSentence(text.strip(), tokens.strip())

        highlighted = []
        for word, new_space in zip(aligned.tokens(), [False] + aligned.tokens_glue()):
            if new_space:
                highlighted[-1] = (highlighted[-1][0], True)

            highlighted.append((" ", True if new_space else None))
            highlighted.append((word, True if new_space else None))

        return highlighted

    # action buttons event handler
    tokenize_btn.click(
        fn=tokenize_text,
        inputs=[
            sentence_text,
            strAbbrevListFile,
            TOKENISATION_CHARACTERS_FILE_NAME,
            strCharacterActionsFile,
            fixedTokensFile,
        ],
        outputs=tokenized_text,
        api_name="tokenize",
    ).then(
        fn=align_tokens_to_input,
        inputs=[sentence_text, tokenized_text],
        outputs=tokenized_highlighted,
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tokenizer.launch(show_api=False, share=False)
