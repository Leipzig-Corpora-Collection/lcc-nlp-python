import os.path

import gradio as gr

# ---------------------------------------------------------------------------

# isort: split
from cleaner import cleaner as cleaner_demo
from lani import lani as lani_demo
from segmentizer import segmentizer as segmentizer_demo
from tokenizer import tokenizer as tokenizer_demo

from lcc_theme import lcc_theme

# NOTE: only works if hosted on huggingface ...
# segmentizer_demo = gr.load(
#     "segmentizer",
#     title="Segmentizer",
#     description="Sentence segmentizer"
# )

# replace with our LCC footer
GRADIO_CSS = """
.gradio-container.app {
  padding-bottom: 0 !important;
}
.prose a strong {
  color: inherit;
}
"""
FOOTER_HTML = """
<div id="footer">
  <ol id="footerlist" class="breadcrumb">
    <li><a href="https://www.uni-leipzig.de" target="_blank" title="Leipzig University">Leipzig University</a></li>
    <li><a href="https://www.saw-leipzig.de" target="_blank" title="SAW">SAW</a></li>
    <li><a href="https://infai.org" target="_blank" title="InfAI">InfAI</a></li>
    <li><a href="https://wortschatz-leipzig.de/de" target="_blank" title="LCC portal">LCC portal</a></li>
    <li><a href="https://wortschatz-leipzig.de/de/documentation" title="Documentation">Documentation</a></li>
    <li><a href="https://wortschatz-leipzig.de/de/usage" title=" Terms of Usage"> Terms of Usage</a></li>
    <li><a href="https://wortschatz-leipzig.de/de/privacy" title=" Privacy"> Privacy</a></li>
    <li><a href="https://wortschatz-leipzig.de/de/accessibility" title="Accessibility">Accessibility</a></li>
    <li> <a href="https://wortschatz-leipzig.de/de/contact" title="Contact">Contact</a></li>
  </ol>
  <span id="copyright">© 1998 - 2024 Deutscher Wortschatz / Wortschatz Leipzig. All rights reserved.</span>
</div>
"""
FOOTER_CSS = """
#footer {
  background: #F5F5F5;
  width: 100%;
  padding-top: 4px;
  min-height: 4.5em;
  box-shadow: 0px -1px 1px rgba(0, 0, 0, 0.05);
  border-top: 1px solid #DDD;
  text-align: center;
}
#footer p {
  font-size: 12px;
  color: #333;
  margin-bottom: 1px;
}
#footerlist {
  margin-bottom: 2px;
  padding-top: 2px;
  padding-bottom: 2px;
}
#footerlist li {
  display: inline-block;
  margin-right: 5px;
}
#footerlist li a {
  color: #007bff;
  text-decoration: none;
}
#footerlist ul.dropdown-menu {
  min-width: auto;
}
#footerlist ul.dropdown-menu li {
  display: list-item;
  margin-right: 0px;
}
#copyright {
  color: #808080;
  font-size: 13px;
}
"""
FOOTER_JS = """
<script>
  var script = document.createElement("script");
  script.appendChild(document.createTextNode("setTimeout(() => { document.querySelector('footer').replaceWith(document.querySelector('#footer')); }, 1000);"));
  document.body.appendChild(script);
</script>
"""

FN_FAVICON = os.path.join(os.path.dirname(__file__), "favicon.ico")

with gr.Blocks(
    title="Leipzig Corpora Collection (LCC) Natural Language Processing (NLP) Tools",
    theme=lcc_theme,
    css="\n".join([FOOTER_CSS, GRADIO_CSS]),
    head=FOOTER_JS,
) as lcc_demo:
    description = gr.Markdown(
        value="""
        # Leipzig Corpora Collection (LCC) Natural Language Processing (NLP) Tools
        
        This is a web application to test the
        [**Leipzig Corpora Collection** (LCC)](https://wortschatz-leipzig.de/en) _Natural Language Processing_ (NLP) tools
        used to curate our [LCC corpora and datasets](https://wortschatz-leipzig.de/en/download).
                            
        You can find the source code at: https://github.com/Leipzig-Corpora-Collection/lcc-nlp-python.
        """
    )

    tabs = [
        ("🪚 Text → Sentences", segmentizer_demo),
        ("🪓 Sentence → Tokens", tokenizer_demo),
        ("🔎 Language Identification", lani_demo),
        ("🧹 Sentence Cleaner", cleaner_demo),
    ]

    gr.TabbedInterface(
        interface_list=[tab[1] for tab in tabs],
        tab_names=[tab[0] for tab in tabs],
    )

    gr.HTML(value=FOOTER_HTML)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lcc_demo.queue().launch(show_api=False, share=False, favicon_path=FN_FAVICON)
