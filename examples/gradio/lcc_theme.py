from typing import Iterable
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from gradio.themes.utils.colors import Color
from gradio.themes.utils.fonts import Font, GoogleFont
from gradio.themes.utils.sizes import Size


# https://github.com/twbs/bootstrap/blob/af092931f09daadc898b77e78728cd763b974f5d/scss/_variables.scss#L39
# https://getbootstrap.com/docs/5.0/customize/color/#theme-colors
# https://codepen.io/emdeoh/pen/zYOQOPB --> mix black white (--> *-90)
# https://www.tints.dev/blue/0D6EFD

bootstrap_blue = colors.Color(
    "#e7f1ff",
    "#cfe2ff",
    "#9ec5fe",
    "#6ea8fe",
    "#3d8bfd",
    "#0d6efd",  # base / blue-500
    "#0a58ca",
    "#084298",
    "#052c65",
    "#031633",
    "#010b19",
    "blue",
)

bootstrap_grey = colors.Color(
    "#f7f8f8",
    "#f8f9fa",
    "#e9ecef",
    "#dee2e6",
    "#ced4da",
    "#adb5bd",  # base / grey-500
    "#6c757d",
    "#495057",
    "#343a40",
    "#212529",
    "#111213",
    "grey",
)

bootstrap_textsize = sizes.Size(
    name="text",
    xxs=".8rem",
    xs=".85rem",
    sm=".875rem",
    md="1rem",
    lg="1.25rem",
    xl="1.5rem",
    xxl="2rem",
)

# https://github.com/twbs/bootstrap/blob/af092931f09daadc898b77e78728cd763b974f5d/scss/_variables.scss#L547

bootstrap_radius = sizes.Size(
    name="radius",
    xxs="1px",
    xs="1px",
    sm=".25rem",
    md=".375rem",
    lg=".5rem",
    xl="1rem",
    xxl="2rem",
)


class LCCTheme(Base):
    def __init__(
        self,
        *,
        primary_hue: Color | str = bootstrap_blue,
        secondary_hue: Color | str = bootstrap_blue,
        neutral_hue: Color | str = bootstrap_grey,
        text_size: Size | str = bootstrap_textsize,
        spacing_size: Size | str = sizes.spacing_md,
        radius_size: Size | str = bootstrap_radius,
        font: Font
        | str
        | Iterable[Font | str] = (
            "system-ui",
            "-apple-system",
            "Segoe UI",
            "Roboto",
            "Helvetica Neue",
            "Arial",
            "Noto Sans",
            "Liberation Sans",
            "sans-serif",
            "Apple Color Emoji",
            "Segoe UI Emoji",
            "Segoe UI Symbol",
            "Noto Color Emoji",
        ),
        font_mono: Font
        | str
        | Iterable[Font | str] = (
            "SFMono-Regular",
            "Menlo",
            "Monaco",
            "Consolas",
            "Liberation Mono",
            "Courier New",
            "monospace",
        )
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            spacing_size=spacing_size,
            radius_size=radius_size,
            font=font,
            font_mono=font_mono,
        )
        self.name = "lcc"
        super().set(
            # colors
            button_primary_background_fill="#0d6efd",
            button_primary_border_color="#0d6efd",
            button_primary_text_color="#fff",
            button_secondary_background_fill="#fff",
            button_secondary_text_color=bootstrap_grey.c700,
            block_title_text_color=bootstrap_grey.c700,
            body_text_color_subdued=bootstrap_grey.c500,
            # padding
            # borders
            block_radius=".3rem",
            # block_label_radius=sizes.radius_none,
            # block_title_radius=sizes.radius_none,
            button_large_radius="0.3rem",
            button_small_radius="0.25rem",
            checkbox_border_radius=".25em",
            checkbox_border_width="1px",
            checkbox_border_color=bootstrap_grey.c300,
            container_radius=".3rem",
            # embed_radius=sizes.radius_none,
            # input_radius=sizes.radius_md,
            # table_radius=sizes.radius_none,
            # block labels
            # ...
        )


lcc_theme = LCCTheme()
