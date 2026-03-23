import colorsys
import re

from imgui_bundle import imgui


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _vec2(x: float, y: float):
    if hasattr(imgui, "ImVec2"):
        return imgui.ImVec2(float(x), float(y))
    return (float(x), float(y))


def _vec4(r: float, g: float, b: float, a: float):
    if hasattr(imgui, "ImVec4"):
        return imgui.ImVec4(float(r), float(g), float(b), float(a))
    return (float(r), float(g), float(b), float(a))


def _safe_setattr(obj, name: str, value):
    if hasattr(obj, name):
        setattr(obj, name, value)


def _build_col_lookup():
    lut = {}
    for n in dir(imgui.Col_):
        if n.startswith("_"):
            continue
        lut[_norm(n)] = getattr(imgui.Col_, n)
    return lut


def _set_col(style, col_lut: dict, name: str, rgba):
    key = _norm(name)
    if key not in col_lut:
        return
    idx = col_lut[key]
    color = _vec4(*rgba)
    if hasattr(style, "set_color_"):
        style.set_color_(idx, color)
        return
    if hasattr(style, "colors"):
        style.colors[idx] = color


def _get_col(style, idx: int):
    if hasattr(style, "color_"):
        return style.color_(idx)
    if hasattr(style, "colors"):
        return style.colors[idx]
    raise AttributeError("Style object has neither color_() nor colors[] API")


def _set_col_idx(style, idx: int, rgba):
    color = _vec4(*rgba)
    if hasattr(style, "set_color_"):
        style.set_color_(idx, color)
        return
    if hasattr(style, "colors"):
        style.colors[idx] = color
        return
    raise AttributeError("Style object has neither set_color_() nor colors[] API")


def _dim(c, lit01: int):
    h, s, v = colorsys.rgb_to_hsv(c[0], c[1], c[2])
    v = v * 0.65 if lit01 else v * 0.65
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    # Keep dominant blue as in original snippet.
    if c[2] > c[0] and c[2] > c[1]:
        return (r, g, c[2], c[3])
    return (r, g, b, c[3])


def apply_theme_v3(
    hue07=0,
    alt07=0,
    nav07=0,
    lit01=0,
    compact01=0,
    border01=1,
    shape0123=1,
):
    rounded = shape0123 == 2
    style = imgui.get_style()
    col_lut = _build_col_lookup()

    _8 = 4.0 if compact01 else 8.0
    _4 = 2.0 if compact01 else 4.0
    _2 = 0.5 if compact01 else 1.0

    _safe_setattr(style, "alpha", 1.0)
    _safe_setattr(style, "disabled_alpha", 0.3)
    _safe_setattr(style, "window_padding", _vec2(4, _8))
    _safe_setattr(style, "frame_padding", _vec2(4, _4))
    _safe_setattr(style, "item_spacing", _vec2(_8, _2 + _2))
    _safe_setattr(style, "item_inner_spacing", _vec2(4, 4))
    _safe_setattr(style, "indent_spacing", 16.0)
    _safe_setattr(style, "scrollbar_size", 12.0 if compact01 else 18.0)
    _safe_setattr(style, "grab_min_size", 16.0 if compact01 else 20.0)

    _safe_setattr(style, "window_border_size", float(border01))
    _safe_setattr(style, "child_border_size", float(border01))
    _safe_setattr(style, "popup_border_size", float(border01))
    _safe_setattr(style, "frame_border_size", 0.0)

    _safe_setattr(style, "window_rounding", 4.0)
    _safe_setattr(style, "child_rounding", 6.0)
    _safe_setattr(style, "frame_rounding", 0.0 if shape0123 == 0 else 4.0 if shape0123 == 1 else 12.0)
    _safe_setattr(style, "popup_rounding", 4.0)
    _safe_setattr(style, "scrollbar_rounding", (8.0 if rounded else 0.0) + 4.0)
    _safe_setattr(style, "grab_rounding", getattr(style, "frame_rounding", 4.0))

    _safe_setattr(style, "tab_border_size", 0.0)
    _safe_setattr(style, "tab_bar_border_size", 2.0)
    _safe_setattr(style, "tab_bar_overline_size", 2.0)
    _safe_setattr(style, "tab_close_button_min_width_selected", -1.0)
    _safe_setattr(style, "tab_close_button_min_width_unselected", -1.0)
    _safe_setattr(style, "tab_rounding", 1.0 if rounded else 0.0)

    _safe_setattr(style, "cell_padding", _vec2(8.0, 4.0))
    _safe_setattr(style, "window_title_align", _vec2(0.5, 0.5))
    _safe_setattr(style, "color_button_position", imgui.Dir_.right if hasattr(imgui, "Dir_") else 1)
    _safe_setattr(style, "button_text_align", _vec2(0.5, 0.5))
    _safe_setattr(style, "selectable_text_align", _vec2(0.5, 0.5))
    _safe_setattr(style, "separator_text_align", _vec2(1.0, 0.5))
    _safe_setattr(style, "separator_text_border_size", 1.0)
    _safe_setattr(style, "separator_text_padding", _vec2(0.0, 0.0))
    _safe_setattr(style, "window_min_size", _vec2(32.0, 16.0))
    _safe_setattr(style, "columns_min_spacing", 6.0)
    _safe_setattr(style, "circle_tessellation_max_error", 4.0 if shape0123 == 3 else 0.3)

    cyan = (0 / 255.0, 192 / 255.0, 255 / 255.0, 1.0)
    red = (230 / 255.0, 0 / 255.0, 0 / 255.0, 1.0)
    yellow = (240 / 255.0, 210 / 255.0, 0 / 255.0, 1.0)
    orange = (255 / 255.0, 144 / 255.0, 0 / 255.0, 1.0)
    lime = (192 / 255.0, 255 / 255.0, 0 / 255.0, 1.0)
    aqua = (0 / 255.0, 255 / 255.0, 192 / 255.0, 1.0)
    magenta = (255 / 255.0, 0 / 255.0, 88 / 255.0, 1.0)
    purple = (192 / 255.0, 0 / 255.0, 255 / 255.0, 1.0)

    choices = [cyan, red, yellow, orange, lime, aqua, magenta, purple]

    def pick(v, default):
        if isinstance(v, str) and len(v) == 1:
            m = {"C": 0, "R": 1, "Y": 2, "O": 3, "L": 4, "A": 5, "M": 6, "P": 7}
            idx = m.get(v.upper(), default)
            return choices[idx]
        if isinstance(v, int) and 0 <= v < len(choices):
            return choices[v]
        return choices[default]

    alt = pick(alt07, 0)
    alt = _dim(alt, lit01) if lit01 else alt

    hi = pick(hue07, 0)
    lo = _dim(hi, lit01)

    nav = pick(nav07, 3)
    nav = _dim(nav, lit01) if lit01 else nav

    link = (0.26, 0.59, 0.98, 1.0)
    grey0 = (0.04, 0.05, 0.07, 1.0)
    grey1 = (0.08, 0.09, 0.11, 1.0)
    grey2 = (0.10, 0.11, 0.13, 1.0)
    grey3 = (0.12, 0.13, 0.15, 1.0)
    grey4 = (0.16, 0.17, 0.19, 1.0)
    grey5 = (0.18, 0.19, 0.21, 1.0)

    luma = lambda v, a: (v / 100.0, v / 100.0, v / 100.0, a / 100.0)

    # Color names mirror Dear ImGui enums; missing names are safely ignored.
    _set_col(style, col_lut, "Text", luma(100, 100))
    _set_col(style, col_lut, "TextDisabled", luma(39, 100))
    _set_col(style, col_lut, "WindowBg", grey1)
    _set_col(style, col_lut, "ChildBg", (0.09, 0.10, 0.12, 1.0))
    _set_col(style, col_lut, "PopupBg", grey1)
    _set_col(style, col_lut, "Border", grey4)
    _set_col(style, col_lut, "BorderShadow", grey1)
    _set_col(style, col_lut, "FrameBg", (0.11, 0.13, 0.15, 1.0))
    _set_col(style, col_lut, "FrameBgHovered", grey4)
    _set_col(style, col_lut, "FrameBgActive", grey4)
    _set_col(style, col_lut, "TitleBg", grey0)
    _set_col(style, col_lut, "TitleBgActive", grey0)
    _set_col(style, col_lut, "TitleBgCollapsed", grey1)
    _set_col(style, col_lut, "MenuBarBg", grey2)
    _set_col(style, col_lut, "ScrollbarBg", grey0)
    _set_col(style, col_lut, "ScrollbarGrab", grey3)
    _set_col(style, col_lut, "ScrollbarGrabHovered", lo)
    _set_col(style, col_lut, "ScrollbarGrabActive", hi)
    _set_col(style, col_lut, "CheckMark", alt)
    _set_col(style, col_lut, "SliderGrab", lo)
    _set_col(style, col_lut, "SliderGrabActive", hi)
    _set_col(style, col_lut, "Button", (0.10, 0.11, 0.14, 1.0))
    _set_col(style, col_lut, "ButtonHovered", lo)
    _set_col(style, col_lut, "ButtonActive", grey5)
    _set_col(style, col_lut, "Header", grey3)
    _set_col(style, col_lut, "HeaderHovered", lo)
    _set_col(style, col_lut, "HeaderActive", hi)
    _set_col(style, col_lut, "Separator", (0.13, 0.15, 0.19, 1.0))
    _set_col(style, col_lut, "SeparatorHovered", lo)
    _set_col(style, col_lut, "SeparatorActive", hi)
    _set_col(style, col_lut, "ResizeGrip", luma(15, 100))
    _set_col(style, col_lut, "ResizeGripHovered", lo)
    _set_col(style, col_lut, "ResizeGripActive", hi)
    _set_col(style, col_lut, "InputTextCursor", luma(100, 100))
    _set_col(style, col_lut, "TabHovered", grey3)
    _set_col(style, col_lut, "Tab", grey1)
    _set_col(style, col_lut, "TabSelected", grey3)
    _set_col(style, col_lut, "TabSelectedOverline", hi)
    _set_col(style, col_lut, "TabDimmed", grey1)
    _set_col(style, col_lut, "TabDimmedSelected", grey1)
    _set_col(style, col_lut, "TabDimmedSelectedOverline", lo)
    _set_col(style, col_lut, "DockingPreview", grey1)
    _set_col(style, col_lut, "DockingEmptyBg", luma(20, 100))
    _set_col(style, col_lut, "PlotLines", grey5)
    _set_col(style, col_lut, "PlotLinesHovered", lo)
    _set_col(style, col_lut, "PlotHistogram", grey5)
    _set_col(style, col_lut, "PlotHistogramHovered", lo)
    _set_col(style, col_lut, "TableHeaderBg", grey0)
    _set_col(style, col_lut, "TableBorderStrong", grey0)
    _set_col(style, col_lut, "TableBorderLight", grey0)
    _set_col(style, col_lut, "TableRowBg", grey3)
    _set_col(style, col_lut, "TableRowBgAlt", grey2)
    _set_col(style, col_lut, "TextLink", link)
    _set_col(style, col_lut, "TextSelectedBg", luma(39, 100))
    _set_col(style, col_lut, "TreeLines", luma(39, 100))
    _set_col(style, col_lut, "DragDropTarget", nav)
    _set_col(style, col_lut, "NavCursor", nav)
    _set_col(style, col_lut, "NavWindowingHighlight", lo)
    _set_col(style, col_lut, "NavWindowingDimBg", luma(0, 63))
    _set_col(style, col_lut, "ModalWindowDimBg", luma(0, 63))

    if lit01:
        col_count = int(getattr(imgui.Col_, "count", getattr(imgui.Col_, "COUNT", 0)))
        for i in range(col_count):
            c = _get_col(style, i)
            h, s, v = colorsys.rgb_to_hsv(c.x, c.y, c.z)
            if s < 0.5:
                v = 1.0 - v
                s *= 0.15
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            _set_col_idx(style, i, (r, g, b, c.w))

    return 0
