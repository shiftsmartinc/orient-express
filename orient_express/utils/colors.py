from typing import Any
import colorsys


def generate_color_scheme(
    strings: list[str],
    saturation: float = 0.7,
    lightness: float = 0.5,
    transparency: float = 0.6,
) -> dict[Any, tuple[int, int, int, int]]:
    """
    Assign unique colors to strings with max distance between them.
    """
    n = len(strings)
    colors = {}

    for i, string in enumerate(strings):
        hue = i / n  # Normalize hue to the range [0, 1]

        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        color_code = (
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255),
            int(transparency * 150),
        )
        colors[string] = color_code

    return colors
