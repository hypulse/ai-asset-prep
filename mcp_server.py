from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

import image_tasks


mcp = FastMCP("image-cut-fit")


@mcp.tool()
def crop_image(
    input_path: str,
    output_path: str | None = None,
    alpha_threshold: int = 16,
    model_name: str = "u2net",
    preserve_interior: bool = True,
    post_process_mask: bool = True,
) -> dict[str, Any]:
    """Remove the background when requested, crop to visible alpha, and write a PNG."""
    return image_tasks.crop_image(
        input_path,
        output_path,
        alpha_threshold=alpha_threshold,
        model_name=model_name,
        preserve_interior=preserve_interior,
        post_process_mask=post_process_mask,
    )


@mcp.tool()
def fit_image(
    input_path: str,
    output_path: str | None = None,
    width: int = 1280,
    height: int = 720,
    resize_mode: str = "contain_center",
    rotation_degrees: float = 0,
    alpha_threshold: int = 16,
    padding_top: int = 0,
    padding_right: int = 0,
    padding_bottom: int = 0,
    padding_left: int = 0,
    transparent_background: bool = True,
    background_color: str = "#000000",
) -> dict[str, Any]:
    """Rotate, resize, pad, and write a PNG."""
    return image_tasks.fit_image(
        input_path,
        output_path,
        width=width,
        height=height,
        resize_mode=resize_mode,
        rotation_degrees=rotation_degrees,
        alpha_threshold=alpha_threshold,
        padding_top=padding_top,
        padding_right=padding_right,
        padding_bottom=padding_bottom,
        padding_left=padding_left,
        transparent_background=transparent_background,
        background_color=background_color,
    )


@mcp.tool()
def make_sprite_sheet(
    input_paths: list[str],
    output_path: str | None = None,
    metadata_path: str | None = None,
    source_scales: list[float] | None = None,
    scale: float = 1.0,
    first_width: int | None = None,
    gap: int = 0,
    resampling: str = "nearest",
) -> dict[str, Any]:
    """Combine PNG/JPG/WebP inputs into a square-ish transparent sprite sheet."""
    return image_tasks.make_sprite_sheet(
        input_paths,
        output_path,
        metadata_path=metadata_path,
        source_scales=source_scales,
        scale=scale,
        first_width=first_width,
        gap=gap,
        resampling=resampling,
    )


@mcp.tool()
def recover_sprite_sheet(
    input_path: str,
    output_dir: str | None = None,
    metadata_path: str | None = None,
    scale: float = 0.5,
    alpha_threshold: int = 16,
    min_area: int = 16,
    resampling: str = "nearest",
) -> dict[str, Any]:
    """Scale a transparent sheet and split connected alpha components into PNGs."""
    return image_tasks.recover_sprite_sheet_to_dir(
        input_path,
        output_dir,
        metadata_path=metadata_path,
        scale=scale,
        alpha_threshold=alpha_threshold,
        min_area=min_area,
        resampling=resampling,
    )


@mcp.tool()
def make_tileset_guide(
    output_path: str | None = None,
    metadata_path: str | None = None,
    tile_size: int = 64,
    prefix: str = "ground",
    gap: int = 0,
    margin: int = 0,
    line_width: int = 1,
) -> dict[str, Any]:
    """Create the 3x3 tileset guide PNG."""
    return image_tasks.make_tileset_guide(
        output_path,
        metadata_path=metadata_path,
        tile_size=tile_size,
        prefix=prefix,
        gap=gap,
        margin=margin,
        line_width=line_width,
    )


@mcp.tool()
def slice_tileset(
    input_path: str,
    output_dir: str | None = None,
    metadata_path: str | None = None,
    tile_size: int = 64,
    prefix: str = "ground",
    gap: int = 0,
    margin: int = 0,
    skip_transparent_tiles: bool = False,
) -> dict[str, Any]:
    """Slice a completed 3x3 tileset guide image into individual PNG tiles."""
    return image_tasks.slice_tileset(
        input_path,
        output_dir,
        metadata_path=metadata_path,
        tile_size=tile_size,
        prefix=prefix,
        gap=gap,
        margin=margin,
        skip_transparent_tiles=skip_transparent_tiles,
    )


if __name__ == "__main__":
    mcp.run()
