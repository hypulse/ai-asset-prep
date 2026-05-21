from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

import image_tasks


mcp = FastMCP("ai-asset-prep")


@mcp.tool()
def generate_image(
    prompt: str,
    output_path: str | None = None,
    name: str | None = None,
    model: str = image_tasks.IMAGE_GENERATION_DEFAULT_MODEL,
    size: str = image_tasks.IMAGE_GENERATION_DEFAULT_SIZE,
    quality: str = image_tasks.IMAGE_GENERATION_DEFAULT_QUALITY,
    output_format: str = image_tasks.IMAGE_GENERATION_DEFAULT_FORMAT,
    background: str | None = None,
) -> dict[str, Any]:
    """Generate an image from a text prompt and save it under outputs or a chosen path.

    Use this first when the user asks to create a new visual asset from text.
    After generation, call crop_image for background removal/alpha trimming,
    fit_image for exact dimensions, or make_sprite_sheet to combine generated
    assets into a sheet. Prefer absolute output_path values. If output_path is
    omitted, the image is written under /Users/seungjae/Codes/ai-asset-prep/outputs
    with a unique filename.

    Defaults are low-cost: model="gpt-image-1-mini", size="1024x1024",
    quality="low", output_format="png". output_format must be png, jpeg, or
    webp. background may be "opaque", "transparent", or null. Requires
    OPENAI_API_KEY in the environment or /Users/seungjae/Codes/ai-asset-prep/.env.
    Returns output path, model, size, quality, output_format, background, prompt,
    and prompt_summary.
    """
    return image_tasks.generate_image(
        prompt,
        output_path,
        name=name,
        model=model,
        size=size,
        quality=quality,
        output_format=output_format,
        background=background,
    )


@mcp.tool()
def crop_image(
    input_path: str,
    output_path: str | None = None,
    alpha_threshold: int = 16,
    model_name: str = "u2net",
    preserve_interior: bool = True,
    post_process_mask: bool = True,
) -> dict[str, Any]:
    """Crop an image to its visible non-transparent pixels and optionally remove its background.

    Use this when the user asks to remove an image background, trim transparent
    whitespace, crop to the alpha bounds, or prepare a transparent PNG asset.
    Prefer absolute file paths. If output_path is omitted, the PNG is written
    under /Users/seungjae/Codes/ai-asset-prep/outputs with a unique filename.

    Set model_name to "u2net" for the default rembg background removal,
    "isnet-general-use" for general image preservation, "isnet-anime" for
    anime/illustration style images, or "none" when the input already has
    transparency and only alpha-crop is needed. alpha_threshold defaults to 16.
    Returns output path, bbox, source size, cropped size, removed size, model,
    alpha threshold, and any fallback reason.
    """
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
    """Resize an image to an exact width and height, with optional rotation and padding.

    Use this when the user asks for a specific final PNG size such as 1280x720,
    24x24, or 36x36. This tool does not remove backgrounds; call crop_image
    first if background removal or alpha trimming is needed. Prefer absolute
    input_path and output_path values. If output_path is omitted, the PNG is
    written under /Users/seungjae/Codes/ai-asset-prep/outputs with a unique
    filename.

    resize_mode must be one of: "contain_center", "contain_top",
    "contain_bottom", "contain_left", "contain_right", or "stretch".
    contain_* preserves aspect ratio and places the image inside the target
    canvas; stretch distorts to fill exactly. rotation_degrees rotates before
    resizing. Padding is added after resizing. If transparent_background is
    false, background_color is used for the canvas. Returns output path,
    source/rotated/resized/final sizes, resize mode, rotation, padding, and
    background settings.
    """
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
    """Combine multiple images into a square-ish transparent sprite sheet.

    Use this when the user asks to make a sprite sheet, combine several PNGs
    into one sheet, preserve pixel art, or pack mixed-size game assets. Provide
    input_paths in the desired order. Prefer absolute paths. If output_path is
    omitted, the PNG is written under /Users/seungjae/Codes/ai-asset-prep/outputs
    with a unique filename. If metadata_path is provided, placement metadata is
    also written as JSON.

    source_scales controls per-image pre-scale before packing. It may be null,
    a single value applied to every input, or one value per input. Use this for
    mixed-size assets before the final whole-sheet scale. scale is applied after
    packing. first_width, when provided, overrides scale by calculating the final
    sheet scale from the first image's post-source-scale width. gap is transparent
    spacing between cells. resampling should usually be "nearest" for pixel art
    and "lanczos" for smooth scaling. Returns output path, sheet sizes, cell
    size, grid dimensions, gap, scale, source scales, and per-image placements.
    """
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
    """Split a transparent sprite sheet into individual connected alpha components.

    Use this when the user asks to recover sprites from a sheet, split a cursor
    or character sheet, or inspect individual silhouettes from one transparent
    PNG. Prefer absolute input_path and output_dir values. If output_dir is
    omitted, files are written under /Users/seungjae/Codes/ai-asset-prep/outputs
    in a unique sprite directory. If metadata_path is provided, split metadata is
    also written as JSON.

    The tool first scales the whole sheet by scale, then detects connected
    components whose alpha is greater than alpha_threshold and whose area is at
    least min_area. For tiny cursor assets, useful values are often scale=0.25,
    alpha_threshold=16, min_area=16, resampling="nearest". Returns output_dir,
    scaled sheet path, source/scaled sizes, sprite count, and each sprite's file,
    bbox, area, and size.
    """
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
    """Create a 3x3 transparent tileset guide PNG for drawing or filling tile art.

    Use this when the user asks for a tileset guide, a 3x3 tile template, or a
    guide image that can later be sliced into top/center/bottom/corner tiles.
    If output_path is omitted, the PNG is written under
    /Users/seungjae/Codes/ai-asset-prep/outputs with a unique filename. If
    metadata_path is provided, tile box metadata is also written as JSON.

    tile_size is the square tile size in pixels. prefix is used only for naming
    outputs. gap and margin default to 0 for flush tile layouts. line_width
    controls the red guide lines. Returns output path, guide size, tile_size,
    gap, margin, line_width, and the 9 tile boxes.
    """
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
    """Slice a completed 3x3 tileset guide image into individual tile PNG files.

    Use this after make_tileset_guide when the user has filled or edited the
    guide image and wants separate tile PNGs. The tile_size, gap, and margin
    values must match the guide settings used to create the image. Prefer
    absolute input_path and output_dir values. If output_dir is omitted, files
    are written under /Users/seungjae/Codes/ai-asset-prep/outputs in a tile
    directory. If metadata_path is provided, slice metadata is also written as
    JSON.

    prefix controls output filenames such as ground_top_left.png. Set
    skip_transparent_tiles to true to omit fully transparent tiles. Returns
    output_dir, source size, expected guide size, tile count, and each tile's
    code, output file path, description, and size.
    """
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
