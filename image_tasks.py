from __future__ import annotations

import base64
import json
import math
import os
import re
import zipfile
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from image_pipeline import (
    apply_padding_and_background,
    build_square_sprite_sheet,
    crop_to_alpha_bbox,
    extract_connected_components,
    image_to_png_bytes,
    load_image,
    remove_background_and_crop,
    resampling_filter,
    rotate_image,
    resize_to_target,
)


PROJECT_DIR = Path(__file__).parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

IMAGE_GENERATION_DEFAULT_MODEL = "gpt-image-1-mini"
IMAGE_GENERATION_DEFAULT_SIZE = "1024x1024"
IMAGE_GENERATION_DEFAULT_QUALITY = "low"
IMAGE_GENERATION_DEFAULT_FORMAT = "png"
IMAGE_GENERATION_QUALITY_OPTIONS = ("low", "medium", "high")
IMAGE_GENERATION_FORMAT_OPTIONS = ("png", "jpeg", "webp")
IMAGE_GENERATION_BACKGROUND_OPTIONS = ("opaque", "transparent")
IMAGE_GENERATION_SIZE_PATTERN = re.compile(r"^[1-9][0-9]*x[1-9][0-9]*$")
MODEL_NONE = "none"
MODEL_OPTIONS = {
    "u2net": "u2net - 기본",
    "isnet-general-use": "isnet-general-use - 일반 이미지 보존 우선",
    "isnet-anime": "isnet-anime - 일러스트/애니풍",
    MODEL_NONE: "모델 선택하지 않음 - 원본 기준",
}
RESIZE_MODE_OPTIONS = {
    "contain_center": "비율 유지 중앙",
    "contain_top": "비율 유지 상단",
    "contain_bottom": "비율 유지 하단",
    "contain_left": "비율 유지 좌측",
    "contain_right": "비율 유지 우측",
    "stretch": "늘려서 채우기",
}
MAX_OUTPUT_SIZE = 8192
PREVIEW_MAX_WIDTH = 420
PREVIEW_FRAME_HEIGHT = 180
PREVIEW_RENDER_SCALE = 2
SPRITE_SHEET_RESAMPLE_OPTIONS = {
    "nearest": "Nearest - 픽셀 유지",
    "lanczos": "Lanczos - 부드럽게",
}
SPRITE_SHEET_SCALE_MIN = 0.05
SPRITE_SHEET_SCALE_MAX = 16.0
SPRITE_SHEET_DEFAULT_SCALE = 1.0
TILESET_GUIDE_LAYOUT = (
    (
        ("TL", "top_left", "위쪽 왼쪽 모서리"),
        ("T", "top", "위쪽 가운데"),
        ("TR", "top_right", "위쪽 오른쪽 모서리"),
    ),
    (
        ("L", "left_wall", "왼쪽 벽"),
        ("C", "center", "가운데 내부"),
        ("R", "right_wall", "오른쪽 벽"),
    ),
    (
        ("BL", "bottom_left", "아래쪽 왼쪽 모서리"),
        ("B", "bottom", "아래쪽 가운데"),
        ("BR", "bottom_right", "아래쪽 오른쪽 모서리"),
    ),
)


try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv(dotenv_path: Any = None, *_args: Any, **_kwargs: Any) -> bool:
        if dotenv_path is None:
            return False

        path = Path(dotenv_path)
        if not path.exists():
            return False

        loaded = False
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue

            key, value = stripped.split("=", 1)
            normalized_key = key.strip()
            normalized_value = value.strip().strip("'\"")
            if normalized_key and normalized_key not in os.environ:
                os.environ[normalized_key] = normalized_value
                loaded = True

        return loaded


def safe_stem(original_name: str) -> str:
    stem = Path(original_name).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "image"


def slugify(text: str, max_len: int = 80) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9가-힣\s_-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text).strip("_")
    return text[:max_len] if len(text) > max_len else text


def safe_output_name(original_name: str, width: int, height: int) -> str:
    return f"{safe_stem(original_name)}_output_{width}x{height}.png"


def unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"저장 가능한 파일명을 만들지 못했습니다: {path.name}")


def write_unique_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_path = unique_output_path(path)
    output_path.write_bytes(data)
    return output_path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return write_unique_bytes(path, encoded)


def normalize_generation_size(value: str) -> str:
    normalized = value.lower().strip()
    if not IMAGE_GENERATION_SIZE_PATTERN.match(normalized):
        raise ValueError("size must use WIDTHxHEIGHT format, for example 1024x1024.")
    return normalized


def normalize_generation_output_format(value: str) -> str:
    normalized = value.lower().strip()
    if normalized not in IMAGE_GENERATION_FORMAT_OPTIONS:
        raise ValueError(
            "output_format must be one of: "
            + ", ".join(IMAGE_GENERATION_FORMAT_OPTIONS)
        )
    return normalized


def normalize_generation_quality(value: str) -> str:
    normalized = value.lower().strip()
    if normalized not in IMAGE_GENERATION_QUALITY_OPTIONS:
        raise ValueError(
            "quality must be one of: " + ", ".join(IMAGE_GENERATION_QUALITY_OPTIONS)
        )
    return normalized


def normalize_generation_background(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    normalized = value.lower().strip()
    if normalized not in IMAGE_GENERATION_BACKGROUND_OPTIONS:
        raise ValueError(
            "background must be one of: "
            + ", ".join(IMAGE_GENERATION_BACKGROUND_OPTIONS)
        )
    return normalized


def load_openai_api_key() -> str:
    load_dotenv(PROJECT_DIR / ".env")
    load_dotenv(Path.cwd() / ".env")
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in env or .env.")
    return api_key


def image_generation_output_path(
    prompt: str,
    output_path: str | Path | None = None,
    *,
    name: str | None = None,
    output_format: str = IMAGE_GENERATION_DEFAULT_FORMAT,
) -> Path:
    output_format = normalize_generation_output_format(output_format)

    if output_path:
        path = Path(output_path).expanduser()
        if path.suffix:
            return path
        stem = slugify(name or prompt[:60]) or "image"
        return path / f"{stem}.{output_format}"

    stem = slugify(name or prompt[:60])
    if not stem:
        raise ValueError("Output name became empty after sanitizing.")
    return OUTPUT_DIR / f"{stem}.{output_format}"


def generate_image_bytes(
    prompt: str,
    *,
    model: str = IMAGE_GENERATION_DEFAULT_MODEL,
    size: str = IMAGE_GENERATION_DEFAULT_SIZE,
    quality: str = IMAGE_GENERATION_DEFAULT_QUALITY,
    output_format: str = IMAGE_GENERATION_DEFAULT_FORMAT,
    background: str | None = None,
) -> bytes:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt is required.")

    size = normalize_generation_size(size)
    quality = normalize_generation_quality(quality)
    output_format = normalize_generation_output_format(output_format)
    background = normalize_generation_background(background)

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package is missing. Install it with: pip install openai"
        ) from exc

    kwargs = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "output_format": output_format,
    }
    if background is not None:
        kwargs["background"] = background

    client = OpenAI(api_key=load_openai_api_key())
    result = client.images.generate(**kwargs)
    if not result.data or not result.data[0].b64_json:
        raise RuntimeError("No image payload returned from images.generate.")

    return base64.b64decode(result.data[0].b64_json)


def image_generation_metadata(
    *,
    prompt: str,
    output: Path,
    model: str,
    size: str,
    quality: str,
    output_format: str,
    background: str | None,
) -> dict[str, Any]:
    return {
        "output": str(output),
        "model": model,
        "size": size,
        "quality": quality,
        "output_format": output_format,
        "background": background,
        "prompt": prompt,
        "prompt_summary": prompt[:160],
    }


def shrink_for_preview(image: Image.Image) -> Image.Image:
    preview = image.copy().convert("RGBA")
    preview.thumbnail(
        (
            PREVIEW_MAX_WIDTH * PREVIEW_RENDER_SCALE,
            PREVIEW_FRAME_HEIGHT * PREVIEW_RENDER_SCALE,
        ),
        Image.Resampling.LANCZOS,
    )
    return preview


def checkerboard_preview(image: Image.Image, square_size: int = 16) -> bytes:
    image = shrink_for_preview(image)
    width, height = image.size
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))

    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            if (x // square_size + y // square_size) % 2 == 0:
                color = (220, 220, 220, 255)
            else:
                color = (248, 248, 248, 255)
            background.paste(
                color,
                (x, y, min(x + square_size, width), min(y + square_size, height)),
            )

    background.alpha_composite(image)
    buffer = BytesIO()
    background.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def normalize_resize_mode(value: Any) -> str:
    if value in RESIZE_MODE_OPTIONS:
        return str(value)
    return "contain_center"


def normalize_sprite_sheet_scale(value: Any) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return SPRITE_SHEET_DEFAULT_SCALE
    if not math.isfinite(scale):
        return SPRITE_SHEET_DEFAULT_SCALE
    return min(max(scale, SPRITE_SHEET_SCALE_MIN), SPRITE_SHEET_SCALE_MAX)


def scaled_size(size: tuple[int, int], scale: float) -> tuple[int, int]:
    width, height = size
    return max(1, round(width * scale)), max(1, round(height * scale))


def resize_by_scale(
    image: Image.Image,
    scale: float,
    resampling: str,
    *,
    file_name: str,
) -> Image.Image:
    if scale <= 0:
        raise ValueError(f"{file_name}: 개별 이미지 scale은 0보다 커야 합니다.")

    target_size = scaled_size(image.size, scale)
    if target_size[0] > MAX_OUTPUT_SIZE or target_size[1] > MAX_OUTPUT_SIZE:
        raise ValueError(
            f"{file_name}: 개별 scale 적용 후 최대 크기 {MAX_OUTPUT_SIZE}px를 넘습니다: "
            f"{target_size[0]}x{target_size[1]}px"
        )
    if target_size == image.size:
        return image
    return image.resize(target_size, resampling_filter(resampling))


@lru_cache(maxsize=8)
def get_rembg_session(model_name: str):
    from rembg import new_session

    return new_session(model_name)


def process_alpha_crop(image_bytes: bytes, alpha_threshold: int = 16) -> dict[str, Any]:
    source = load_image(image_bytes)
    try:
        cropped, bbox = crop_to_alpha_bbox(source, alpha_threshold=alpha_threshold)
        fallback_reason = ""
    except ValueError as exc:
        cropped = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        bbox = (0, 0, 1, 1)
        fallback_reason = f"원본 crop 실패, 투명 이미지로 편집합니다: {exc}"

    return {
        "png_bytes": image_to_png_bytes(cropped),
        "preview_bytes": checkerboard_preview(cropped),
        "bbox": bbox,
        "source_size": source.size,
        "cropped_size": cropped.size,
        "removed_size": source.size,
        "fallback_reason": fallback_reason,
    }


def process_crop(
    image_bytes: bytes,
    alpha_threshold: int = 16,
    model_name: str = "u2net",
    preserve_interior: bool = True,
    post_process_mask: bool = True,
    session: Any | None = None,
) -> dict[str, Any]:
    if model_name == MODEL_NONE:
        return process_alpha_crop(image_bytes, alpha_threshold)

    fallback_reason = ""
    try:
        result = remove_background_and_crop(
            image_bytes,
            alpha_threshold=alpha_threshold,
            preserve_interior=preserve_interior,
            post_process_mask=post_process_mask,
            session=session if session is not None else get_rembg_session(model_name),
        )
    except ValueError as exc:
        source = load_image(image_bytes)
        try:
            cropped, bbox = crop_to_alpha_bbox(source, alpha_threshold=alpha_threshold)
        except ValueError:
            cropped = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            bbox = (0, 0, 1, 1)

        fallback_reason = f"모델 crop 실패, 원본 기준으로 편집합니다: {exc}"
        return {
            "png_bytes": image_to_png_bytes(cropped),
            "preview_bytes": checkerboard_preview(cropped),
            "bbox": bbox,
            "source_size": source.size,
            "cropped_size": cropped.size,
            "removed_size": source.size,
            "fallback_reason": fallback_reason,
        }

    return {
        "png_bytes": result.png_bytes,
        "preview_bytes": checkerboard_preview(result.cropped),
        "bbox": result.bbox,
        "source_size": result.source_size,
        "cropped_size": result.cropped.size,
        "removed_size": result.removed_background.size,
        "fallback_reason": fallback_reason,
    }


def build_output_image(
    cropped_png_bytes: bytes,
    width: int,
    height: int,
    resize_mode: str = "contain_center",
    padding_top: int = 0,
    padding_right: int = 0,
    padding_bottom: int = 0,
    padding_left: int = 0,
    transparent_background: bool = True,
    background_color: str = "#000000",
) -> dict[str, Any]:
    cropped = Image.open(BytesIO(cropped_png_bytes)).convert("RGBA")
    resized = resize_to_target(
        cropped,
        width=int(width),
        height=int(height),
        mode=normalize_resize_mode(resize_mode),
    )
    output = apply_padding_and_background(
        resized,
        padding_top=int(padding_top),
        padding_right=int(padding_right),
        padding_bottom=int(padding_bottom),
        padding_left=int(padding_left),
        transparent_background=transparent_background,
        background_color=background_color,
    )
    return {
        "png_bytes": image_to_png_bytes(output),
        "preview_bytes": checkerboard_preview(output),
        "resized_size": resized.size,
        "size": output.size,
    }


def build_combined_sprite_sheet(
    image_payloads: tuple[tuple[str, bytes, float], ...],
    scale_factor: float = 1.0,
    gap: int = 0,
    resampling: str = "nearest",
) -> dict[str, Any]:
    images: list[Image.Image] = []
    prepared_sources: list[tuple[str, tuple[int, int], float]] = []
    for file_name, image_bytes, source_scale in image_payloads:
        image = load_image(image_bytes)
        original_size = image.size
        image = resize_by_scale(
            image,
            float(source_scale),
            resampling,
            file_name=file_name,
        )
        images.append(image)
        prepared_sources.append((file_name, original_size, float(source_scale)))

    result = build_square_sprite_sheet(
        images,
        scale=float(scale_factor),
        gap=int(gap),
        resampling=resampling,
        max_dimension=MAX_OUTPUT_SIZE,
    )
    placements = []
    for (file_name, original_size, source_scale), placement in zip(
        prepared_sources,
        result.placements,
        strict=True,
    ):
        paste_x0, paste_y0, paste_x1, paste_y1 = placement.paste_box
        cell_x0, cell_y0, cell_x1, cell_y1 = placement.cell_box
        placements.append(
            {
                "index": placement.index + 1,
                "file": file_name,
                "source_scale": f"{source_scale:.2f}x",
                "source": (
                    f"{original_size[0]}x{original_size[1]} -> "
                    f"{placement.source_size[0]}x{placement.source_size[1]}"
                ),
                "cell": f"({cell_x0}, {cell_y0})-({cell_x1}, {cell_y1})",
                "paste": f"({paste_x0}, {paste_y0})-({paste_x1}, {paste_y1})",
            }
        )

    return {
        "png_bytes": result.png_bytes,
        "preview_bytes": checkerboard_preview(result.image),
        "unscaled_size": result.unscaled_size,
        "scaled_size": result.scaled_size,
        "cell_size": result.cell_size,
        "columns": result.columns,
        "rows": result.rows,
        "gap": result.gap,
        "scale": result.scale,
        "placements": placements,
    }


def recover_sprite_sheet(
    image_bytes: bytes,
    source_name: str,
    scale_factor: float = 0.5,
    alpha_threshold: int = 16,
    min_area: int = 16,
    resampling: str = "nearest",
) -> dict[str, Any]:
    source = load_image(image_bytes)
    scaled_size_value = (
        max(1, round(source.width * scale_factor)),
        max(1, round(source.height * scale_factor)),
    )
    if (
        scaled_size_value[0] > MAX_OUTPUT_SIZE
        or scaled_size_value[1] > MAX_OUTPUT_SIZE
    ):
        raise ValueError(
            f"스케일 적용 후 최대 크기 {MAX_OUTPUT_SIZE}px를 넘습니다: "
            f"{scaled_size_value[0]}x{scaled_size_value[1]}px"
        )

    scaled = source
    if scaled.size != scaled_size_value:
        scaled = source.resize(scaled_size_value, resampling_filter(resampling))

    components = extract_connected_components(
        scaled,
        alpha_threshold=int(alpha_threshold),
        min_area=int(min_area),
    )
    prefix = safe_stem(source_name)
    sprites = []
    for index, component in enumerate(components, start=1):
        file_name = f"{prefix}_sprite_{index:02d}.png"
        png_bytes = component.png_bytes
        sprites.append(
            {
                "index": index,
                "file": file_name,
                "bbox": component.bbox,
                "area": component.area,
                "size": component.image.size,
                "png_bytes": png_bytes,
                "preview_bytes": checkerboard_preview(component.image),
            }
        )

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for sprite in sprites:
            archive.writestr(sprite["file"], sprite["png_bytes"])

    return {
        "scaled_png_bytes": image_to_png_bytes(scaled),
        "scaled_preview_bytes": checkerboard_preview(scaled),
        "source_size": source.size,
        "scaled_size": scaled.size,
        "sprites": sprites,
        "zip_bytes": zip_buffer.getvalue(),
    }


def tileset_guide_size(tile_size: int, gap: int = 0, margin: int = 0) -> tuple[int, int]:
    columns = max(len(row) for row in TILESET_GUIDE_LAYOUT)
    rows = len(TILESET_GUIDE_LAYOUT)
    return (
        margin * 2 + columns * tile_size + (columns - 1) * gap,
        margin * 2 + rows * tile_size + (rows - 1) * gap,
    )


def tileset_guide_specs(
    tile_size: int,
    gap: int = 0,
    margin: int = 0,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for row_index, row in enumerate(TILESET_GUIDE_LAYOUT):
        for column_index, (code, file_stem, description) in enumerate(row):
            x0 = margin + column_index * (tile_size + gap)
            y0 = margin + row_index * (tile_size + gap)
            specs.append(
                {
                    "code": code,
                    "file_stem": file_stem,
                    "description": description,
                    "row": row_index + 1,
                    "column": column_index + 1,
                    "box": (x0, y0, x0 + tile_size, y0 + tile_size),
                }
            )
    return specs


def tileset_guide_table(
    tile_size: int,
    gap: int = 0,
    margin: int = 0,
) -> list[dict[str, Any]]:
    rows = []
    for spec in tileset_guide_specs(tile_size, gap, margin):
        x0, y0, x1, y1 = spec["box"]
        rows.append(
            {
                "코드": spec["code"],
                "파일": f"{spec['file_stem']}.png",
                "설명": spec["description"],
                "행": spec["row"],
                "열": spec["column"],
                "박스": f"({x0}, {y0})-({x1}, {y1})",
            }
        )
    return rows


def build_tileset_guide_background(
    tile_size: int,
    gap: int = 0,
    margin: int = 0,
    line_width: int = 1,
) -> dict[str, Any]:
    width, height = tileset_guide_size(tile_size, gap, margin)
    if width > MAX_OUTPUT_SIZE or height > MAX_OUTPUT_SIZE:
        raise ValueError(
            f"가이드 크기가 최대 {MAX_OUTPUT_SIZE}px를 넘습니다: {width}x{height}px"
        )

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    columns = max(len(row) for row in TILESET_GUIDE_LAYOUT)
    rows = len(TILESET_GUIDE_LAYOUT)
    line_color = (255, 75, 75, 230)

    for column in range(columns + 1):
        x = min(margin + column * (tile_size + gap), width - 1)
        draw.line((x, 0, x, height - 1), fill=line_color, width=line_width)

    for row in range(rows + 1):
        y = min(margin + row * (tile_size + gap), height - 1)
        draw.line((0, y, width - 1, y), fill=line_color, width=line_width)

    return {
        "png_bytes": image_to_png_bytes(image),
        "size": image.size,
        "tiles": tileset_guide_table(tile_size, gap, margin),
    }


def slice_tileset_guide_image(
    image_bytes: bytes,
    tile_size: int,
    gap: int = 0,
    margin: int = 0,
    file_prefix: str = "ground",
    skip_transparent_tiles: bool = False,
) -> dict[str, Any]:
    source = load_image(image_bytes)
    expected_width, expected_height = tileset_guide_size(tile_size, gap, margin)
    if source.width < expected_width or source.height < expected_height:
        raise ValueError(
            "입력 이미지가 현재 가이드 설정보다 작습니다: "
            f"입력={source.width}x{source.height}px, "
            f"필요={expected_width}x{expected_height}px"
        )

    prefix = safe_stem(file_prefix)
    tiles = []
    for spec in tileset_guide_specs(tile_size, gap, margin):
        tile = source.crop(spec["box"])
        if skip_transparent_tiles and tile.getchannel("A").getbbox() is None:
            continue

        file_name = f"{prefix}_{spec['file_stem']}.png"
        png_bytes = image_to_png_bytes(tile)
        tiles.append(
            {
                "code": spec["code"],
                "file": file_name,
                "description": spec["description"],
                "size": f"{tile.width}x{tile.height}",
                "png_bytes": png_bytes,
                "preview_bytes": checkerboard_preview(tile),
            }
        )

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for tile in tiles:
            archive.writestr(tile["file"], tile["png_bytes"])

    return {
        "tiles": tiles,
        "zip_bytes": zip_buffer.getvalue(),
        "source_size": source.size,
        "expected_size": (expected_width, expected_height),
    }


def _default_output_path(input_path: Path, suffix: str) -> Path:
    return OUTPUT_DIR / f"{safe_stem(input_path.name)}_{suffix}.png"


def _metadata_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(".json")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _json_ready(item)
            for key, item in value.items()
            if not str(key).endswith("_bytes")
        }
    return value


def generate_image(
    prompt: str,
    output_path: str | Path | None = None,
    *,
    name: str | None = None,
    model: str = IMAGE_GENERATION_DEFAULT_MODEL,
    size: str = IMAGE_GENERATION_DEFAULT_SIZE,
    quality: str = IMAGE_GENERATION_DEFAULT_QUALITY,
    output_format: str = IMAGE_GENERATION_DEFAULT_FORMAT,
    background: str | None = None,
) -> dict[str, Any]:
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("Prompt is required.")
    size = normalize_generation_size(size)
    quality = normalize_generation_quality(quality)
    output_format = normalize_generation_output_format(output_format)
    background = normalize_generation_background(background)
    target = image_generation_output_path(
        prompt,
        output_path,
        name=name,
        output_format=output_format,
    )
    image_bytes = generate_image_bytes(
        prompt,
        model=model,
        size=size,
        quality=quality,
        output_format=output_format,
        background=background,
    )
    written = write_unique_bytes(target, image_bytes)
    return _json_ready(
        image_generation_metadata(
            prompt=prompt,
            output=written,
            model=model,
            size=size,
            quality=quality,
            output_format=output_format,
            background=background,
        )
    )


def crop_image(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    alpha_threshold: int = 16,
    model_name: str = "u2net",
    preserve_interior: bool = True,
    post_process_mask: bool = True,
) -> dict[str, Any]:
    input_path = Path(input_path)
    image_bytes = input_path.read_bytes()
    result = process_crop(
        image_bytes,
        alpha_threshold=alpha_threshold,
        model_name=model_name,
        preserve_interior=preserve_interior,
        post_process_mask=post_process_mask,
    )
    target = Path(output_path) if output_path else _default_output_path(input_path, "crop")
    written = write_unique_bytes(target, result["png_bytes"])
    metadata = {
        "input": str(input_path),
        "output": str(written),
        "bbox": result["bbox"],
        "source_size": result["source_size"],
        "cropped_size": result["cropped_size"],
        "removed_size": result["removed_size"],
        "model_name": model_name,
        "alpha_threshold": alpha_threshold,
        "fallback_reason": result["fallback_reason"],
    }
    return _json_ready(metadata)


def fit_image(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    width: int,
    height: int,
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
    input_path = Path(input_path)
    source = load_image(input_path.read_bytes())
    rotated = rotate_image(
        source,
        degrees=float(rotation_degrees),
        alpha_threshold=int(alpha_threshold),
    )
    result = build_output_image(
        image_to_png_bytes(rotated),
        width=int(width),
        height=int(height),
        resize_mode=resize_mode,
        padding_top=int(padding_top),
        padding_right=int(padding_right),
        padding_bottom=int(padding_bottom),
        padding_left=int(padding_left),
        transparent_background=transparent_background,
        background_color=background_color,
    )
    target = (
        Path(output_path)
        if output_path
        else OUTPUT_DIR / safe_output_name(input_path.name, result["size"][0], result["size"][1])
    )
    written = write_unique_bytes(target, result["png_bytes"])
    metadata = {
        "input": str(input_path),
        "output": str(written),
        "source_size": source.size,
        "rotated_size": rotated.size,
        "resized_size": result["resized_size"],
        "size": result["size"],
        "resize_mode": normalize_resize_mode(resize_mode),
        "rotation_degrees": rotation_degrees,
        "padding": {
            "top": padding_top,
            "right": padding_right,
            "bottom": padding_bottom,
            "left": padding_left,
        },
        "transparent_background": transparent_background,
        "background_color": background_color,
    }
    return _json_ready(metadata)


def make_sprite_sheet(
    input_paths: list[str | Path],
    output_path: str | Path | None = None,
    *,
    metadata_path: str | Path | None = None,
    source_scales: list[float] | None = None,
    scale: float = 1.0,
    first_width: int | None = None,
    gap: int = 0,
    resampling: str = "nearest",
) -> dict[str, Any]:
    paths = [Path(path) for path in input_paths]
    if not paths:
        raise ValueError("이미지가 하나 이상 필요합니다.")

    if source_scales is None or not source_scales:
        resolved_scales = [SPRITE_SHEET_DEFAULT_SCALE] * len(paths)
    elif len(source_scales) == 1:
        resolved_scales = [float(source_scales[0])] * len(paths)
    elif len(source_scales) == len(paths):
        resolved_scales = [float(value) for value in source_scales]
    else:
        raise ValueError("--source-scale 값은 1개 또는 입력 이미지 개수와 같아야 합니다.")

    payloads = tuple(
        (path.name, path.read_bytes(), resolved_scales[index])
        for index, path in enumerate(paths)
    )
    scale_factor = float(scale)
    if first_width is not None:
        first_source = load_image(payloads[0][1])
        first_scaled_width, _ = scaled_size(first_source.size, resolved_scales[0])
        scale_factor = int(first_width) / first_scaled_width

    sheet = build_combined_sprite_sheet(
        payloads,
        scale_factor=scale_factor,
        gap=int(gap),
        resampling=resampling,
    )
    output_target = (
        Path(output_path)
        if output_path
        else OUTPUT_DIR / safe_output_name("sprite_sheet.png", sheet["scaled_size"][0], sheet["scaled_size"][1])
    )
    written = write_unique_bytes(output_target, sheet["png_bytes"])
    metadata = {
        "inputs": [str(path) for path in paths],
        "output": str(written),
        "unscaled_size": sheet["unscaled_size"],
        "scaled_size": sheet["scaled_size"],
        "cell_size": sheet["cell_size"],
        "columns": sheet["columns"],
        "rows": sheet["rows"],
        "gap": sheet["gap"],
        "scale": sheet["scale"],
        "resampling": resampling,
        "source_scales": resolved_scales,
        "placements": sheet["placements"],
    }
    if metadata_path is not None:
        metadata["metadata"] = str(write_json(Path(metadata_path), _json_ready(metadata)))
    return _json_ready(metadata)


def recover_sprite_sheet_to_dir(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    scale: float = 0.5,
    alpha_threshold: int = 16,
    min_area: int = 16,
    resampling: str = "nearest",
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR / f"{safe_stem(input_path.name)}_sprites"
    output_dir.mkdir(parents=True, exist_ok=True)
    recovered = recover_sprite_sheet(
        input_path.read_bytes(),
        input_path.name,
        scale_factor=float(scale),
        alpha_threshold=int(alpha_threshold),
        min_area=int(min_area),
        resampling=resampling,
    )
    scaled_name = (
        f"{safe_stem(input_path.name)}_scaled_"
        f"{recovered['scaled_size'][0]}x{recovered['scaled_size'][1]}.png"
    )
    scaled_path = write_unique_bytes(output_dir / scaled_name, recovered["scaled_png_bytes"])
    sprite_entries = []
    for sprite in recovered["sprites"]:
        sprite_path = write_unique_bytes(output_dir / sprite["file"], sprite["png_bytes"])
        sprite_entries.append(
            {
                "index": sprite["index"],
                "file": str(sprite_path),
                "bbox": sprite["bbox"],
                "area": sprite["area"],
                "size": sprite["size"],
            }
        )

    metadata = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "scaled_sheet": str(scaled_path),
        "source_size": recovered["source_size"],
        "scaled_size": recovered["scaled_size"],
        "scale": scale,
        "alpha_threshold": alpha_threshold,
        "min_area": min_area,
        "resampling": resampling,
        "sprite_count": len(sprite_entries),
        "sprites": sprite_entries,
    }
    if metadata_path is not None:
        metadata["metadata"] = str(write_json(Path(metadata_path), _json_ready(metadata)))
    return _json_ready(metadata)


def make_tileset_guide(
    output_path: str | Path | None = None,
    *,
    tile_size: int = 64,
    prefix: str = "ground",
    gap: int = 0,
    margin: int = 0,
    line_width: int = 1,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    guide = build_tileset_guide_background(
        tile_size=int(tile_size),
        gap=int(gap),
        margin=int(margin),
        line_width=int(line_width),
    )
    target = (
        Path(output_path)
        if output_path
        else OUTPUT_DIR / f"{safe_stem(prefix)}_tileset_guide_{int(tile_size)}px.png"
    )
    written = write_unique_bytes(target, guide["png_bytes"])
    metadata = {
        "output": str(written),
        "size": guide["size"],
        "tile_size": tile_size,
        "gap": gap,
        "margin": margin,
        "line_width": line_width,
        "tiles": guide["tiles"],
    }
    if metadata_path is not None:
        metadata["metadata"] = str(write_json(Path(metadata_path), _json_ready(metadata)))
    return _json_ready(metadata)


def slice_tileset(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    tile_size: int = 64,
    prefix: str = "ground",
    gap: int = 0,
    margin: int = 0,
    skip_transparent_tiles: bool = False,
    metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR / f"{safe_stem(prefix)}_tiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    sliced = slice_tileset_guide_image(
        input_path.read_bytes(),
        tile_size=int(tile_size),
        gap=int(gap),
        margin=int(margin),
        file_prefix=prefix,
        skip_transparent_tiles=skip_transparent_tiles,
    )
    tile_entries = []
    for tile in sliced["tiles"]:
        tile_path = write_unique_bytes(output_dir / tile["file"], tile["png_bytes"])
        tile_entries.append(
            {
                "code": tile["code"],
                "file": str(tile_path),
                "description": tile["description"],
                "size": tile["size"],
            }
        )

    metadata = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "source_size": sliced["source_size"],
        "expected_size": sliced["expected_size"],
        "tile_size": tile_size,
        "gap": gap,
        "margin": margin,
        "skip_transparent_tiles": skip_transparent_tiles,
        "tile_count": len(tile_entries),
        "tiles": tile_entries,
    }
    if metadata_path is not None:
        metadata["metadata"] = str(write_json(Path(metadata_path), _json_ready(metadata)))
    return _json_ready(metadata)
