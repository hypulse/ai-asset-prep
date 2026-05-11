from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import image_tasks


def print_result(result: dict[str, Any]) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2))


def add_common_metadata_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--metadata",
        type=Path,
        help="처리 결과 metadata JSON을 저장할 경로입니다.",
    )


def add_padding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--padding", type=int, default=None, help="상하좌우 공통 padding.")
    parser.add_argument("--padding-top", type=int, default=0)
    parser.add_argument("--padding-right", type=int, default=0)
    parser.add_argument("--padding-bottom", type=int, default=0)
    parser.add_argument("--padding-left", type=int, default=0)


def resolved_padding(args: argparse.Namespace) -> dict[str, int]:
    if args.padding is None:
        return {
            "padding_top": args.padding_top,
            "padding_right": args.padding_right,
            "padding_bottom": args.padding_bottom,
            "padding_left": args.padding_left,
        }
    return {
        "padding_top": args.padding,
        "padding_right": args.padding,
        "padding_bottom": args.padding,
        "padding_left": args.padding,
    }


def handle_crop(args: argparse.Namespace) -> dict[str, Any]:
    return image_tasks.crop_image(
        args.input,
        args.output,
        alpha_threshold=args.alpha,
        model_name=args.model,
        preserve_interior=not args.no_preserve_interior,
        post_process_mask=not args.no_post_process_mask,
    )


def handle_fit(args: argparse.Namespace) -> dict[str, Any]:
    return image_tasks.fit_image(
        args.input,
        args.output,
        width=args.width,
        height=args.height,
        resize_mode=args.resize_mode,
        rotation_degrees=args.rotate,
        alpha_threshold=args.alpha,
        transparent_background=not args.opaque_background,
        background_color=args.background_color,
        **resolved_padding(args),
    )


def handle_sheet_make(args: argparse.Namespace) -> dict[str, Any]:
    return image_tasks.make_sprite_sheet(
        args.inputs,
        args.output,
        metadata_path=args.metadata,
        source_scales=args.source_scale,
        scale=args.scale,
        first_width=args.first_width,
        gap=args.gap,
        resampling=args.resampling,
    )


def handle_sheet_recover(args: argparse.Namespace) -> dict[str, Any]:
    return image_tasks.recover_sprite_sheet_to_dir(
        args.input,
        args.output_dir,
        scale=args.scale,
        alpha_threshold=args.alpha,
        min_area=args.min_area,
        resampling=args.resampling,
        metadata_path=args.metadata,
    )


def handle_tiles_guide(args: argparse.Namespace) -> dict[str, Any]:
    return image_tasks.make_tileset_guide(
        args.output,
        tile_size=args.tile_size,
        prefix=args.prefix,
        gap=args.gap,
        margin=args.margin,
        line_width=args.line_width,
        metadata_path=args.metadata,
    )


def handle_tiles_slice(args: argparse.Namespace) -> dict[str, Any]:
    return image_tasks.slice_tileset(
        args.input,
        args.output_dir,
        tile_size=args.tile_size,
        prefix=args.prefix,
        gap=args.gap,
        margin=args.margin,
        skip_transparent_tiles=args.skip_transparent_tiles,
        metadata_path=args.metadata,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-cut-fit",
        description="Image Cut Fit image processing CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    crop = subparsers.add_parser("crop", help="배경 제거 또는 alpha 기준 crop.")
    crop.add_argument("input", type=Path)
    crop.add_argument("--output", "-o", type=Path)
    crop.add_argument("--alpha", type=int, default=16)
    crop.add_argument(
        "--model",
        choices=list(image_tasks.MODEL_OPTIONS.keys()),
        default="u2net",
        help="배경 제거 모델입니다. none이면 alpha crop만 수행합니다.",
    )
    crop.add_argument("--no-preserve-interior", action="store_true")
    crop.add_argument("--no-post-process-mask", action="store_true")
    crop.set_defaults(handler=handle_crop)

    fit = subparsers.add_parser("fit", help="이미지를 회전/리사이즈/padding 처리.")
    fit.add_argument("input", type=Path)
    fit.add_argument("--output", "-o", type=Path)
    fit.add_argument("--width", type=int, required=True)
    fit.add_argument("--height", type=int, required=True)
    fit.add_argument(
        "--resize-mode",
        choices=list(image_tasks.RESIZE_MODE_OPTIONS.keys()),
        default="contain_center",
    )
    fit.add_argument("--rotate", type=float, default=0)
    fit.add_argument("--alpha", type=int, default=16)
    add_padding_args(fit)
    fit.add_argument("--opaque-background", action="store_true")
    fit.add_argument("--background-color", default="#000000")
    fit.set_defaults(handler=handle_fit)

    sheet = subparsers.add_parser("sheet", help="스프라이트 시트 생성/복구.")
    sheet_subparsers = sheet.add_subparsers(dest="sheet_command", required=True)

    sheet_make = sheet_subparsers.add_parser("make", help="여러 이미지를 한 시트로 합칩니다.")
    sheet_make.add_argument("inputs", nargs="+", type=Path)
    sheet_make.add_argument("--output", "-o", type=Path)
    add_common_metadata_arg(sheet_make)
    sheet_make.add_argument(
        "--source-scale",
        type=float,
        action="append",
        help="입력별 사전 scale입니다. 1개면 모든 입력에 적용합니다.",
    )
    sheet_make.add_argument("--scale", type=float, default=1.0)
    sheet_make.add_argument(
        "--first-width",
        type=int,
        help="첫 이미지의 사전 scale 적용 후 목표 가로폭으로 전체 scale을 계산합니다.",
    )
    sheet_make.add_argument("--gap", type=int, default=0)
    sheet_make.add_argument(
        "--resampling",
        choices=list(image_tasks.SPRITE_SHEET_RESAMPLE_OPTIONS.keys()),
        default="nearest",
    )
    sheet_make.set_defaults(handler=handle_sheet_make)

    sheet_recover = sheet_subparsers.add_parser(
        "recover",
        help="투명 배경 시트에서 개별 스프라이트를 분리합니다.",
    )
    sheet_recover.add_argument("input", type=Path)
    sheet_recover.add_argument("--output-dir", type=Path)
    add_common_metadata_arg(sheet_recover)
    sheet_recover.add_argument("--scale", type=float, default=0.5)
    sheet_recover.add_argument("--alpha", type=int, default=16)
    sheet_recover.add_argument("--min-area", type=int, default=16)
    sheet_recover.add_argument(
        "--resampling",
        choices=list(image_tasks.SPRITE_SHEET_RESAMPLE_OPTIONS.keys()),
        default="nearest",
    )
    sheet_recover.set_defaults(handler=handle_sheet_recover)

    tiles = subparsers.add_parser("tiles", help="타일셋 가이드 생성/추출.")
    tiles_subparsers = tiles.add_subparsers(dest="tiles_command", required=True)

    tiles_guide = tiles_subparsers.add_parser("guide", help="3x3 타일셋 가이드를 만듭니다.")
    tiles_guide.add_argument("--output", "-o", type=Path)
    add_common_metadata_arg(tiles_guide)
    tiles_guide.add_argument("--tile-size", type=int, default=64)
    tiles_guide.add_argument("--prefix", default="ground")
    tiles_guide.add_argument("--gap", type=int, default=0)
    tiles_guide.add_argument("--margin", type=int, default=0)
    tiles_guide.add_argument("--line-width", type=int, default=1)
    tiles_guide.set_defaults(handler=handle_tiles_guide)

    tiles_slice = tiles_subparsers.add_parser("slice", help="가이드 이미지에서 타일을 자릅니다.")
    tiles_slice.add_argument("input", type=Path)
    tiles_slice.add_argument("--output-dir", type=Path)
    add_common_metadata_arg(tiles_slice)
    tiles_slice.add_argument("--tile-size", type=int, default=64)
    tiles_slice.add_argument("--prefix", default="ground")
    tiles_slice.add_argument("--gap", type=int, default=0)
    tiles_slice.add_argument("--margin", type=int, default=0)
    tiles_slice.add_argument("--skip-transparent-tiles", action="store_true")
    tiles_slice.set_defaults(handler=handle_tiles_slice)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print_result(args.handler(args))


if __name__ == "__main__":
    main()
