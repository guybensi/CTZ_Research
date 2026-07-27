#!/usr/bin/env python3
"""Consolidate paired moments/scalars (and optional images) into one dataset root.

This utility creates symlinks by default so data is not duplicated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Consolidate paired dataset folders into one root")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            str(repo_root / "data" / "test"),
            str(repo_root / "data" / "legacy_code" / "old_codes" / "test"),
            str(repo_root / "data" / "March_2014N15_momentsNscalars" / "2014"),
            str(repo_root / "data" / "March_2014N15_momentsNscalars" / "2015"),
        ],
        help="Dataset roots that may contain moments/scalars(/images)",
    )
    parser.add_argument(
        "--output-root",
        default=str(repo_root / "data" / "all_paired"),
        help="Output root for consolidated links/files",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks",
    )
    return parser.parse_args()


def ensure_dirs(base: Path) -> None:
    for name in ("moments", "scalars", "images"):
        (base / name).mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    if dst.exists():
        return
    if copy_files:
        dst.write_bytes(src.read_bytes())
    else:
        dst.symlink_to(src.resolve())


def gather_pairs(roots: Iterable[Path]) -> Tuple[int, int, int]:
    total_moments = 0
    total_scalars = 0
    total_images = 0
    for root in roots:
        if (root / "moments").is_dir():
            total_moments += len(list((root / "moments").glob("*.npz")))
        if (root / "scalars").is_dir():
            total_scalars += len(list((root / "scalars").glob("*.npz")))
        if (root / "images").is_dir():
            total_images += len(list((root / "images").glob("*.npz")))
    return total_moments, total_scalars, total_images


def main() -> int:
    args = parse_args()
    roots = [Path(p).resolve() for p in args.roots]
    output_root = Path(args.output_root).resolve()

    ensure_dirs(output_root)

    moment_added = 0
    scalar_added = 0
    image_added = 0

    for root in roots:
        moments_dir = root / "moments"
        scalars_dir = root / "scalars"
        images_dir = root / "images"

        if moments_dir.is_dir() and scalars_dir.is_dir():
            for moment_file in sorted(moments_dir.glob("*.npz")):
                scalar_file = scalars_dir / moment_file.name
                if not scalar_file.exists():
                    continue

                out_moment = output_root / "moments" / moment_file.name
                out_scalar = output_root / "scalars" / scalar_file.name

                before_m = out_moment.exists()
                before_s = out_scalar.exists()

                link_or_copy(moment_file, out_moment, args.copy)
                link_or_copy(scalar_file, out_scalar, args.copy)

                if not before_m and out_moment.exists():
                    moment_added += 1
                if not before_s and out_scalar.exists():
                    scalar_added += 1

        if images_dir.is_dir():
            for image_file in sorted(images_dir.glob("*.npz")):
                out_image = output_root / "images" / image_file.name
                before_i = out_image.exists()
                link_or_copy(image_file, out_image, args.copy)
                if not before_i and out_image.exists():
                    image_added += 1

    total_m, total_s, total_i = gather_pairs(roots)
    print(f"Input candidates - moments: {total_m}, scalars: {total_s}, images: {total_i}")
    print(f"Added to {output_root} - moments: {moment_added}, scalars: {scalar_added}, images: {image_added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
