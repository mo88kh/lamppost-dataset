"""
Tile generator
==============

Usage
-----
python -m src.tiler                          # uses values from config.yaml
python -m src.tiler --input path --output path --tile 128
"""

from __future__ import annotations
import argparse, os, sys, yaml, cv2
from pathlib import Path
from typing import Iterable, Tuple

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def iter_image_files(folder: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for p in folder.iterdir():
        if p.suffix.lower() in exts and p.is_file():
            yield p

def tile_image(img_path: Path, out_dir: Path, tile_size: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)  # <---- Add this line

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Cannot read {img_path}")
        return 0

    h, w = img.shape[:2]
    rows, cols = h // tile_size, w // tile_size
    tile_count = 0

    for r in range(rows):
        for c in range(cols):
            y0, y1 = r*tile_size, (r+1)*tile_size
            x0, x1 = c*tile_size, (c+1)*tile_size
            tile = img[y0:y1, x0:x1]
            name = f"{img_path.stem}_r{r:02d}_c{c:02d}.png"
            cv2.imwrite(str(out_dir / name), tile)
            tile_count += 1

    return tile_count


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text())

    parser = argparse.ArgumentParser(description="Generate fixed-size tiles.")
    parser.add_argument("--input",  "-i", default=cfg["RAW_DIR"],
                        help="Folder with raw images")
    parser.add_argument("--output", "-o", default=cfg["TILE_DIR"],
                        help="Destination folder for tiles")
    parser.add_argument("--tile",   "-t", type=int, default=cfg["TILE_SIZE"],
                        help="Tile size in pixels (square)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Do not re-export if tile already exists")
    args = parser.parse_args(argv)

    in_dir  = Path(args.input).expanduser()
    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for img_path in iter_image_files(in_dir):
        if args.skip_existing and any(out_dir.glob(f"{img_path.stem}_*.png")):
            continue
        count = tile_image(img_path, out_dir, args.tile)
        print(f"[✓] {img_path.name}  →  {count} tiles")
        total += count

    print(f"\nFinished. {total} tiles written to {out_dir}")

if __name__ == "__main__":
    main(sys.argv[1:])
