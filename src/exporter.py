#  src/exporter.py
from __future__ import annotations
import argparse, random, json, shutil
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm   # pip install tqdm for a nice progress bar

TILE = 100
RAW_DIR   = Path("data/raw")
LABEL_DIR = Path("data/labels")

def crop_tile(img, r: int, c: int):
    y0, y1 = r * TILE, (r + 1) * TILE
    x0, x1 = c * TILE, (c + 1) * TILE
    return img[y0:y1, x0:x1]

def export_dataset(out_dir: Path,
                   neg_ratio: int,
                   val_split: float,
                   test_split: float,
                   seed: int):

    rng = random.Random(seed)
    samples: list[tuple[np.ndarray, str]] = []   # (tile, "pos"/"neg")

    # ---- gather tiles ----------------------------------------------------
    for img_file in tqdm(sorted(RAW_DIR.glob("*.*g")), desc="scanning"):
        json_path = LABEL_DIR / f"{img_file.stem}.json"
        if not json_path.exists():
            continue            # unlabeled → skip

        img = cv2.imread(str(img_file))
        labels = np.array(json.loads(json_path.read_text())["labels"],
                          dtype=np.int8)

        # positives
        for r, c in zip(*np.where(labels == 1)):
            samples.append((crop_tile(img, r, c), "pos"))

        # negatives (randomly subsample)
        neg_cells = list(zip(*np.where(labels == -1)))
        rng.shuffle(neg_cells)
        for r, c in neg_cells[:neg_ratio * (labels == 1).sum()]:
            samples.append((crop_tile(img, r, c), "neg"))

    print(f"Total tiles: {len(samples)} "
          f"({sum(1 for _,cls in samples if cls=='pos')} pos / "
          f"{sum(1 for _,cls in samples if cls=='neg')} neg)")

    # ---- shuffle and split ----------------------------------------------
    rng.shuffle(samples)
    n = len(samples)
    n_test = int(n * test_split)
    n_val  = int(n * val_split)

    splits = (
        ("test",  samples[:n_test]),
        ("val",   samples[n_test:n_test+n_val]),
        ("train", samples[n_test+n_val:]),
    )

    # ---- write to disk ---------------------------------------------------
    for split, subset in splits:
        for cls in ("pos", "neg"):
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)

        for idx, (tile, cls) in enumerate(subset):
            fn = out_dir / split / cls / f"{cls}_{idx:06d}.png"
            cv2.imwrite(str(fn), tile)

    print("✅ export complete:", out_dir)

# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path,
                    help="output root folder (will be overwritten)")
    ap.add_argument("--neg-ratio", type=int, default=3,
                    help="how many negatives per positive tile")
    ap.add_argument("--val", type=float, default=0.2, help="val fraction")
    ap.add_argument("--test", type=float, default=0.1, help="test fraction")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)     # start fresh
    export_dataset(args.out, args.neg_ratio, args.val, args.test, args.seed)

if __name__ == "__main__":
    main()
