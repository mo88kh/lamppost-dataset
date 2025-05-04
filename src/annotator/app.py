"""
Annotate lamppost dataset.

Run a single image:
    python -m src.annotator.app --image data/raw/img01.jpg

Run a whole folder:
    python -m src.annotator.app --folder data/raw  --skip-done
"""
from __future__ import annotations
import argparse, sys, json
from pathlib import Path

import cv2, numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore

from . import io, utils


# ─────────────────────────────  Canvas  ────────────────────────────────────
class GridCanvas(QtWidgets.QLabel):
    def __init__(self, img_path: Path):
        super().__init__()
        self.img_path = img_path

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(img_path)
        self.img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        self.rows, self.cols = utils.image_to_grid(self.img_rgb.shape[:2])
        self.labels = np.full((self.rows, self.cols), -1, dtype=np.int8)  # pre-fill background

        # size hint for scroll-area
        h, w = self.img_rgb.shape[:2]
        self.setFixedSize(w, h)

        # existing label file?
        self.label_path = Path("data/labels") / f"{img_path.stem}.json"
        self.label_path.parent.mkdir(parents=True, exist_ok=True)
        if self.label_path.exists():
            self.labels = io.load_labels(self.label_path)

        self.setPixmap(self._grid_pixmap())
        self.setMouseTracking(True)

    # permanent grid
    def _grid_pixmap(self) -> QtGui.QPixmap:
        overlay = self.img_rgb.copy()
        h, w = overlay.shape[:2]
        for x in range(0, w, utils.TILE_SIZE):
            cv2.line(overlay, (x, 0), (x, h), (0, 255, 0), 1)
        for y in range(0, h, utils.TILE_SIZE):
            cv2.line(overlay, (0, y), (w, y), (0, 255, 0), 1)
        qimg = QtGui.QImage(
            overlay.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888
        )
        return QtGui.QPixmap.fromImage(qimg)

    # ── mouse ──
    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        x, y = int(ev.position().x()), int(ev.position().y())
        r, c = utils.pixel_to_cell(x, y)
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.labels[r, c] = 1
        elif ev.button() == QtCore.Qt.MouseButton.RightButton:
            self.labels[r, c] = -1
        elif ev.button() == QtCore.Qt.MouseButton.MiddleButton:
            self.labels[r, c] = 0
        self.update()

    # draw translucent fills
    def paintEvent(self, e):
        super().paintEvent(e)
        p = QtGui.QPainter(self)
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.labels[r, c]
                if val == 0:
                    continue
                clr = QtGui.QColor(0, 255, 0, 100) if val == 1 else QtGui.QColor(255, 0, 0, 100)
                p.fillRect(c * utils.TILE_SIZE, r * utils.TILE_SIZE,
                           utils.TILE_SIZE, utils.TILE_SIZE, clr)

    def save(self):
        io.save_labels(self.label_path, self.labels)

    # quick helper to know if at least one ROI exists
    def has_roi(self) -> bool:
        return bool((self.labels == 1).any())


# ─────────────────────────────  Main window  ───────────────────────────────
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, img_list: list[Path], skip_done=False):
        super().__init__()
        self.img_list = img_list
        self.skip_done = skip_done
        self.idx = 0
        self.canvas: GridCanvas | None = None

        self._build_toolbar()
        self._load_current()

    # build once
    def _build_toolbar(self):
        tb = self.addToolBar("Main")

        prev_act = QtGui.QAction("◀ Prev", self)
        prev_act.setShortcut("A")
        prev_act.triggered.connect(lambda: self.move(-1))
        tb.addAction(prev_act)

        next_act = QtGui.QAction("Next ▶", self)
        next_act.setShortcut("D")
        next_act.triggered.connect(lambda: self.move(+1))
        tb.addAction(next_act)

        save_act = QtGui.QAction("Save", self)
        save_act.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        save_act.triggered.connect(lambda: self.canvas.save())
        tb.addAction(save_act)

        self.statusBar().showMessage(
            "A/◀ Prev, D/▶ Next  |  L-click ROI, R-click background, M-click clear"
        )

    # load image at self.idx
    def _load_current(self):
        if self.canvas:
            self.canvas.save()        # autosave previous

        path = self.img_list[self.idx]
        self.canvas = GridCanvas(path)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(False)
        self.setCentralWidget(scroll)

        self.setWindowTitle(f"[{self.idx+1}/{len(self.img_list)}]  {path.name}")

    # move ±1 (+ wrap around) with optional skip-done
    def move(self, step: int):
        n = len(self.img_list)
        start = self.idx
        while True:
            self.idx = (self.idx + step) % n
            if not self.skip_done:
                break
            # if skip_done: advance until file has no ROI
            tmp_canvas = GridCanvas(self.img_list[self.idx])
            if not tmp_canvas.has_roi():
                break
            if self.idx == start:     # looped all => none left
                break
        self._load_current()


# ───────────────────────────────────────────────────────────────────────────
def collect_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("-i", "--image", help="single image file")
    grp.add_argument("-f", "--folder", help="folder with images")
    ap.add_argument("--skip-done", action="store_true",
                    help="skip pictures that already contain ROI")
    args = ap.parse_args()

    if args.image:
        img_list = [Path(args.image)]
    else:
        img_list = collect_images(Path(args.folder))
        if not img_list:
            print("No images found."); sys.exit(1)

    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(img_list, skip_done=args.skip_done)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
