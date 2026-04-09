#!/usr/bin/env python3
"""
ScanThis - Document scanning and image manipulation application.
Replicates lookscanned.io functionality with a native macOS GUI.
"""

import sys
import os
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QSlider, QLabel, QPushButton, QFileDialog, QMessageBox,
    QComboBox, QGroupBox, QScrollArea, QProgressDialog,
    QStyleFactory, QSpinBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QDragEnterEvent, QDropEvent

from src.image_processor import (
    ProcessingParams, ColorMode, default_scanned_params,
    process_image, create_proxy_image, export_images_to_pdf
)


# ─────────────────────────────────────────────
#  Worker thread for async preview
# ─────────────────────────────────────────────
class PreviewWorker(QThread):
    preview_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.params: Optional[ProcessingParams] = None
        self.source_image: Optional[np.ndarray] = None
        self._abort = False

    def set_job(self, source: np.ndarray, params: ProcessingParams):
        self.source_image = source
        self.params = params
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        if self.source_image is None or self.params is None:
            return
        proxy = create_proxy_image(self.source_image, max_dim=1200)
        result = process_image(proxy, self.params, preview_scale=1.0)
        if not self._abort:
            self.preview_ready.emit(result)


# ─────────────────────────────────────────────
#  Full-screen drag overlay
# ─────────────────────────────────────────────
class DragOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(0, 50, 120, 180))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(self.rect())
        r = self.rect().adjusted(20, 20, -20, -20)
        painter.setPen(QColor(100, 180, 255, 220))
        painter.setBrush(QColor(0, 80, 160, 100))
        painter.drawRoundedRect(r, 20, 20)
        painter.setPen(QColor(255, 255, 255, 240))
        font = painter.font()
        font.setPointSize(28)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                         "Drop document here")

    def showEvent(self, event):
        self.raise_()


# ─────────────────────────────────────────────
#  Labeled slider widget
# ─────────────────────────────────────────────
class LabeledSlider(QWidget):
    def __init__(self, label: str, min_val: float, max_val: float,
                 default: float, suffix: str = "", parent=None):
        super().__init__(parent)
        self.suffix = suffix

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        header = QHBoxLayout()
        self.label_widget = QLabel(label)
        self.value_label = QLabel(f"{default:.2f}{suffix}")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        header.addWidget(self.label_widget)
        header.addWidget(self.value_label)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(min_val * 100))
        self.slider.setMaximum(int(max_val * 100))
        self.slider.setValue(int(default * 100))
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(int((max_val - min_val) * 10))

        layout.addLayout(header)
        layout.addWidget(self.slider)

        self.slider.valueChanged.connect(self._on_value_changed)

    def _on_value_changed(self, val):
        real = val / 100.0
        self.value_label.setText(f"{real:.2f}{self.suffix}")


# ─────────────────────────────────────────────
#  Image canvas
# ─────────────────────────────────────────────
class ImageCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setStyleSheet("background-color: #2b2b2b; border: 1px solid #444;")
        self.setText("Drop or open a document to begin")

    def set_image(self, img: np.ndarray):
        if img is None:
            self.setText("Drop or open a document to begin")
            self.setPixmap(QPixmap())
            return
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        if w <= self.width() and h <= self.height():
            self.setPixmap(pixmap)
        else:
            scaled = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
            self.setPixmap(scaled)


# ─────────────────────────────────────────────
#  Helper: load a single PDF page as BGR numpy image
# ─────────────────────────────────────────────
def _load_pdf_page(doc, page_idx: int, zoom: float = 2.0) -> np.ndarray:
    page = doc[page_idx]
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img_data = np.frombuffer(pix.samples, dtype=np.uint8)
    if pix.n == 4:
        img = img_data.reshape(pix.height, pix.width, 4)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        img = img_data.reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# ─────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ScanThis")
        self.setMinimumSize(1100, 735)

        # State: list of source images (one per page)
        self._source_images: list[np.ndarray] = []
        self._source_path: Optional[str] = None
        self._current_page: int = 0
        self._current_preview: Optional[np.ndarray] = None
        self._current_params = default_scanned_params()
        self._preview_timer = QTimer()

        self._setup_ui()
        self._connect_signals()
        self._apply_params_to_ui()

        self._preview_worker = PreviewWorker()
        self._preview_worker.preview_ready.connect(self._on_preview_ready)

        self._preview_timer.setSingleShot(True)
        self._preview_timer.timeout.connect(self._trigger_preview)

        self.setAcceptDrops(True)
        self._overlay = DragOverlay(self)
        self._overlay.hide()

    # ── UI Setup ──
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)

        # ── Page navigation ──
        self._page_nav = QGroupBox("Pages")
        nav_layout = QHBoxLayout(self._page_nav)

        self._prev_btn = QPushButton("◀")
        self._prev_btn.setMaximumWidth(40)
        self._page_spin = QSpinBox()
        self._page_spin.setMinimum(1)
        self._page_spin.setMaximum(1)
        self._page_spin.setMinimumWidth(60)
        self._page_spin.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self._next_btn = QPushButton("▶")
        self._next_btn.setMaximumWidth(40)
        self._page_label = QLabel(" of 1")

        nav_layout.addWidget(self._prev_btn)
        nav_layout.addWidget(self._page_spin)
        nav_layout.addWidget(self._page_label)
        nav_layout.addWidget(self._next_btn)

        self._page_nav.setVisible(False)  # hidden until doc is loaded

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)

        config_widget = QWidget()
        config_layout = QVBoxLayout(config_widget)

        title = QLabel("Document Settings")
        title.setStyleSheet("font-size: 16px; font-weight: bold; padding: 4px;")
        config_layout.addWidget(title)

        # Color Mode
        color_box = QGroupBox("Color Mode")
        color_layout = QVBoxLayout(color_box)
        self._color_mode_combo = QComboBox()
        self._color_mode_combo.addItems(["Color", "Grayscale", "Black & White"])
        color_layout.addWidget(self._color_mode_combo)
        config_layout.addWidget(color_box)

        # Rotation
        self._rotation_slider = LabeledSlider("Rotation (°)", -10, 10, 0.5, "°")
        config_layout.addWidget(self._rotation_slider)

        # Rotation Variance
        self._rot_var_slider = LabeledSlider("Rotation Variance (°)", 0, 5, 0, "°")
        config_layout.addWidget(self._rot_var_slider)

        # Brightness
        self._bright_slider = LabeledSlider("Brightness", -100, 100, 0, "")
        config_layout.addWidget(self._bright_slider)

        # Contrast
        self._contrast_slider = LabeledSlider("Contrast", -100, 100, 0, "")
        config_layout.addWidget(self._contrast_slider)

        # Blur
        self._blur_slider = LabeledSlider("Blur", 0, 10, 0, "px")
        config_layout.addWidget(self._blur_slider)

        # Noise
        self._noise_slider = LabeledSlider("Noise", 0, 50, 0, "")
        config_layout.addWidget(self._noise_slider)

        # Yellowing
        self._yellow_slider = LabeledSlider("Yellowing", 0, 1, 0, "")
        config_layout.addWidget(self._yellow_slider)

        # Resolution
        res_box = QGroupBox("Resolution (DPI)")
        res_layout = QVBoxLayout(res_box)
        self._res_combo = QComboBox()
        self._res_combo.addItems(["150", "200", "300", "400", "600"])
        self._res_combo.setCurrentText("300")
        res_layout.addWidget(self._res_combo)
        config_layout.addWidget(res_box)

        config_layout.addStretch()

        scroll.setWidget(config_widget)
        left_layout.addWidget(self._page_nav)
        left_layout.addWidget(scroll)

        # Buttons
        btn_layout = QHBoxLayout()
        open_btn = QPushButton("Open File")
        export_btn = QPushButton("Export PDF")
        reset_btn = QPushButton("Reset")
        scanned_btn = QPushButton("Scanned Look")
        btn_layout.addWidget(open_btn)
        btn_layout.addWidget(reset_btn)
        btn_layout.addWidget(scanned_btn)
        btn_layout.addWidget(export_btn)
        left_layout.addLayout(btn_layout)

        open_btn.clicked.connect(self._open_file)
        export_btn.clicked.connect(self._export_pdf)
        reset_btn.clicked.connect(self._reset_params)
        scanned_btn.clicked.connect(self._apply_scanned_look)

        self._canvas = ImageCanvas()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(self._canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([350, 750])

        main_layout.addWidget(splitter)

    def _connect_signals(self):
        self._rotation_slider.slider.valueChanged.connect(self._on_any_param_changed)
        self._rot_var_slider.slider.valueChanged.connect(self._on_any_param_changed)
        self._bright_slider.slider.valueChanged.connect(self._on_any_param_changed)
        self._contrast_slider.slider.valueChanged.connect(self._on_any_param_changed)
        self._blur_slider.slider.valueChanged.connect(self._on_any_param_changed)
        self._noise_slider.slider.valueChanged.connect(self._on_any_param_changed)
        self._yellow_slider.slider.valueChanged.connect(self._on_any_param_changed)
        self._res_combo.currentIndexChanged.connect(self._on_any_param_changed)
        self._color_mode_combo.currentIndexChanged.connect(self._on_any_param_changed)

        self._prev_btn.clicked.connect(self._prev_page)
        self._next_btn.clicked.connect(self._next_page)
        self._page_spin.valueChanged.connect(self._on_page_spin_changed)

    def _apply_params_to_ui(self):
        p = self._current_params
        color_map = [ColorMode.COLOR, ColorMode.GRAYSCALE, ColorMode.BLACK_WHITE]
        idx = color_map.index(p.color_mode)
        self._color_mode_combo.setCurrentIndex(idx)
        self._rotation_slider.slider.setValue(int(p.rotation * 100))
        self._rot_var_slider.slider.setValue(int(p.rotation_variance * 100))
        self._bright_slider.slider.setValue(int(p.brightness * 100))
        self._contrast_slider.slider.setValue(int(p.contrast * 100))
        self._blur_slider.slider.setValue(int(p.blur * 100))
        self._noise_slider.slider.setValue(int(p.noise * 100))
        self._yellow_slider.slider.setValue(int(p.yellowing * 100))
        self._res_combo.setCurrentText(str(p.resolution))

    def _read_params_from_ui(self):
        color_map = [ColorMode.COLOR, ColorMode.GRAYSCALE, ColorMode.BLACK_WHITE]
        self._current_params.color_mode = color_map[self._color_mode_combo.currentIndex()]
        self._current_params.rotation = self._rotation_slider.slider.value() / 100.0
        self._current_params.rotation_variance = self._rot_var_slider.slider.value() / 100.0
        self._current_params.brightness = self._bright_slider.slider.value() / 100.0
        self._current_params.contrast = self._contrast_slider.slider.value() / 100.0
        self._current_params.blur = self._blur_slider.slider.value() / 100.0
        self._current_params.noise = self._noise_slider.slider.value() / 100.0
        self._current_params.yellowing = self._yellow_slider.slider.value() / 100.0
        self._current_params.resolution = int(self._res_combo.currentText())

    def _on_any_param_changed(self):
        if not self._source_images:
            return
        self._read_params_from_ui()
        self._preview_timer.start(50)

    def _trigger_preview(self):
        if not self._source_images:
            return
        self._preview_worker.set_job(self._source_images[self._current_page],
                                       self._current_params)
        if not self._preview_worker.isRunning():
            self._preview_worker.start()

    def _on_preview_ready(self, result: np.ndarray):
        self._current_preview = result
        self._canvas.set_image(result)

    # ── Page navigation ──
    def _prev_page(self):
        if self._current_page > 0:
            self._current_page -= 1
            self._sync_page_ui()
            self._trigger_preview()

    def _next_page(self):
        if self._current_page < len(self._source_images) - 1:
            self._current_page += 1
            self._sync_page_ui()
            self._trigger_preview()

    def _on_page_spin_changed(self, value: int):
        idx = value - 1  # spin is 1-based
        if 0 <= idx < len(self._source_images):
            self._current_page = idx
            self._trigger_preview()

    def _sync_page_ui(self):
        """Sync spin + label to current page."""
        total = len(self._source_images)
        if total == 0:
            return
        self._page_spin.setValue(self._current_page + 1)
        self._page_label.setText(f" of {total}")
        self._prev_btn.setEnabled(self._current_page > 0)
        self._next_btn.setEnabled(self._current_page < total - 1)

    def _reset_params(self):
        self._current_params = ProcessingParams(
            color_mode=ColorMode.COLOR, rotation=0.5,
            rotation_variance=0.0, brightness=0.0,
            contrast=0.0, blur=0.0, noise=0.0,
            yellowing=0.0, resolution=300,
        )
        self._apply_params_to_ui()
        if self._source_images:
            self._trigger_preview()

    def _apply_scanned_look(self):
        self._current_params = default_scanned_params()
        self._apply_params_to_ui()
        if self._source_images:
            self._trigger_preview()

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image or PDF",
            "", "Documents (*.pdf *.png *.jpg *.jpeg *.tiff *.tif *.bmp);;All files (*)"
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str):
        ext = Path(path).suffix.lower()
        try:
            if ext == ".pdf":
                doc = fitz.open(path)
                total_pages = len(doc)
                self._source_images = [_load_pdf_page(doc, i) for i in range(total_pages)]
                doc.close()
            else:
                img = cv2.imread(path)
                if img is None:
                    QMessageBox.warning(self, "Error", f"Could not load image: {path}")
                    return
                self._source_images = [img]

            self._source_path = path
            self._current_page = 0
            self._page_nav.setVisible(True)
            self._page_spin.setMaximum(len(self._source_images))
            self._sync_page_ui()
            self._trigger_preview()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load file:\n{e}")

    def _get_default_export_path(self) -> str:
        if self._source_path:
            p = Path(self._source_path)
            default = p.parent / (p.stem + "_scanned.pdf")
        else:
            default = Path.home() / "Downloads" / "output_scanned.pdf"
        return str(default)

    def _export_pdf(self):
        if not self._source_images:
            QMessageBox.information(self, "No Document", "Please open a document first.")
            return

        default_path = self._get_default_export_path()

        if os.path.exists(default_path):
            reply = QMessageBox.question(
                self, "File Exists",
                f"'{os.path.basename(default_path)}' already exists.\nOverwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                path, _ = QFileDialog.getSaveFileName(
                    self, "Save PDF", default_path, "PDF Files (*.pdf)"
                )
                if not path:
                    return
                default_path = path

        self._read_params_from_ui()
        total = len(self._source_images)

        progress = QProgressDialog(f"Exporting {total} pages...", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setValue(0)
        QApplication.processEvents()

        try:
            processed_pages = []
            for i, src_img in enumerate(self._source_images):
                if progress.wasCanceled():
                    return
                processed = process_image(src_img, self._current_params,
                                         preview_scale=1.0, apply_random_variance=False)
                processed_pages.append(processed)
                progress.setValue(int((i + 1) / total * 100))
                QApplication.processEvents()

            export_images_to_pdf(processed_pages, default_path)
            progress.setValue(100)
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))
        finally:
            progress.close()

    # ── Drag and Drop ──
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._overlay.show()
            self._overlay.raise_()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._overlay.hide()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent):
        self._overlay.hide()
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self._load_file(path)
        event.acceptProposedAction()


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ScanThis")
    app.setStyle(QStyleFactory.create("macOS"))
    window = MainWindow()
    window.resize(1200, 840)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
