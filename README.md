# ScanThis

macOS desktop application for processing and exporting PDF documents.

## Features

- Import PDF and images (PNG, JPG, TIFF, BMP)
- Navigate through multi-page documents
- Real-time preview
- Drag-and-drop support
- Export to PDF with applied settings
- Global settings: rotation, brightness, contrast, blur, noise, aging
- "Scanned" look mode

## Tech Stack

- **PyQt6** — GUI
- **PyMuPDF** — PDF read/write
- **OpenCV + NumPy** — Image processing
- **PyInstaller** — .app bundle building

## Run from source

```bash
pip install -r requirements.txt
python -m src.main
```

## Build

```bash
pip install pyinstaller
pyinstaller build.spec
```

Result: `dist/ScanThis.app`
