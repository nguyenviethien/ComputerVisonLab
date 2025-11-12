ComputerVisionLab

A small Python desktop app with a Tkinter UI for basic image processing using OpenCV. Includes grayscale, binary/adaptive threshold, Canny edges, contours, blur, rotate/flip, zoom/pan, and AI tools (OCR + Object Detection). 

Requirements

- Python 3.8+
- Libraries: `opencv-python`, `Pillow`, `numpy`, `pytesseract`

Install

```
pip install -r requirements.txt
```

Run

```
python app.py
```

Features

- Open/Save image (File → Open/Save As)
- Undo and Reset to original (Edit → Undo/Reset)
- Image operations:
  - Grayscale
  - Binary threshold (with slider)
  - Adaptive threshold
  - Canny edge detection (two thresholds)
  - Contours (draw outlines or filled, min area filter)
  - Gaussian blur (kernel size)
  - Rotate 90° (left/right), Flip (H/V)
  - Zoom in/out, Fit/100%, mouse pan
  - AI: OCR & Detection
    - OCR current image (Tesseract)
    - Detect Plate & OCR (heuristic plate finder + perspective warp, then OCR)
    - Object Detection (YOLOv8n by default; MobileNet‑SSD as alternative)

Notes

- The viewer scales images to fit the window, but processing and saving are done on the full‑resolution image.
- Some operations convert to single‑channel grayscale; the UI adapts for display.

OCR (Tesseract)

The app uses `pytesseract`, which requires the Tesseract OCR binary installed on your system.

- Windows (recommended): UB Mannheim build – https://github.com/UB-Mannheim/tesseract/wiki
  - Typical path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- macOS: `brew install tesseract`
- Linux (Debian/Ubuntu): `sudo apt-get install tesseract-ocr`

Configuring Tesseract in the app

- Use `Settings → Set Tesseract Path…` to select the executable. The path is saved to `settings.json`.
- `Settings → Check Tesseract` shows the detected version and path.
- If OCR is triggered without a configured binary, the app prompts you to locate it.

Object Detection (default = YOLOv8n)

- Default model is YOLOv8n (Ultralytics). Install once:
  - `pip install ultralytics`
  - If PyTorch wheels are unavailable for your Python version, use Python 3.10–3.12.
- Alternative: MobileNet‑SSD (Caffe, VOC 20 classes). If selected, the app can download the model files (~23 MB) or you can provide them manually.
- You can filter by class names in the dialog (comma‑separated) and set a confidence threshold.

Shortcuts

- Ctrl+O: Open, Ctrl+S: Save As, Ctrl+Z: Undo, Ctrl+R: Reset
- Zoom: Ctrl +/−, Ctrl+1 (100%), Ctrl+0 (Fit)
