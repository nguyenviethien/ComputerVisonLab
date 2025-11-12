import os
import sys
import math
import json
import platform
import shutil
import urllib.request
import urllib.error
from typing import Optional, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
import pytesseract


class ImageApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("ImageUtility - OpenCV Tools")
        self.root.geometry("1100x700")

        # State
        self.original: Optional[np.ndarray] = None  # BGR or Gray
        self.current: Optional[np.ndarray] = None   # BGR or Gray
        self.undo_stack: list[np.ndarray] = []
        self.max_undo = 10

        # UI
        self._create_menu()
        self._create_toolbar()
        self._create_main_panes()
        self._create_operations_panel()

        # Config and integrations
        self.config: dict = {}
        self._load_config()
        self._init_tesseract()

        # Bindings
        self.root.bind("<Configure>", self._on_resize)

        # Zoom/pan state
        self.zoom_mode: str = 'fit'  # 'fit' or 'fixed'
        self.zoom_factor: float = 1.0  # used when zoom_mode == 'fixed'

        # Status
        self._set_status("Ready")

    # ---- UI Construction ----
    def _create_menu(self) -> None:
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open...", accelerator="Ctrl+O", command=self.open_image)
        file_menu.add_command(label="Save As...", accelerator="Ctrl+S", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Undo", accelerator="Ctrl+Z", command=self.undo)
        edit_menu.add_command(label="Reset", accelerator="Ctrl+R", command=self.reset_image)

        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Set Tesseract Path...", command=self.set_tesseract_path)
        settings_menu.add_command(label="Check Tesseract", command=self.check_tesseract)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)

        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menubar)

        # Shortcuts
        self.root.bind_all("<Control-o>", lambda e: self.open_image())
        self.root.bind_all("<Control-s>", lambda e: self.save_image())
        self.root.bind_all("<Control-z>", lambda e: self.undo())
        self.root.bind_all("<Control-r>", lambda e: self.reset_image())

    def _create_toolbar(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(8, 6))
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)

        ttk.Button(toolbar, text="Open", command=self.open_image).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(toolbar, text="Save", command=self.save_image).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(toolbar, text="Undo", command=self.undo).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(toolbar, text="Reset", command=self.reset_image).grid(row=0, column=3, padx=(0, 6))

        # Zoom controls
        ttk.Separator(toolbar, orient=tk.VERTICAL).grid(row=0, column=4, padx=6, sticky="ns")
        ttk.Button(toolbar, text="-", width=3, command=self.zoom_out).grid(row=0, column=5, padx=(0, 2))
        ttk.Button(toolbar, text="+", width=3, command=self.zoom_in).grid(row=0, column=6, padx=(0, 6))
        ttk.Button(toolbar, text="100%", command=self.zoom_100).grid(row=0, column=7, padx=(0, 6))
        ttk.Button(toolbar, text="Fit", command=self.zoom_fit).grid(row=0, column=8, padx=(0, 6))

        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(toolbar, textvariable=self.status_var, anchor="w")
        self.status_label.grid(row=0, column=9, sticky="ew")
        toolbar.grid_columnconfigure(9, weight=1)

    def _create_main_panes(self) -> None:
        # Main image area (left)
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=1, column=0, sticky="nsew")
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Scrollable canvas for image viewing
        self.viewer_frame = ttk.Frame(main_frame)
        self.viewer_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.viewer_frame, background="#222", highlightthickness=0)
        self.hbar = ttk.Scrollbar(self.viewer_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.vbar = ttk.Scrollbar(self.viewer_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.hbar.grid(row=1, column=0, sticky="ew")
        self.viewer_frame.grid_rowconfigure(0, weight=1)
        self.viewer_frame.grid_columnconfigure(0, weight=1)

        # Canvas interactions
        self.canvas.bind('<ButtonPress-1>', self._canvas_pan_start)
        self.canvas.bind('<B1-Motion>', self._canvas_pan_move)
        self.canvas.bind('<MouseWheel>', self._on_mouse_wheel)           # Windows scroll
        self.canvas.bind('<Control-MouseWheel>', self._on_ctrl_mouse_wheel)  # Windows zoom
        self.canvas.bind('<Button-4>', self._on_mouse_wheel)            # Linux scroll up
        self.canvas.bind('<Button-5>', self._on_mouse_wheel)            # Linux scroll down

        # Keyboard shortcuts for zoom
        self.root.bind_all('<Control-plus>', lambda e: self.zoom_in())
        self.root.bind_all('<Control-KP_Add>', lambda e: self.zoom_in())
        self.root.bind_all('<Control-minus>', lambda e: self.zoom_out())
        self.root.bind_all('<Control-KP_Subtract>', lambda e: self.zoom_out())
        self.root.bind_all('<Control-0>', lambda e: self.zoom_fit())
        self.root.bind_all('<Control-1>', lambda e: self.zoom_100())

        self.tk_img: Optional[ImageTk.PhotoImage] = None
        self.canvas_img_id: Optional[int] = None

        # Operations side panel (right)
        self.ops_panel = ttk.Frame(self.root, padding=(8, 6))
        self.ops_panel.grid(row=1, column=1, sticky="ns")
        self.root.grid_columnconfigure(1, weight=0)

    def _create_operations_panel(self) -> None:
        title = ttk.Label(self.ops_panel, text="Operations", font=("Segoe UI", 11, "bold"))
        title.grid(row=0, column=0, pady=(0, 8), sticky="w")

        row = 1

        ttk.Button(self.ops_panel, text="Grayscale", command=self.apply_grayscale).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        ttk.Button(self.ops_panel, text="Binary Threshold...", command=self._dialog_threshold).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        ttk.Button(self.ops_panel, text="Adaptive Threshold...", command=self._dialog_adaptive).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        ttk.Button(self.ops_panel, text="Canny Edge...", command=self._dialog_canny).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        ttk.Button(self.ops_panel, text="Contours...", command=self._dialog_contours).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        ttk.Separator(self.ops_panel, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky="ew", pady=6)
        row += 1

        ttk.Button(self.ops_panel, text="Gaussian Blur...", command=self._dialog_blur).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        # OCR group
        ttk.Separator(self.ops_panel, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky="ew", pady=6)
        row += 1
        ttk.Button(self.ops_panel, text="OCR Text...", command=self._dialog_ocr).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1
        ttk.Button(self.ops_panel, text="Detect Plate & OCR", command=self.detect_plate_and_ocr).grid(row=row, column=0, sticky="ew", pady=2)
        
        ttk.Button(self.ops_panel, text="Rotate Left 90°", command=lambda: self.apply_rotate(-90)).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        ttk.Button(self.ops_panel, text="Rotate Right 90°", command=lambda: self.apply_rotate(90)).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        ttk.Button(self.ops_panel, text="Flip Horizontal", command=lambda: self.apply_flip('h')).grid(row=row, column=0, sticky="ew", pady=2)
        row += 1

        ttk.Button(self.ops_panel, text="Flip Vertical", command=lambda: self.apply_flip('v')).grid(row=row, column=0, sticky="ew", pady=2)

        row += 1
        ttk.Separator(self.ops_panel, orient=tk.HORIZONTAL).grid(row=row, column=0, sticky="ew", pady=6)
        row += 1
        ttk.Button(self.ops_panel, text="Object Detection...", command=self._dialog_object_detection).grid(row=row, column=0, sticky="ew", pady=2)

        for c in range(1):
            self.ops_panel.grid_columnconfigure(c, weight=1)

    # ---- Helpers ----
    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _show_about(self) -> None:
        # Show brief about with Tesseract status
        tver = self._get_tesseract_version_str()
        messagebox.showinfo(
            "About",
            f"ImageUtility\nTkinter + OpenCV tools for basic image operations.\n\nTesseract: {tver}",
            parent=self.root,
        )

    # ---- Config & Tesseract integration ----
    def _config_file(self) -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')

    def _load_config(self) -> None:
        try:
            path = self._config_file()
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                self.config = {}
        except Exception:
            self.config = {}

    def _save_config(self) -> None:
        try:
            with open(self._config_file(), 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _init_tesseract(self) -> None:
        # Use configured path if present
        cmd = self.config.get('tesseract_cmd')
        if cmd and os.path.exists(cmd):
            pytesseract.pytesseract.tesseract_cmd = cmd
            return

        # Try environment variables
        for env in ('TESSERACT_CMD', 'TESSERACT_PATH'):
            val = os.environ.get(env)
            if val and os.path.exists(val):
                pytesseract.pytesseract.tesseract_cmd = val
                self.config['tesseract_cmd'] = val
                self._save_config()
                return

        # On Windows, probe common locations
        if platform.system().lower().startswith('win'):
            candidates = [
                r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
                r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
            ]
            for c in candidates:
                if os.path.exists(c):
                    pytesseract.pytesseract.tesseract_cmd = c
                    self.config['tesseract_cmd'] = c
                    self._save_config()
                    return
        else:
            # Non-Windows: rely on PATH if available
            which = shutil.which('tesseract')
            if which:
                pytesseract.pytesseract.tesseract_cmd = which
                self.config['tesseract_cmd'] = which
                self._save_config()
                return

    def _get_tesseract_version_str(self) -> str:
        try:
            v = pytesseract.get_tesseract_version()
            return f"{v}"
        except Exception:
            # Not found or not working
            return "not found"

    def _ensure_tesseract(self) -> bool:
        try:
            pytesseract.get_tesseract_version()
            return True
        except pytesseract.TesseractNotFoundError:
            if messagebox.askyesno(
                "Tesseract not found",
                "Không tìm thấy Tesseract OCR. Bạn có muốn chọn file tesseract.exe bây giờ không?",
                parent=self.root,
            ):
                self.set_tesseract_path()
                try:
                    pytesseract.get_tesseract_version()
                    return True
                except Exception:
                    return False
            return False
        except Exception:
            return False

    def set_tesseract_path(self) -> None:
        initial = None
        if platform.system().lower().startswith('win'):
            if os.path.exists(r"C:\\Program Files\\Tesseract-OCR"):
                initial = r"C:\\Program Files\\Tesseract-OCR"
            elif os.path.exists(r"C:\\Program Files (x86)\\Tesseract-OCR"):
                initial = r"C:\\Program Files (x86)\\Tesseract-OCR"
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Select Tesseract executable",
            initialdir=initial,
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")] if platform.system().lower().startswith('win') else [("All files", "*.*")],
        )
        if not path:
            return
        if not os.path.exists(path):
            messagebox.showerror("Invalid path", "Đường dẫn không tồn tại.", parent=self.root)
            return
        # Test the executable
        try:
            old = pytesseract.pytesseract.tesseract_cmd
        except Exception:
            old = None
        pytesseract.pytesseract.tesseract_cmd = path
        try:
            v = pytesseract.get_tesseract_version()
            self.config['tesseract_cmd'] = path
            self._save_config()
            messagebox.showinfo("Tesseract", f"Đã cấu hình: {path}\nVersion: {v}", parent=self.root)
            self._set_status(f"Tesseract OK: {v}")
        except Exception as ex:
            # revert
            if old is not None:
                pytesseract.pytesseract.tesseract_cmd = old
            messagebox.showerror("Tesseract", f"Không thể chạy Tesseract tại đường dẫn này.\n{ex}", parent=self.root)

    def check_tesseract(self) -> None:
        v = self._get_tesseract_version_str()
        messagebox.showinfo("Tesseract", f"Tesseract: {v}\nPath: {getattr(pytesseract.pytesseract, 'tesseract_cmd', '(PATH)')}", parent=self.root)

    def _on_resize(self, event: tk.Event) -> None:
        if event.widget == self.root:
            self._refresh_display()

    def _refresh_display(self) -> None:
        if self.current is None:
            # Clear canvas and show hint text
            self.canvas.delete('all')
            self.canvas.configure(scrollregion=(0, 0, 0, 0))
            # Draw hint text centered
            w = self.canvas.winfo_width() or 800
            h = self.canvas.winfo_height() or 600
            self.canvas.create_text(w//2, h//2, text="Open an image to start", fill="#DDD")
            return
        pil_img = self._to_pil(self.current)
        disp = self._scale_for_view(pil_img)
        self._draw_on_canvas(disp)

    def _scale_for_view(self, pil_img: Image.Image) -> Image.Image:
        # Determine target size based on zoom mode and canvas size
        self.canvas.update_idletasks()
        iw, ih = pil_img.size
        cw = self.canvas.winfo_width() or 800
        ch = self.canvas.winfo_height() or 600
        if self.zoom_mode == 'fit':
            if iw == 0 or ih == 0:
                return pil_img
            scale = min(cw / iw, ch / ih)
            if scale <= 0:
                scale = 0.01
        else:
            scale = max(0.05, min(10.0, float(self.zoom_factor)))
        new_size = (max(1, int(round(iw * scale))), max(1, int(round(ih * scale))))
        if new_size == (iw, ih):
            return pil_img
        return pil_img.resize(new_size, Image.LANCZOS)

    def _draw_on_canvas(self, pil_img: Image.Image) -> None:
        # Render PIL image to canvas and adjust scrollregion
        self.tk_img = ImageTk.PhotoImage(pil_img)
        if self.canvas_img_id is None:
            self.canvas_img_id = self.canvas.create_image(0, 0, image=self.tk_img, anchor='nw')
        else:
            self.canvas.itemconfigure(self.canvas_img_id, image=self.tk_img)
        w, h = pil_img.size
        self.canvas.configure(scrollregion=(0, 0, w, h))

    def _to_pil(self, img: np.ndarray) -> Image.Image:
        if img.ndim == 2:
            return Image.fromarray(img)
        elif img.ndim == 3 and img.shape[2] == 3:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)
        else:
            raise ValueError("Unsupported image format for display")

    def _ensure_image(self) -> bool:
        if self.current is None:
            messagebox.showwarning("No Image", "Vui lòng mở ảnh trước.", parent=self.root)
            return False
        return True

    def _push_undo(self) -> None:
        if self.current is None:
            return
        if len(self.undo_stack) >= self.max_undo:
            self.undo_stack.pop(0)
        self.undo_stack.append(self.current.copy())

    # ---- File ops ----
    def open_image(self) -> None:
        path = filedialog.askopenfilename(title="Open image", parent=self.root,
                                          filetypes=[
                                              ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif"),
                                              ("All files", "*.*"),
                                          ])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror("Open image", "Không thể mở ảnh: " + os.path.basename(path), parent=self.root)
            return
        # If image has alpha, drop alpha for simplicity
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        self.original = img
        self.current = img.copy()
        self.undo_stack.clear()
        # Reset zoom to fit on open
        self.zoom_fit()
        self._set_status(f"Opened {os.path.basename(path)} | {img.shape[1]}x{img.shape[0]}")
        self._refresh_display()

    def save_image(self) -> None:
        if not self._ensure_image():
            return
        path = filedialog.asksaveasfilename(title="Save image", parent=self.root,
                                            defaultextension=".png",
                                            filetypes=[
                                                ("PNG", "*.png"),
                                                ("JPEG", "*.jpg;*.jpeg"),
                                                ("BMP", "*.bmp"),
                                                ("TIFF", "*.tif;*.tiff"),
                                            ])
        if not path:
            return
        ok = cv2.imwrite(path, self.current)
        if not ok:
            messagebox.showerror("Save image", "Lưu ảnh thất bại.", parent=self.root)
        else:
            self._set_status(f"Saved {os.path.basename(path)}")

    # ---- Edit ops ----
    def undo(self) -> None:
        if not self.undo_stack:
            self._set_status("Nothing to undo")
            return
        self.current = self.undo_stack.pop()
        self._set_status("Undo")
        self._refresh_display()

    def reset_image(self) -> None:
        if self.original is None:
            return
        self.current = self.original.copy()
        self.undo_stack.clear()
        self._set_status("Reset to original")
        self._refresh_display()

    # ---- Operations ----
    def apply_grayscale(self) -> None:
        if not self._ensure_image():
            return
        self._push_undo()
        if self.current.ndim == 3:
            self.current = cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY)
        self._set_status("Applied grayscale")
        self._refresh_display()

    def _dialog_threshold(self) -> None:
        if not self._ensure_image():
            return
        dlg = ParamDialog(self.root, title="Binary Threshold", params=[
            IntParam("threshold", "Threshold", 128, 0, 255),
            BoolParam("inverse", "Inverse", False),
        ])
        res = dlg.show()
        if res is None:
            return
        thr = int(res["threshold"])
        inv = bool(res["inverse"])
        self.apply_threshold(thr, inv)

    def apply_threshold(self, threshold: int, inverse: bool = False) -> None:
        if not self._ensure_image():
            return
        self._push_undo()
        img = self.current
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mode = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
        _, out = cv2.threshold(img, threshold, 255, mode)
        self.current = out
        self._set_status(f"Applied threshold ({threshold}{', inv' if inverse else ''})")
        self._refresh_display()

    def _dialog_adaptive(self) -> None:
        if not self._ensure_image():
            return
        dlg = ParamDialog(self.root, title="Adaptive Threshold", params=[
            IntParam("block", "Block Size (odd)", 11, 3, 99),
            IntParam("C", "C", 2, -20, 20),
            ChoiceParam("method", "Method", ["MEAN", "GAUSSIAN"], "GAUSSIAN"),
        ])
        res = dlg.show()
        if res is None:
            return
        block = int(res["block"])
        if block % 2 == 0:
            block += 1
        C = int(res["C"])
        method = str(res["method"]).upper()
        self.apply_adaptive(block, C, method)

    def apply_adaptive(self, block_size: int, C: int, method: str = "GAUSSIAN") -> None:
        if not self._ensure_image():
            return
        if block_size < 3:
            block_size = 3
        if block_size % 2 == 0:
            block_size += 1
        self._push_undo()
        img = self.current
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        adapt_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "GAUSSIAN" else cv2.ADAPTIVE_THRESH_MEAN_C
        out = cv2.adaptiveThreshold(img, 255, adapt_method, cv2.THRESH_BINARY, block_size, C)
        self.current = out
        self._set_status(f"Applied adaptive threshold (block={block_size}, C={C}, {method})")
        self._refresh_display()

    def _dialog_canny(self) -> None:
        if not self._ensure_image():
            return
        dlg = ParamDialog(self.root, title="Canny Edge", params=[
            IntParam("t1", "Threshold 1", 100, 0, 255),
            IntParam("t2", "Threshold 2", 200, 0, 255),
        ])
        res = dlg.show()
        if res is None:
            return
        t1 = int(res["t1"]) ; t2 = int(res["t2"])
        self.apply_canny(t1, t2)

    def apply_canny(self, t1: int, t2: int) -> None:
        if not self._ensure_image():
            return
        self._push_undo()
        img = self.current
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img, t1, t2)
        self.current = edges
        self._set_status(f"Applied Canny (t1={t1}, t2={t2})")
        self._refresh_display()

    def _dialog_contours(self) -> None:
        if not self._ensure_image():
            return
        dlg = ParamDialog(self.root, title="Contours", params=[
            IntParam("min_area", "Min area", 50, 0, 5000),
            ChoiceParam("mode", "Draw", ["OUTLINE", "FILLED"], "OUTLINE"),
        ])
        res = dlg.show()
        if res is None:
            return
        min_area = int(res["min_area"]) ; mode = str(res["mode"]).upper()
        self.apply_contours(min_area, mode)

    def apply_contours(self, min_area: int = 50, mode: str = "OUTLINE") -> None:
        if not self._ensure_image():
            return
        self._push_undo()
        src = self.current
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) if src.ndim == 3 else src
        # Otsu binary for robust contour finding
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Prepare drawing canvas (color)
        if src.ndim == 2:
            canvas = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        else:
            canvas = src.copy()

        drawn = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(0, min_area):
                continue
            color = (0, 255, 0)
            if mode == "FILLED":
                cv2.drawContours(canvas, [cnt], -1, color, thickness=cv2.FILLED)
            else:
                cv2.drawContours(canvas, [cnt], -1, color, thickness=2)
            drawn += 1

        self.current = canvas
        self._set_status(f"Contours drawn: {drawn}")
        self._refresh_display()

    def _dialog_blur(self) -> None:
        if not self._ensure_image():
            return
        dlg = ParamDialog(self.root, title="Gaussian Blur", params=[
            IntParam("ksize", "Kernel size (odd)", 5, 1, 51),
        ])
        res = dlg.show()
        if res is None:
            return
        k = int(res["ksize"]) ;
        self.apply_blur(k)

    def apply_blur(self, ksize: int = 5) -> None:
        if not self._ensure_image():
            return
        if ksize < 1:
            ksize = 1
        if ksize % 2 == 0:
            ksize += 1
        self._push_undo()
        self.current = cv2.GaussianBlur(self.current, (ksize, ksize), 0)
        self._set_status(f"Applied Gaussian blur (k={ksize})")
        self._refresh_display()

    def apply_rotate(self, angle_deg: int) -> None:
        if not self._ensure_image():
            return
        self._push_undo()
        if angle_deg % 360 == 90 or angle_deg % 360 == -270:
            self.current = cv2.rotate(self.current, cv2.ROTATE_90_CLOCKWISE)
        elif angle_deg % 360 == -90 or angle_deg % 360 == 270:
            self.current = cv2.rotate(self.current, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif angle_deg % 360 == 180 or angle_deg % 360 == -180:
            self.current = cv2.rotate(self.current, cv2.ROTATE_180)
        else:
            # Arbitrary rotation using warpAffine around center
            (h, w) = self.current.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
            self.current = cv2.warpAffine(self.current, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        self._set_status(f"Rotated {angle_deg}°")
        self._refresh_display()

    def apply_flip(self, mode: str) -> None:
        if not self._ensure_image():
            return
        self._push_undo()
        if mode == 'h':
            self.current = cv2.flip(self.current, 1)
            self._set_status("Flipped horizontally")
        elif mode == 'v':
            self.current = cv2.flip(self.current, 0)
            self._set_status("Flipped vertically")
        else:
            return
        self._refresh_display()

    # ---- OCR ----
    def _dialog_ocr(self) -> None:
        if not self._ensure_image():
            return
        dlg = ParamDialog(self.root, title="OCR Text", params=[
            ChoiceParam("lang", "Language", ["eng", "vie", "eng+vie"], "eng"),
            ChoiceParam("psm", "Page Seg Mode", ["Auto(3)", "Block(6)", "SingleLine(7)", "SingleWord(8)", "Sparse(11)", "Raw(13)"], "Auto(3)"),
            ChoiceParam("charset", "Charset", ["General", "Digits", "Alphanumeric"], "General"),
            BoolParam("invert", "Auto invert for dark bg", True),
        ])
        res = dlg.show()
        if res is None:
            return
        lang = str(res["lang"]).strip()
        psm_map = {"Auto(3)":3, "Block(6)":6, "SingleLine(7)":7, "SingleWord(8)":8, "Sparse(11)":11, "Raw(13)":13}
        psm = psm_map.get(str(res["psm"])) or 3
        charset = str(res["charset"]) or "General"
        invert = bool(res["invert"]) if res.get("invert") is not None else True
        self.ocr_current_image(lang, psm, charset, invert)

    def _prepare_for_ocr(self, src: np.ndarray, invert: bool = True) -> Image.Image:
        # Convert image to good contrast grayscale/binary for OCR
        img = src
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        # Adaptive threshold for robustness
        bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, 5)
        if invert:
            # If background is dark and text is bright, or vice versa, try to keep text dark on light bg.
            # Heuristic: invert if mean < 127 (mostly dark)
            if np.mean(bin_img) < 127:
                bin_img = cv2.bitwise_not(bin_img)
        return Image.fromarray(bin_img)

    def _tesseract_ocr(self, pil_img: Image.Image, lang: str, psm: int, charset: str) -> str:
        try:
            config_parts = [f"--oem 3", f"--psm {int(psm)}"]
            if charset == "Digits":
                config_parts.append("-c tessedit_char_whitelist=0123456789")
            elif charset == "Alphanumeric":
                config_parts.append("-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            config = " ".join(config_parts)
            text = pytesseract.image_to_string(pil_img, lang=lang, config=config)
            return text.strip()
        except pytesseract.TesseractNotFoundError:
            messagebox.showerror(
                "Tesseract not found",
                "Không tìm thấy Tesseract OCR. Hãy cài đặt Tesseract (Windows: C\\Program Files\\Tesseract-OCR) hoặc thêm vào PATH.",
                parent=self.root
            )
            return ""
        except Exception as ex:
            messagebox.showerror("OCR error", f"{ex}", parent=self.root)
            return ""

    def _show_ocr_result(self, text: str, title: str = "OCR Result") -> None:
        # Show text and copy to clipboard
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo(title, f"Text:\n{text if text else '(empty)'}\n\n(Đã copy vào clipboard)", parent=self.root)

    def ocr_current_image(self, lang: str = "eng", psm: int = 3, charset: str = "General", invert: bool = True) -> None:
        if not self._ensure_image():
            return
        if not self._ensure_tesseract():
            return
        pre = self._prepare_for_ocr(self.current, invert=invert)
        text = self._tesseract_ocr(pre, lang=lang, psm=psm, charset=charset)
        self._set_status("OCR done")
        self._show_ocr_result(text, "OCR Text")

    def detect_plate_and_ocr(self) -> None:
        if not self._ensure_image():
            return
        if not self._ensure_tesseract():
            return
        # Try detect a plate-like rectangular region and OCR
        src = self.current
        if src.ndim == 3:
            gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        else:
            gray = src
        # Preprocessing for edge detection
        blur = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        edges = cv2.Canny(blur, 100, 200)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:50]

        target_quad = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                # Check aspect ratio using bounding rect
                x, y, w, h = cv2.boundingRect(approx)
                if h == 0:
                    continue
                ar = w / float(h)
                if 2.0 <= ar <= 6.0:
                    target_quad = approx.reshape(4, 2)
                    break

        if target_quad is None:
            # fallback: OCR whole image
            pre = self._prepare_for_ocr(src, invert=True)
            text = self._tesseract_ocr(pre, lang="eng", psm=7, charset="Alphanumeric")
            self._show_ocr_result(text or "", "Detect Plate & OCR (fallback)")
            self._set_status("Plate not found; OCR on full image")
            return

        # Order the points for perspective transform
        quad = self._order_points(target_quad)
        (tl, tr, br, bl) = quad
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxW = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxH = int(max(heightA, heightB))
        maxW = max(1, maxW)
        maxH = max(1, maxH)

        dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(quad.astype("float32"), dst)
        warped = cv2.warpPerspective(src, M, (maxW, maxH))

        pre = self._prepare_for_ocr(warped, invert=True)
        text = self._tesseract_ocr(pre, lang="eng", psm=7, charset="Alphanumeric")
        self._show_ocr_result(text or "", "Detect Plate & OCR")
        self._set_status("Plate OCR done")

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        # Order 4 points as tl, tr, br, bl
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # tl
        rect[2] = pts[np.argmax(s)]  # br
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # tr
        rect[3] = pts[np.argmax(diff)]  # bl
        return rect

    # ---- Object Detection (MobileNet-SSD) ----
    def _dialog_object_detection(self) -> None:
        if not self._ensure_image():
            return
        dlg = ParamDialog(self.root, title="Object Detection", params=[
            ChoiceParam("model", "Model", ["MobileNet-SSD"], "MobileNet-SSD"),
            IntParam("conf", "Confidence %", 50, 1, 100),
            TextParam("classes", "Filter classes (comma, optional)", ""),
            BoolParam("labels", "Draw labels", True),
        ])
        res = dlg.show()
        if res is None:
            return
        model = str(res.get("model", "MobileNet-SSD"))
        conf = int(res.get("conf", 50)) / 100.0
        filt = str(res.get("classes", "")).strip()
        draw_labels = bool(res.get("labels", True))
        include = None
        if filt:
            include = {s.strip().lower() for s in filt.split(',') if s.strip()}
        self.detect_objects_mobilenet(confidence=conf, include_classes=include, draw_labels=draw_labels)

    def _models_dir(self) -> str:
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
        os.makedirs(d, exist_ok=True)
        return d

    def _download(self, url: str, dest: str) -> bool:
        try:
            self._set_status(f"Downloading model: {os.path.basename(dest)}")
            urllib.request.urlretrieve(url, dest)
            return True
        except Exception as ex:
            messagebox.showerror("Download failed", f"{ex}", parent=self.root)
            return False

    def _ensure_mobilenet_ssd(self) -> Optional[tuple[str, str]]:
        models_dir = self._models_dir()
        proto = os.path.join(models_dir, 'MobileNetSSD_deploy.prototxt')
        weights = os.path.join(models_dir, 'MobileNetSSD_deploy.caffemodel')
        have_proto = os.path.exists(proto)
        have_weights = os.path.exists(weights)
        if have_proto and have_weights:
            return proto, weights
        # Ask to download
        if not messagebox.askyesno("Model missing", "MobileNet-SSD model not found. Download now (~23MB)?", parent=self.root):
            return None
        # Sources (primary and fallback)
        proto_urls = [
            'https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt',
            'https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/MobileNetSSD_deploy.prototxt',
        ]
        weight_urls = [
            'https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel',
        ]
        ok = have_proto
        if not have_proto:
            for u in proto_urls:
                if self._download(u, proto):
                    ok = True
                    break
        if not ok:
            return None
        ok2 = have_weights
        if not have_weights:
            for u in weight_urls:
                if self._download(u, weights):
                    ok2 = True
                    break
        if not ok2:
            return None
        return proto, weights

    def _mobilenet_classes(self) -> list[str]:
        return [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
        ]

    def _class_color(self, idx: int) -> tuple[int, int, int]:
        # Deterministic pseudo-color by class index
        np.random.seed(idx + 123)
        c = np.random.randint(0, 255, size=3).tolist()
        return (int(c[0]), int(c[1]), int(c[2]))

    def detect_objects_mobilenet(self, confidence: float = 0.5, include_classes: Optional[set[str]] = None, draw_labels: bool = True) -> None:
        if not self._ensure_image():
            return
        ensured = self._ensure_mobilenet_ssd()
        if not ensured:
            return
        proto, weights = ensured
        try:
            net = cv2.dnn.readNetFromCaffe(proto, weights)
        except Exception as ex:
            messagebox.showerror("Load model", f"Failed to load model.\n{ex}", parent=self.root)
            return

        img = self.current
        # Work on color image for drawing
        if img.ndim == 2:
            src = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            src = img.copy()
        (h, w) = src.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(src, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        classes = self._mobilenet_classes()
        drawn = 0
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < confidence:
                continue
            class_id = int(detections[0, 0, i, 1])
            if class_id < 0 or class_id >= len(classes):
                continue
            label = classes[class_id]
            if include_classes is not None and label.lower() not in include_classes:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            startX = max(0, min(w - 1, startX))
            startY = max(0, min(h - 1, startY))
            endX = max(0, min(w - 1, endX))
            endY = max(0, min(h - 1, endY))
            color = self._class_color(class_id)
            cv2.rectangle(src, (startX, startY), (endX, endY), color, 2)
            if draw_labels:
                text = f"{label}: {conf*100:.1f}%"
                y = startY - 6 if startY - 6 > 6 else startY + 15
                cv2.putText(src, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            drawn += 1

        if drawn == 0:
            messagebox.showinfo("Object Detection", "No objects above threshold.", parent=self.root)
        self._push_undo()
        self.current = src
        self._set_status(f"Detections: {drawn}")
        self._refresh_display()

    # ---- Zoom controls ----
    def zoom_fit(self) -> None:
        self.zoom_mode = 'fit'
        self._set_status("Zoom: Fit to window")
        self._refresh_display()

    def zoom_100(self) -> None:
        self.zoom_mode = 'fixed'
        self.zoom_factor = 1.0
        self._set_status("Zoom: 100%")
        self._refresh_display()

    def zoom_in(self) -> None:
        if self.current is None:
            return
        if self.zoom_mode == 'fit':
            # Switch to fixed at current scale approximated by fit
            iw, ih = self._to_pil(self.current).size
            cw = self.canvas.winfo_width() or 800
            ch = self.canvas.winfo_height() or 600
            self.zoom_factor = min(cw / iw, ch / ih)
            self.zoom_mode = 'fixed'
        self.zoom_factor = min(10.0, self.zoom_factor * 1.1)
        self._set_status(f"Zoom: {int(self.zoom_factor*100)}%")
        self._refresh_display()

    def zoom_out(self) -> None:
        if self.current is None:
            return
        if self.zoom_mode == 'fit':
            iw, ih = self._to_pil(self.current).size
            cw = self.canvas.winfo_width() or 800
            ch = self.canvas.winfo_height() or 600
            self.zoom_factor = min(cw / iw, ch / ih)
            self.zoom_mode = 'fixed'
        self.zoom_factor = max(0.05, self.zoom_factor / 1.1)
        self._set_status(f"Zoom: {int(self.zoom_factor*100)}%")
        self._refresh_display()

    # ---- Canvas interactions ----
    def _canvas_pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def _canvas_pan_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_mouse_wheel(self, event):
        # Scroll vertically when not holding Ctrl
        if getattr(event, 'delta', 0) != 0:
            delta = -1 if event.delta > 0 else 1  # Windows: delta positive when wheel up
            self.canvas.yview_scroll(delta, 'units')
        else:
            # Linux Button-4/5
            if event.num == 4:
                self.canvas.yview_scroll(-1, 'units')
            elif event.num == 5:
                self.canvas.yview_scroll(1, 'units')

    def _on_ctrl_mouse_wheel(self, event):
        # Ctrl + wheel -> zoom
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()


# ---- Simple param dialog primitives ----
class ParamBase:
    def __init__(self, key: str, label: str):
        self.key = key
        self.label = label


class IntParam(ParamBase):
    def __init__(self, key: str, label: str, default: int, minval: int, maxval: int):
        super().__init__(key, label)
        self.default = default
        self.minval = minval
        self.maxval = maxval


class BoolParam(ParamBase):
    def __init__(self, key: str, label: str, default: bool):
        super().__init__(key, label)
        self.default = default


class ChoiceParam(ParamBase):
    def __init__(self, key: str, label: str, choices: list[str], default: Optional[str] = None):
        super().__init__(key, label)
        self.choices = choices
        self.default = default or (choices[0] if choices else "")


class TextParam(ParamBase):
    def __init__(self, key: str, label: str, default: Optional[str] = None):
        super().__init__(key, label)
        self.default = default or ""


class ParamDialog:
    def __init__(self, parent: tk.Tk, title: str, params: list[ParamBase]):
        self.parent = parent
        self.title = title
        self.params = params
        self.values: dict[str, object] = {}
        self.result: Optional[dict[str, object]] = None

    def show(self) -> Optional[dict[str, object]]:
        win = tk.Toplevel(self.parent)
        win.title(self.title)
        win.transient(self.parent)
        win.grab_set()
        frm = ttk.Frame(win, padding=(10, 10))
        frm.grid(row=0, column=0, sticky="nsew")
        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)

        row = 0
        widgets = []
        for p in self.params:
            if isinstance(p, IntParam):
                ttk.Label(frm, text=p.label).grid(row=row, column=0, sticky="w", pady=4)
                var = tk.IntVar(value=p.default)
                scale = ttk.Scale(frm, from_=p.minval, to=p.maxval, orient=tk.HORIZONTAL)
                scale.set(p.default)
                scale.grid(row=row, column=1, sticky="ew", padx=(8, 0))
                # Show current value
                val_lbl = ttk.Label(frm, text=str(p.default), width=6)
                val_lbl.grid(row=row, column=2, sticky="w", padx=(6, 0))

                def mk_update(lbl=val_lbl, s=scale, v=var):
                    def _u(e=None):
                        v.set(int(round(s.get())))
                        lbl.configure(text=str(v.get()))
                    return _u

                scale.configure(command=mk_update())
                widgets.append((p, var))
                row += 1
            elif isinstance(p, BoolParam):
                var = tk.BooleanVar(value=p.default)
                chk = ttk.Checkbutton(frm, text=p.label, variable=var)
                chk.grid(row=row, column=0, columnspan=3, sticky="w", pady=4)
                widgets.append((p, var))
                row += 1
            elif isinstance(p, ChoiceParam):
                ttk.Label(frm, text=p.label).grid(row=row, column=0, sticky="w", pady=4)
                var = tk.StringVar(value=p.default)
                cb = ttk.Combobox(frm, textvariable=var, values=p.choices, state="readonly")
                cb.grid(row=row, column=1, columnspan=2, sticky="ew")
                widgets.append((p, var))
                row += 1
            elif isinstance(p, TextParam):
                ttk.Label(frm, text=p.label).grid(row=row, column=0, sticky="w", pady=4)
                var = tk.StringVar(value=p.default or "")
                ent = ttk.Entry(frm, textvariable=var)
                ent.grid(row=row, column=1, columnspan=2, sticky="ew")
                widgets.append((p, var))
                row += 1

        frm.grid_columnconfigure(1, weight=1)

        btns = ttk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=3, pady=(10, 0), sticky="e")
        ok_btn = ttk.Button(btns, text="OK", command=lambda: self._close(win, widgets, ok=True))
        cancel_btn = ttk.Button(btns, text="Cancel", command=lambda: self._close(win, widgets, ok=False))
        ok_btn.grid(row=0, column=0, padx=4)
        cancel_btn.grid(row=0, column=1)

        # Center dialog over parent
        try:
            win.update_idletasks()
            self.parent.update_idletasks()
            w = win.winfo_width(); h = win.winfo_height()
            pw = self.parent.winfo_width(); ph = self.parent.winfo_height()
            if pw <= 1 or ph <= 1:
                sx = win.winfo_screenwidth(); sy = win.winfo_screenheight()
                x = max(0, (sx - w) // 2); y = max(0, (sy - h) // 2)
            else:
                px = self.parent.winfo_rootx(); py = self.parent.winfo_rooty()
                x = max(0, int(px + (pw - w) / 2)); y = max(0, int(py + (ph - h) / 2))
            win.geometry(f"+{x}+{y}")
        except Exception:
            pass

        win.bind("<Return>", lambda e: self._close(win, widgets, ok=True))
        win.bind("<Escape>", lambda e: self._close(win, widgets, ok=False))

        self.parent.wait_window(win)
        return self.result

    def _close(self, win: tk.Toplevel, widgets, ok: bool):
        if ok:
            out: dict[str, object] = {}
            for p, var in widgets:
                if isinstance(p, IntParam):
                    out[p.key] = int(var.get())
                elif isinstance(p, BoolParam):
                    out[p.key] = bool(var.get())
                elif isinstance(p, ChoiceParam):
                    out[p.key] = str(var.get())
                elif isinstance(p, TextParam):
                    out[p.key] = str(var.get())
            self.result = out
        else:
            self.result = None
        win.destroy()


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use('vista')
    except tk.TclError:
        pass
    app = ImageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
