import os
import sys
import math
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

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._show_about)

        menubar.add_cascade(label="File", menu=file_menu)
        menubar.add_cascade(label="Edit", menu=edit_menu)
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

        for c in range(1):
            self.ops_panel.grid_columnconfigure(c, weight=1)

    # ---- Helpers ----
    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _show_about(self) -> None:
        messagebox.showinfo("About", "ImageUtility\nTkinter + OpenCV tools for basic image operations.")

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
            messagebox.showwarning("No Image", "Vui lòng mở ảnh trước.")
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
        path = filedialog.askopenfilename(title="Open image",
                                          filetypes=[
                                              ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.tif"),
                                              ("All files", "*.*"),
                                          ])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            messagebox.showerror("Open image", "Không thể mở ảnh: " + os.path.basename(path))
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
        path = filedialog.asksaveasfilename(title="Save image",
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
            messagebox.showerror("Save image", "Lưu ảnh thất bại.")
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
                "Không tìm thấy Tesseract OCR. Hãy cài đặt Tesseract (Windows: C\\Program Files\\Tesseract-OCR) hoặc thêm vào PATH."
            )
            return ""
        except Exception as ex:
            messagebox.showerror("OCR error", f"{ex}")
            return ""

    def _show_ocr_result(self, text: str, title: str = "OCR Result") -> None:
        # Show text and copy to clipboard
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo(title, f"Text:\n{text if text else '(empty)'}\n\n(Đã copy vào clipboard)")

    def ocr_current_image(self, lang: str = "eng", psm: int = 3, charset: str = "General", invert: bool = True) -> None:
        if not self._ensure_image():
            return
        pre = self._prepare_for_ocr(self.current, invert=invert)
        text = self._tesseract_ocr(pre, lang=lang, psm=psm, charset=charset)
        self._set_status("OCR done")
        self._show_ocr_result(text, "OCR Text")

    def detect_plate_and_ocr(self) -> None:
        if not self._ensure_image():
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

        frm.grid_columnconfigure(1, weight=1)

        btns = ttk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=3, pady=(10, 0), sticky="e")
        ok_btn = ttk.Button(btns, text="OK", command=lambda: self._close(win, widgets, ok=True))
        cancel_btn = ttk.Button(btns, text="Cancel", command=lambda: self._close(win, widgets, ok=False))
        ok_btn.grid(row=0, column=0, padx=4)
        cancel_btn.grid(row=0, column=1)

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
