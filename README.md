ImageUtility

Ứng dụng Python đơn giản với giao diện Tkinter để xử lý ảnh cơ bản sử dụng OpenCV: chuyển xám, nhị phân (binary), edge (Canny), contour, làm mờ, xoay, lật, hoàn tác.

Yêu cầu

- Python 3.8+
- Thư viện: `opencv-python`, `Pillow`, `numpy`

Cài đặt nhanh:

```
pip install -r requirements.txt
```

Chạy ứng dụng

```
python app.py
```

Tính năng chính

- Mở/lưu ảnh (File -> Open/Save As)
- Hoàn tác và đặt lại về ảnh gốc (Edit -> Undo/Reset)
- Các xử lý ảnh:
  - Grayscale (chuyển xám)
  - Binary threshold (ngưỡng nhị phân) với thanh trượt
  - Adaptive threshold (ngưỡng thích nghi)
  - Canny edge detection (phát hiện biên) với 2 ngưỡng
  - Contours (tìm và vẽ đường viền) với lọc theo diện tích tối thiểu
  - Gaussian blur (làm mờ) với kích thước kernel
  - Xoay trái/phải 90°, lật ngang/dọc
  - Zoom in/out, Fit/100%, pan bằng chuột
  - OCR (nhận dạng chữ/số) từ ảnh hiện tại
  - Detect Plate & OCR: thử phát hiện vùng biển số và đọc text

Ghi chú

- Ảnh hiển thị được scale vừa cửa sổ, nhưng thao tác xử lý và lưu luôn làm trên ảnh gốc (kích thước ban đầu).
- Một số thao tác chuyển ảnh thành 1 kênh (grayscale). Khi cần hiển thị, ảnh sẽ được chuyển sang RGB để trình bày trên UI.

## OCR (Tesseract)

Ứng dụng dùng `pytesseract`, yêu cầu máy có cài Tesseract OCR binary.

- Windows: tải và cài đặt Tesseract từ "UB Mannheim" build (khuyến nghị) hoặc bản chính thức.
  - Ví dụ đường dẫn: `C:\Program Files\Tesseract-OCR\tesseract.exe`
- macOS: `brew install tesseract`
- Linux (Debian/Ubuntu): `sudo apt-get install tesseract-ocr`

Nếu ứng dụng không tìm thấy Tesseract, bạn sẽ thấy thông báo lỗi. Hãy cài đặt và (nếu cần) cấu hình biến `pytesseract.pytesseract.tesseract_cmd` trong mã hoặc thêm vào PATH của hệ thống.
