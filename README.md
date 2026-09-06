# Peel

Peel is a single-file AI image layer extractor. Upload an image and it
separates the subject, reconstructed background, and detected text into
independent downloadable PNG layers.

The complete Flask API and browser UI are contained in [`peel.py`](./peel.py).

## Features

- Subject extraction with `rembg` and the U<sup>2</sup>-Net model
- Background reconstruction with OpenCV Telea inpainting
- Text detection with Tesseract OCR
- In-browser layer visibility controls, zoom, reset, and PNG export
- JSON API for integrating the processing pipeline into another client

## Requirements

- Python 3.10 or newer
- Tesseract OCR installed and available on `PATH`
- Internet access on the first run so `rembg` can download its model

Install the Python dependencies:

```bash
python -m pip install flask opencv-python-headless numpy rembg onnxruntime pytesseract Pillow
```

Install Tesseract separately:

```bash
# Debian/Ubuntu
sudo apt-get install -y tesseract-ocr
```

On Windows, install Tesseract using a trusted installer and add its
installation directory to `PATH`.

## Run

```bash
python peel.py
```

Open <http://localhost:5000> in a browser and upload a JPG or PNG image.

Useful environment variables:

| Variable | Default | Description |
| --- | ---: | --- |
| `PEEL_PORT` | `5000` | HTTP server port |
| `PEEL_DEBUG` | `false` | Enable Flask debug mode |
| `PEEL_MAX_UPLOAD_MB` | `15` | Maximum upload size |
| `PEEL_MAX_DIMENSION` | `1600` | Maximum processed image dimension |

Example:

```bash
PEEL_PORT=8080 PEEL_MAX_DIMENSION=2000 python peel.py
```

## API

Health check:

```bash
curl http://localhost:5000/api/health
```

Extract layers from an image:

```bash
curl -X POST http://localhost:5000/api/extract \
  -F "image=@photo.jpg"
```

The extraction response contains the image dimensions, processing time,
base64-encoded PNG data URLs for the `background`, `subject`, and `text`
layers, and any OCR text boxes detected.

## Notes

- The first extraction may take longer while the segmentation model loads.
- If `rembg` is unavailable, the server reports the dependency error when
  extraction is attempted.
- OCR is optional for the server to start, but text layers will be empty
  unless both `pytesseract` and the Tesseract executable are available.
