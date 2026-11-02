# PlateVision — Automatic Number Plate Recognition

A local-first ANPR starter project using Python/Flask, HTML, CSS and JavaScript.

## Features
- Webcam capture in the browser (localhost/HTTPS required).
- Upload a vehicle photo.
- OCR using EasyOCR; contrast enhancement for difficult lighting.
- Vehicle category classification using YOLOv8 nano (car, motorcycle, bus, truck).
- SQLite detection history with delete and refresh.
- Dark responsive dashboard UI.

## Important accuracy and coverage notes
This is a working **starter prototype**, not a guaranteed all-country plate-recognition product. EasyOCR reads Latin-script text and the broad candidate filter accepts many alphanumeric plate styles, but plate layouts, alphabets, fonts and legal formats vary by country/state. It does not reliably identify the issuing country/state or fetch owner/RTO details. The UI deliberately labels region as Unknown rather than guessing. OCR confidence is not a guarantee of correctness. Review each result before using it.

The included YOLOv8n model classifies general vehicle types; it is not a dedicated license-plate detector. For production-grade ANPR, train/use a dedicated plate detector (for example, on a properly licensed, diverse plate dataset) and country-specific OCR/format validation.

## Requirements
- Windows 10/11, Python 3.10 or 3.11 recommended
- 8 GB RAM recommended minimum; first model setup needs internet
- Webcam (optional)

## Windows setup
1. Extract the ZIP.
2. Open the extracted `automatic_number_plate_recognition` folder.
3. In the folder address bar type `cmd` and press Enter.
4. Create and activate a virtual environment:

   ```bat
   py -3.11 -m venv venv
   venv\Scripts\activate
   ```

5. Install dependencies (first install can take several minutes):

   ```bat
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the app:

   ```bat
   python app.py
   ```

7. Open **http://127.0.0.1:5000** in Chrome or Edge.
8. Allow camera permission, then click Start camera. Or choose Upload photo.

The first scan initializes EasyOCR and downloads the YOLOv8n model if needed. Keep internet available for this first-time setup. Later inference is local.

## Troubleshooting
- If `py -3.11` is unavailable, install Python 3.11 from python.org and tick **Add Python to PATH**.
- If camera access fails, use `http://127.0.0.1:5000` (not a LAN HTTP address), allow browser camera permission, or upload a photo.
- If installation fails on Windows, use a fresh Python 3.11 64-bit environment and rerun pip install.
- If a plate is missed: use a close, straight, sharp photo with good light and minimal glare.
- The app stores detections in `plates.db` in this folder. Delete that file to erase all local history (stop the app first).

## Privacy
Images are processed by the local Python server and are not saved. Only OCR text, vehicle category, confidence, source and timestamp are stored in the local SQLite database. Do not expose this development server publicly without authentication, HTTPS, rate limiting and a privacy review.
