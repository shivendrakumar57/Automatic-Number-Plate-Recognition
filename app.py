import os, re, sqlite3, base64, datetime
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), "plates.db")
reader = None
vehicle_model = None

def db():
    con=sqlite3.connect(DB_PATH)
    con.row_factory=sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS detections(
        id INTEGER PRIMARY KEY AUTOINCREMENT, plate TEXT, vehicle TEXT,
        confidence REAL, country TEXT, created_at TEXT, source TEXT)""")
    con.commit()
    return con

def models():
    global reader, vehicle_model
    if reader is None:
        import easyocr
        # English + common Latin-script plate OCR. Add languages as needed.
        reader = easyocr.Reader(['en'], gpu=False)
    if vehicle_model is None:
        try:
            from ultralytics import YOLO
            vehicle_model = YOLO("yolov8n.pt")  # downloads on first run
        except Exception:
            vehicle_model = False
    return reader, vehicle_model

def clean_plate(s):
    s=re.sub(r'[^A-Z0-9-]', '', (s or '').upper())
    return s.strip('-')

def detect_vehicle(img, model):
    if model:
        try:
            result=model.predict(img, imgsz=640, conf=0.25, verbose=False)[0]
            names=model.names
            labels=[]
            for c in result.boxes.cls.tolist():
                name=names[int(c)]
                if name in ("car","motorcycle","bus","truck"):
                    labels.append(name.title())
            if labels: return max(set(labels), key=labels.count)
        except Exception:
            pass
    return "Vehicle not classified"

def process_image(img):
    reader, model=models()
    vehicle=detect_vehicle(img,model)
    # OCR the original plus contrast-enhanced image to improve varied lighting.
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    enhanced=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
    candidates=[]
    for image in (img, enhanced):
        try:
            for item in reader.readtext(image, detail=1, paragraph=False):
                box, txt, conf=item
                cleaned=clean_plate(txt)
                # Broad international candidate filter; intentionally avoids assuming one country's format.
                if len(cleaned)>=5 and re.search(r'[A-Z]',cleaned) and re.search(r'\d',cleaned):
                    score=float(conf) * min(len(cleaned)/8,1)
                    candidates.append((cleaned,score))
        except Exception:
            continue
    if not candidates:
        return {"plate":"Not detected","confidence":0,"vehicle":vehicle,
                "note":"No readable plate found. Try a sharper, closer, well-lit image."}
    # Favor plausible longer alphanumeric strings; de-duplicate OCR variants.
    candidates.sort(key=lambda x:x[1],reverse=True)
    plate,score=candidates[0]
    return {"plate":plate,"confidence":round(score*100,1),"vehicle":vehicle,
            "note":"OCR result is a best-effort reading. Verify characters before relying on it."}

@app.route("/")
def home(): return render_template("index.html")

@app.route("/api/detect",methods=["POST"])
def detect():
    try:
        data=request.get_json(silent=True) or {}
        raw=data.get("image","")
        if "," in raw: raw=raw.split(",",1)[1]
        img_bytes=base64.b64decode(raw)
        arr=np.frombuffer(img_bytes,np.uint8)
        img=cv2.imdecode(arr,cv2.IMREAD_COLOR)
        if img is None: return jsonify(error="Invalid image"),400
        if max(img.shape[:2])>1600:
            scale=1600/max(img.shape[:2])
            img=cv2.resize(img,None,fx=scale,fy=scale)
        result=process_image(img)
        if result["plate"]!="Not detected":
            con=db()
            con.execute("INSERT INTO detections(plate,vehicle,confidence,country,created_at,source) VALUES(?,?,?,?,?,?)",
                (result["plate"],result["vehicle"],result["confidence"],data.get("country","Unknown"),
                 datetime.datetime.now().isoformat(timespec="seconds"),data.get("source","Upload")))
            con.commit(); con.close()
        return jsonify(result)
    except Exception as e:
        return jsonify(error=f"Processing failed: {str(e)}"),500

@app.route("/api/history")
def history():
    con=db()
    rows=con.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 200").fetchall()
    con.close()
    return jsonify([dict(r) for r in rows])

@app.route("/api/history/<int:item_id>",methods=["DELETE"])
def delete_item(item_id):
    con=db(); con.execute("DELETE FROM detections WHERE id=?",(item_id,)); con.commit(); con.close()
    return jsonify(ok=True)

@app.route("/api/health")
def health(): return jsonify(status="ok")

if __name__=="__main__":
    db()
    app.run(host="127.0.0.1",port=5000,debug=False)
