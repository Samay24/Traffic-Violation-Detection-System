from flask import Flask, render_template, request, redirect, url_for, flash, Response
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from ultralytics import YOLO
import os
import cv2
from datetime import datetime
from collections import defaultdict
import easyocr
import re
from reportlab.pdfgen import canvas

# ---------------- APP CONFIG ----------------

app = Flask(__name__)
app.config['SECRET_KEY'] = "traffic-secret"
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///traffic.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")

model = YOLO(MODEL_PATH)

VEHICLE_CLASSES = [2,3,5,7]

reader = easyocr.Reader(['en'])

# ---------------- DATABASE ----------------

class Violation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    violation_type = db.Column(db.String(100))
    video_name = db.Column(db.String(200))
    evidence_file = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Challan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    vehicle_number = db.Column(db.String(20))
    violation_type = db.Column(db.String(100))
    fine_amount = db.Column(db.Integer)
    evidence_file = db.Column(db.String(200))
    pdf_file = db.Column(db.String(200))
    date = db.Column(db.DateTime, default=datetime.utcnow)

# ---------------- FINES ----------------

FINE_RULES = {
"Helmet":500,
"Red Light":1000,
"Wrong Lane":1500,
"Overspeed":2000
}

# ---------------- UTIL ----------------

def clean_filename(text):
    return re.sub(r'[^A-Za-z0-9]', '', text)

# ---------------- OCR ----------------

def read_plate(image_path):

    img = cv2.imread(image_path)
    if img is None:
        return "UNKNOWN"

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img)

    for r in results:
        plate = clean_filename(r[1])
        if len(plate) >= 6:
            return plate

    return "UNKNOWN"

# ---------------- CHALLAN ----------------

def generate_challan_pdf(vehicle, violation, fine, evidence):

    vehicle = clean_filename(vehicle)
    filename = f"challan_{vehicle}_{datetime.now().strftime('%H%M%S')}.pdf"

    path = os.path.join("static/challans", filename)

    c = canvas.Canvas(path)

    c.setFont("Helvetica-Bold",16)
    c.drawString(200,800,"Traffic Violation Challan")

    c.setFont("Helvetica",12)
    c.drawString(100,750,f"Vehicle Number : {vehicle}")
    c.drawString(100,720,f"Violation : {violation}")
    c.drawString(100,690,f"Fine Amount : ₹{fine}")
    c.drawString(100,660,f"Evidence : {evidence}")
    c.drawString(100,630,f"Date : {datetime.now()}")

    c.save()

    return filename

def create_challan(evidence_file, violation_type):

    with app.app_context():

        image_path = os.path.join("static/evidence", evidence_file)

        vehicle_number = read_plate(image_path)

        fine = FINE_RULES.get(violation_type,500)

        pdf_file = generate_challan_pdf(vehicle_number, violation_type, fine, evidence_file)

        challan = Challan(
            vehicle_number = vehicle_number,
            violation_type = violation_type,
            fine_amount = fine,
            evidence_file = evidence_file,
            pdf_file = pdf_file
        )

        db.session.add(challan)
        db.session.commit()

def save_violation(vtype, video_path, evidence):

    with app.app_context():

        db.session.add(
            Violation(
                violation_type=vtype,
                video_name=os.path.basename(video_path),
                evidence_file=evidence
            )
        )

        db.session.commit()

# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():

    file = request.files.get("video")
    violation_type = request.form.get("type")

    if not file:
        flash("No file selected")
        return redirect(url_for("home"))

    filename = clean_filename(file.filename.replace(" ","_"))
    upload_path = os.path.join("static/uploads", filename)

    file.save(upload_path)

    return redirect(url_for("detect", video=upload_path, type=violation_type))

@app.route("/detect")
def detect():

    video = request.args.get("video")
    violation_type = request.args.get("type")

    return render_template("detect.html", video=video, type=violation_type)

@app.route("/dashboard")
def dashboard():

    total = Violation.query.count()
    helmet = Violation.query.filter_by(violation_type="Helmet").count()
    overspeed = Violation.query.filter_by(violation_type="Overspeed").count()
    wronglane = Violation.query.filter_by(violation_type="Wrong Lane").count()
    redlight = Violation.query.filter_by(violation_type="Red Light").count()
    challans = Challan.query.count()
    revenue = db.session.query(db.func.sum(Challan.fine_amount)).scalar() or 0

    return render_template(
        "dashboard.html",
        total=total,
        helmet=helmet,
        overspeed=overspeed,
        wronglane=wronglane,
        redlight=redlight,
        challans=challans,
        revenue=revenue
    )

@app.route("/history")
def history():

    violations = Violation.query.order_by(Violation.timestamp.desc()).all()

    return render_template("history.html", violations=violations)

@app.route("/analytics")
def analytics():

    total = Violation.query.count()
    helmet = Violation.query.filter_by(violation_type="Helmet").count()
    overspeed = Violation.query.filter_by(violation_type="Overspeed").count()
    wronglane = Violation.query.filter_by(violation_type="Wrong Lane").count()
    redlight = Violation.query.filter_by(violation_type="Red Light").count()

    return render_template(
        "analytics.html",
        total=total,
        helmet=helmet,
        overspeed=overspeed,
        wronglane=wronglane,
        redlight=redlight
    )

@app.route("/evidence")
def evidence():

    violations = Violation.query.order_by(Violation.timestamp.desc()).all()

    return render_template(
        "evidence.html",
        violations=violations
    )

@app.route("/challans")
def challans():

    challans = Challan.query.order_by(Challan.date.desc()).all()

    return render_template(
        "challans.html",
        challans=challans
    )


# ---------------- VIDEO PROCESS ----------------

def generate_frames(video_path, violation_type):

    cap = cv2.VideoCapture(video_path)

    STOP_LINE_Y = 350
    violated_vehicles = set()

    prev_y = {}
    wrong_ids = set()
    motion_history = {}

    vehicle_entry = {}
    vehicle_speed = {}
    overspeed_locked = set()

    helmet_count = defaultdict(int)
    no_helmet_count = defaultdict(int)
    final_status = {}
    saved_evidence = set()

    frame_no = 0

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        H,W,_ = frame.shape

        # ================= RED LIGHT =================

        if violation_type == "Red Light":

            cv2.line(frame,(0,STOP_LINE_Y),(W,STOP_LINE_Y),(0,0,255),3)

            cv2.putText(frame,"STOP LINE (RED SIGNAL)",
                        (10,STOP_LINE_Y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,(0,0,255),2)

            results = model(frame)

            for r in results:
                for box in r.boxes:

                    cls = int(box.cls[0])

                    if cls in VEHICLE_CLASSES:

                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        cx = (x1+x2)//2
                        cy = (y1+y2)//2

                        vehicle_id = (cx//50,cy//50)

                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.circle(frame,(cx,cy),5,(255,0,0),-1)

                        if cy > STOP_LINE_Y and vehicle_id not in violated_vehicles:

                            violated_vehicles.add(vehicle_id)

                            cv2.putText(frame,"RED LIGHT VIOLATION",
                                        (x1,y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.9,(0,0,255),3)

                            evidence=f"redlight_{frame_no}.jpg"

                            cv2.imwrite(os.path.join("static/evidence",evidence),frame)

                            save_violation("Red Light",video_path,evidence)
                            create_challan(evidence,"Red Light")

        # ================= WRONG LANE =================

        elif violation_type == "Wrong Lane":

            FRAME_CONFIRM = 5
            DIRECTION_THRESHOLD = 3

            results = model.track(frame,persist=True)

            for r in results:

                if r.boxes.id is None:
                    continue

                for box,track_id,cls in zip(r.boxes.xyxy,r.boxes.id,r.boxes.cls):

                    track_id=int(track_id)
                    cls=int(cls)

                    if cls not in VEHICLE_CLASSES:
                        continue

                    x1,y1,x2,y2=map(int,box)
                    cx=(x1+x2)//2
                    cy=(y1+y2)//2

                    if track_id not in motion_history:
                        motion_history[track_id]=[]

                    if track_id in prev_y:

                        dy=cy-prev_y[track_id]
                        motion_history[track_id].append(dy)

                        if len(motion_history[track_id])>FRAME_CONFIRM:
                            motion_history[track_id].pop(0)

                        avg_dy=sum(motion_history[track_id])/len(motion_history[track_id])

                        if avg_dy>DIRECTION_THRESHOLD:
                            wrong_ids.add(track_id)

                    prev_y[track_id]=cy

                    if track_id in wrong_ids:

                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                        cv2.putText(frame,"WRONG LANE",(x1,y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),3)

        # ================= OVERSPEED =================

        elif violation_type == "Overspeed":

            LINE_1=int(H*0.40)
            LINE_2=int(H*0.65)

            cv2.line(frame,(0,LINE_1),(W,LINE_1),(255,0,0),2)
            cv2.line(frame,(0,LINE_2),(W,LINE_2),(0,255,255),2)

            results=model.track(frame,persist=True)

            for r in results:

                if r.boxes.id is None:
                    continue

                for box,track_id,cls in zip(r.boxes.xyxy,r.boxes.id,r.boxes.cls):

                    track_id=int(track_id)
                    cls=int(cls)

                    if cls not in VEHICLE_CLASSES:
                        continue

                    x1,y1,x2,y2=map(int,box)
                    cx=(x1+x2)//2
                    cy=(y1+y2)//2

                    if cy>LINE_1 and track_id not in vehicle_entry:
                        vehicle_entry[track_id]=(frame_no,cy)

                    if cy>LINE_2 and track_id in vehicle_entry and track_id not in vehicle_speed:

                        start_frame,start_y=vehicle_entry[track_id]

                        time_taken=(frame_no-start_frame)/25

                        if time_taken>0:

                            pixel_dist=abs(cy-start_y)
                            speed=pixel_dist/time_taken

                            vehicle_speed[track_id]=speed

                            if speed>180:
                                overspeed_locked.add(track_id)

                    if track_id in overspeed_locked:

                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                        cv2.putText(frame,"OVER SPEED",(x1,y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

        ret,buffer=cv2.imencode(".jpg",frame)
        frame_bytes=buffer.tobytes()

        yield(
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'+frame_bytes+b'\r\n'
        )

    cap.release()

@app.route("/video_feed")
def video_feed():

    video=request.args.get("video")
    violation_type=request.args.get("type")

    return Response(
        generate_frames(video,violation_type),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# ---------------- MAIN ----------------

if __name__=="__main__":

    os.makedirs("static/uploads",exist_ok=True)
    os.makedirs("static/evidence",exist_ok=True)
    os.makedirs("static/challans",exist_ok=True)

    with app.app_context():
        db.create_all()

    app.run(debug=True)

