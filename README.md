# 🚦 Traffic Violation Detection System

The **Traffic Violation Detection System** is an AI-powered web application that automatically detects traffic rule violations using computer vision and deep learning. The system uses the YOLOv8 object detection model along with OpenCV to analyze traffic videos and identify violations such as helmet violations, red-light jumping, wrong-lane driving, and over-speeding.

When a violation is detected, the system captures evidence images and generates digital challans automatically. All violation records are stored in a database and can be monitored through a web dashboard.

---

## 🚀 Features

* 🚦 Red Light Violation Detection
* 🪖 Helmet Detection
* 🚗 Wrong Lane Detection
* ⚡ Over-speeding Detection
* 📸 Automatic Evidence Capture
* 📄 Auto Challan Generation
* 📊 Analytics Dashboard
* 🎥 Video Upload and Processing

---

## 🛠️ Technologies Used

* **Python**
* **Flask**
* **YOLOv8 (Ultralytics)**
* **OpenCV**
* **SQLAlchemy**
* **HTML, CSS, JavaScript**
* **Matplotlib**

---

## 📂 Project Structure

Traffic-Violation-Detection-System
│
├── app.py
├── config.py
├── requirements.txt
│
├── models/
│   └── yolov8n.pt
│
├── templates/
│   ├── index.html
│   ├── detect.html
│   ├── analytics.html
│   ├── challans.html
│   └── evidence.html
│
├── static/
│   ├── css/
│   ├── js/
│   ├── uploads/
│   ├── evidence/
│   └── challans/

---

## ⚙️ Installation

Clone the repository:

git clone https://github.com/Samay24/Traffic-Violation-Detection-System.git

Go to project directory:

cd Traffic-Violation-Detection-System

Install dependencies:

pip install -r requirements.txt

Run the application:

python app.py

---

## 📊 How It Works

1. User uploads a traffic video.
2. YOLOv8 model detects vehicles and violations.
3. Violations are recorded with evidence images.
4. System generates challans automatically.
5. All data is displayed in the analytics dashboard.

---

## 📌 Future Improvements

* License plate recognition (ANPR)
* Real-time CCTV integration
* Automated fine payment system
* Cloud deployment

---


