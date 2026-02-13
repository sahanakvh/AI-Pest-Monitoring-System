from datetime import datetime
import pytz
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask import send_file
import io
import sqlite3
from datetime import datetime
import requests
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease TEXT,
            confidence REAL,
            temperature REAL,
            humidity REAL,
            risk TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()
# Load trained model
model = tf.keras.models.load_model("pest_model.h5", compile=False)

class_names = ['Early_blight', 'Late_blight', 'Septoria_leaf_spot', 'healthy']

API_KEY = "dcd1c978dd016f397ae17ae92341bfc9"

TREATMENT_GUIDE = {
    "Early_blight": {
        "recommendation": "Spray Mancozeb or Chlorothalonil fungicide every 7 days.",
        "prevention": "Avoid overhead irrigation and remove infected leaves."
    },
    "Late_blight": {
        "recommendation": "Use Metalaxyl-based fungicide immediately.",
        "prevention": "Ensure proper drainage and avoid high humidity."
    },
    "Septoria_leaf_spot": {
        "recommendation": "Apply Copper-based fungicide.",
        "prevention": "Rotate crops and remove plant debris."
    },
    "healthy": {
        "recommendation": "No treatment required.",
        "prevention": "Maintain regular monitoring and balanced fertilization."
    }
}

IMG_SIZE = 224


def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    print("Weather API response:", data)

    if "main" not in data:
        return 0, 0

    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]

    return temp, humidity


@app.route("/")
def home():
    return "Pest Detection API is Running 🚀"


@app.route("/ui")
def ui():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    city = request.form.get("city", "Bangalore")

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))
    print("Raw confidence:", confidence)
    predicted_class = class_names[class_index]
    # Create probability dictionary
    probabilities = {
         class_names[i]: round(float(prediction[0][i]) * 100, 2)
         for i in range(len(class_names))
    }
    temp, humidity = get_weather(city)

    if humidity > 80:
        risk = "High"
    elif humidity > 60:
        risk = "Moderate"
    else:
        risk = "Low"

    treatment = TREATMENT_GUIDE.get(predicted_class, {})
    # Save prediction to database
    conn = sqlite3.connect("history.db")
    c = conn.cursor()

    c.execute("""
        INSERT INTO predictions 
        (disease, confidence, temperature, humidity, risk, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        predicted_class,
        confidence,   # already percentage
        temp,
        humidity,
        risk,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

    return jsonify({
        "disease": predicted_class,
        "confidence": round(confidence * 100, 2),
        "temperature": temp,
        "humidity": humidity,
        "risk_level": risk,
        "recommendation": treatment.get("recommendation", "N/A"),
        "prevention": treatment.get("prevention", "N/A")
    })


import os
from werkzeug.utils import secure_filename

@app.route("/predict-ui", methods=["POST"])
def predict_ui():
    file = request.files["image"]
    city = request.form.get("city")

    # Save uploaded image
    filename = secure_filename(file.filename)
    upload_path = os.path.join("static/uploads", filename)
    file.save(upload_path)

    # Read image for prediction
    img = cv2.imread(upload_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) 
    predicted_class = class_names[class_index]
    probabilities = {
   	 class_names[i]: round(float(prediction[0][i]) * 100, 2)
   	 for i in range(len(class_names))
    }
    temp, humidity = get_weather(city)

    if humidity > 80:
        risk = "High"
    elif humidity > 60:
        risk = "Moderate"
    else:
        risk = "Low"
    # Determine severity
    if confidence > 85 and humidity > 70:
    	severity = "Severe"
    elif confidence > 70:
    	severity = "Moderate" 
    else:
    	severity = "Mild"
    treatment = TREATMENT_GUIDE.get(predicted_class, {})
    # Save to database
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    ist = pytz.timezone('Asia/Kolkata')
    timestamp = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
         INSERT INTO predictions 
         (disease, confidence, temperature, humidity, risk, timestamp)
         VALUES (?, ?, ?, ?, ?, ?)
    """, (
    	predicted_class,
    	round(confidence, 2),
    	temp,
    	humidity,
    	risk,
        timestamp
    ))
    conn.commit()
    conn.close()
    return render_template(
    	"result.html",
    	disease=predicted_class,
   	confidence=round(confidence * 100, 2),
    	temperature=temp,
    	humidity=humidity,
    	risk=risk,
        severity=severity,
    	recommendation=treatment.get("recommendation", "N/A"),
    	prevention=treatment.get("prevention", "N/A"),
    	probabilities=probabilities
    )
@app.route("/download-report")
def download_report():

    # Get latest prediction from DB
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    conn.close()

    if not row:
        return "No report available"

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)

    pdf.drawString(100, 750, "AI Crop Disease Analysis Report")
    pdf.drawString(100, 720, f"Disease: {row[1]}")
    pdf.drawString(100, 700, f"Confidence: {row[2]}%")
    pdf.drawString(100, 680, f"Temperature: {row[3]}°C")
    pdf.drawString(100, 660, f"Humidity: {row[4]}%")
    pdf.drawString(100, 640, f"Risk Level: {row[5]}")
    pdf.drawString(100, 620, f"Date: {row[6]}")

    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Crop_Analysis_Report.pdf",
        mimetype="application/pdf"
    )
@app.route("/history")
def history():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()

    c.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = c.fetchall()

    # Total predictions
    total_predictions = len(rows)

    # Most common disease
    c.execute("""
        SELECT disease, COUNT(*) as count 
        FROM predictions 
        GROUP BY disease 
        ORDER BY count DESC 
        LIMIT 1
    """)
    most_common = c.fetchone()
    most_common_disease = most_common[0] if most_common else "N/A"

    # Average confidence
    c.execute("SELECT AVG(confidence) FROM predictions")
    avg_conf = c.fetchone()[0]
    avg_confidence = round(avg_conf, 2) if avg_conf else 0

    # High risk count
    c.execute("SELECT COUNT(*) FROM predictions WHERE risk='High'")
    high_risk = c.fetchone()[0]

    conn.close()

    return render_template(
        "history.html",
        rows=rows,
        total_predictions=total_predictions,
        most_common_disease=most_common_disease,
        avg_confidence=avg_confidence,
        high_risk=high_risk
    )
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
  
