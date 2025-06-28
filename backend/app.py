from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.utils import secure_filename
from torchvision import transforms
from datetime import datetime
from PIL import Image
import pandas as pd
import torch
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "../uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'
db = SQLAlchemy(app)

model = torch.load("../ai_model/model.pth", map_location=torch.device('cpu'), weights_only=False)
model.eval()

classes = ['eczema', 'ringworm', 'psoriasis']

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100))
    prediction = db.Column(db.String(20))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def save_to_excel(filename, label):
    file = "predictions.xlsx"
    row = {
        "filename": [filename],
        "prediction": [label],
        "timestamp": [datetime.now()]
    }
    df_new = pd.DataFrame(row)

    if os.path.exists(file):
        df_existing = pd.read_excel(file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_excel(file, index=False)
    else:
        df_new.to_excel(file, index=False)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    input_tensor = preprocess_image(filepath)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = classes[predicted.item()]

    record = Prediction(filename=filename, prediction=label)
    db.session.add(record)
    db.session.commit()

    save_to_excel(filename, label)

    return jsonify({"prediction": label})

@app.route('/history', methods=['GET'])
def history():
    records = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    return jsonify([
        {
            "filename": r.filename,
            "prediction": r.prediction,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        } for r in records
    ])

if __name__ == '__main__':
    with app.app_context():
        print("📦 Creating database if not exists...")
        db.create_all()
    app.run(debug=True)


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
