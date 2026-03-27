from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load model
model = load_model('model/plant_disease_model.h5')

# Class labels (same as your training)
classes = ['Tomato_Early_blight', 'Tomato_Healthy', 'Tomato_Late_blight']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)

    result = classes[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100

    return result, round(confidence, 2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file:
        filepath = os.path.join('static', file.filename)
        file.save(filepath)

        result, confidence = predict_image(filepath)

        return render_template(
            'result.html',
            prediction=result,
            confidence=confidence,
            img_path=filepath
        )

    return "No file uploaded"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)