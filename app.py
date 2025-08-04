from flask import Flask, render_template_string, request
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('skin_cancer_model.h5')
classes = sorted([d for d in os.listdir('data') if os.path.isdir(os.path.join('data', d))])

HTML_FORM = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Skin Cancer Detection</title>
  <link href="https://fonts.googleapis.com/css?family=Roboto:400,700&display=swap" rel="stylesheet">
  <style>
    body {
      font-family: 'Roboto', Arial, sans-serif;
      background: #f5f6fa;
      margin: 0;
      padding: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
    }
    .container {
      background: #fff;
      border-radius: 12px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.08);
      padding: 32px 40px;
      max-width: 400px;
      width: 100%;
      text-align: center;
    }
    h2 {
      color: #222;
      margin-bottom: 24px;
    }
    .upload-btn {
      background: #0984e3;
      color: #fff;
      border: none;
      padding: 10px 24px;
      border-radius: 6px;
      font-size: 16px;
      cursor: pointer;
      margin-top: 12px;
      margin-bottom: 16px;
      transition: background 0.2s;
    }
    .upload-btn:hover {
      background: #74b9ff;
    }
    .result-card {
      background: #f1f2f6;
      border-radius: 8px;
      padding: 18px;
      margin-top: 18px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .prediction {
      font-size: 20px;
      font-weight: 700;
      color: #636e72;
      margin-bottom: 10px;
    }
    .uploaded-img {
      margin-top: 10px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
      max-width: 100%;
      height: auto;
    }
    label {
      font-weight: 500;
      color: #636e72;
    }
  </style>
</head>
<body>
  <div class="container">
    <h2>Skin Cancer Detection</h2>
    <form method=post enctype=multipart/form-data>
      <label for="image">Upload a skin lesion image:</label><br><br>
      <input type=file name=image id="image" accept="image/*">
      <br>
      <input type=submit value=Predict class="upload-btn">
    </form>
    {% if prediction %}
      <div class="result-card">
        <div class="prediction">Prediction: {{ prediction }}</div>
        {% if image_data %}
          <img src="data:image/jpeg;base64,{{ image_data }}" class="uploaded-img" />
        {% endif %}
      </div>
    {% endif %}
  </div>
</body>
</html>
'''

def preprocess_image(img_path, img_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    prediction = None
    image_data = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join('uploads', file.filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(filepath)
            img = preprocess_image(filepath)
            pred = model.predict(img)
            class_idx = np.argmax(pred)
            confidence = float(pred[0][class_idx])
            prediction = f"{classes[class_idx]} (Confidence: {confidence:.2f})"
            # Convert image to base64 for display
            import base64
            with open(filepath, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            os.remove(filepath)
    return render_template_string(HTML_FORM, prediction=prediction, image_data=image_data)

if __name__ == '__main__':
    app.run(debug=True)
