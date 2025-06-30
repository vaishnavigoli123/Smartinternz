from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
model = load_model("poultry_model.h5")  # replace with your actual .h5 file name
class_labels = ['Coccidiosis', 'Healthy', 'NewCastle', 'Salmonella']

# ⬇️ Fix: Define preprocessing function
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    prediction = None
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            save_path = os.path.join('static', 'uploads', file.filename)
            file.save(save_path)

            img = preprocess_image(save_path)
            preds = model.predict(img)[0]
            predicted_label = class_labels[np.argmax(preds)]
            prediction = f"The infection type detected as {predicted_label}"
        else:
            prediction = "Unsupported file format. Please upload a JPG, JPEG, or PNG."
    return render_template('prediction.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

