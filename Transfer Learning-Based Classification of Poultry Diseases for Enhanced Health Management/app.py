import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load your trained model
def dummy_predict(img_array):
    import random
    class_names = ['Coccidiosis', 'New Castle Disease', 'Salmonella', 'Healthy']
    return random.choice(class_names)
# Change path if different
class_names = ['Coccidiosis', 'New Castle Disease', 'Salmonella', 'Healthy']

# Make sure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Dummy prediction
    predicted_class = dummy_predict(None)

    return render_template('index.html', prediction=predicted_class)


    # Image preprocessing for prediction
    img = image.load_img(filepath, target_size=(224, 224))  # adjust as per your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize if needed

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
