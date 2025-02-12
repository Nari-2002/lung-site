from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "inceptionv3_lung_disease_classifier.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define the class names
CLASS_NAMES = ['COVID', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

# Define the folder to store uploaded files
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def is_valid_lung_image(image):
    """
    Check whether the uploaded image is likely to be a lung X-ray.
    """
    grayscale = image.convert('L')  # Convert to grayscale
    image_array = np.array(grayscale)

    # Simple heuristic check: X-rays tend to have a wide grayscale range but are not colorful
    if len(np.unique(image_array)) < 50:  # If very few unique pixel intensities, reject
        return False
    return True

def preprocess_image(image_path):
    """
    Preprocess the image to the required format for the model.
    """
    image = Image.open(image_path).convert('RGB')  # Ensure 3 channels

    # Validate if the image is a lung X-ray
    if not is_valid_lung_image(image):
        return None  # Return None if the image is not valid

    image = image.resize((299, 299))  # Resize to match InceptionV3 input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

@app.route('/')
def index():
    """
    Render the home page.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle file upload and return prediction as an HTML page.
    """
    if 'file' not in request.files:
        return render_template('error.html', message="No file uploaded.")

    file = request.files['file']
    if file.filename == '':
        return render_template('error.html', message="No selected file.")

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        image_data = preprocess_image(filepath)

        if image_data is None:
            os.remove(filepath)
            return render_template('error.html', message="Invalid image. Please upload a lung X-ray.")

        # Predict the class
        predictions = model.predict(image_data)
        predicted_class = CLASS_NAMES[np.argmax(predictions)]
        confidence = float(np.max(predictions))

        # Clean up the uploaded file
        os.remove(filepath)

        return render_template('result.html', predicted_class=predicted_class, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
