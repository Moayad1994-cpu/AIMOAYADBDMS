from flask import Flask, render_template, request, redirect, url_for, send_from_directory, send_file
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import shutil
import zipfile

# Flask configuration
app = Flask(__name__)

# Paths
MODEL_PATH = r"flask/keras_model.h5"  # Path to your model
LABELS_PATH = r"flask/labels.txt"    # Path to your labels file
UPLOAD_FOLDER = "uploads"     # Folder to save uploaded images
OUTPUT_FOLDER = r"D:/AI LEARN/model classfiy/.venv"  # Folder to save classified images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload and output folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the model and labels
try:
    print("Loading the Keras model...")
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

try:
    print("Loading labels...")
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"Labels loaded: {class_names}")
except Exception as e:
    print(f"Error loading labels: {e}")
    exit()

# Ensure subfolders for each label exist
for label in class_names:
    os.makedirs(os.path.join(OUTPUT_FOLDER, label), exist_ok=True)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess the image to the required input shape for the model."""
    try:
        image = Image.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        return data
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def classify_image(image_path):
    """Classify the image and return the predicted label and confidence score."""
    data = preprocess_image(image_path)
    if data is None:
        return "Error", 0.0
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

@app.route('/', methods=['GET', 'POST'])
def index():
    """Home page for uploading and processing images."""
    if request.method == 'POST':
        if 'files' not in request.files:
            return redirect(request.url)
        files = request.files.getlist('files')  # Handle multiple file uploads
        results = []

        for file in files:
            if file and allowed_file(file.filename):
                # Save the uploaded file
                filename = file.filename
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)

                # Classify the image
                predicted_label, confidence_score = classify_image(filepath)

                if predicted_label != "Error":
                    # Rename and move the image to the corresponding folder
                    new_name = f"{os.path.splitext(filename)[0]}_{predicted_label}.png"
                    output_path = os.path.join(OUTPUT_FOLDER, predicted_label, new_name)
                    shutil.move(filepath, output_path)

                    # Append result
                    results.append({
                        'label': predicted_label,
                        'confidence': confidence_score,
                        'filename': new_name,
                        'path': output_path
                    })
                else:
                    results.append({
                        'label': "Error",
                        'confidence': 0.0,
                        'filename': filename,
                        'path': None
                    })

        return render_template(
            'index.html',
            results=results,
            message="Images classified successfully!",
        )

    return render_template('index.html', results=None, message=None)

@app.route('/classified/<label>/<filename>')
def classified_file(label, filename):
    """Serve classified files."""
    return send_from_directory(os.path.join(OUTPUT_FOLDER, label), filename)

@app.route('/download/<label>/<filename>')
def download_file(label, filename):
    """Download the classified file."""
    return send_from_directory(os.path.join(OUTPUT_FOLDER, label), filename, as_attachment=True)

@app.route('/download_all')
def download_all():
    """Download all classified images as a zip file."""
    zip_filename = "classified_images.zip"
    zip_path = os.path.join(OUTPUT_FOLDER, zip_filename)

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for label in class_names:
            label_folder = os.path.join(OUTPUT_FOLDER, label)
            for root, _, files in os.walk(label_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, OUTPUT_FOLDER)
                    zipf.write(file_path, arcname)

    return send_file(zip_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
