from flask import Flask, render_template, request, url_for
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import json
import os

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, Model

app = Flask(__name__)

# Base directory (absolute path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure upload folder exists
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASS_INDICES_PATH = os.path.join(BASE_DIR, 'class_indices.json')
WEIGHTS_PATH = os.path.join(BASE_DIR, 'efficientnet.weights.h5')
CSV_PATH = os.path.join(BASE_DIR, 'data-info.csv')

# ---------------------------------------------------------
#              LOAD LABELS + REBUILD MODEL
# ---------------------------------------------------------
model = None
class_labels = {}
model_height, model_width = 224, 224

try:
    print("Loading EfficientNet model...")

    if not os.path.exists(CLASS_INDICES_PATH):
        raise FileNotFoundError(f"class_indices.json not found at {CLASS_INDICES_PATH}")

    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)

    class_labels = {str(v): k for k, v in class_indices.items()}
    NUM_CLASSES = len(class_indices)
    if NUM_CLASSES == 0:
        raise ValueError("class_indices.json is empty or invalid")

    IMG_SIZE = 224

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model = EfficientNetB0(
        include_top=False,
        weights=None,
        input_tensor=inputs
    )

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs, outputs)

    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"efficientnet.weights.h5 not found at {WEIGHTS_PATH}")

    print(f"Loading weights from: {WEIGHTS_PATH}")
    model.load_weights(WEIGHTS_PATH)

    model_height, model_width = IMG_SIZE, IMG_SIZE
    print("✅ EfficientNet model loaded successfully!")
    print("   Input size:", model_height, "x", model_width)
    print("   Num classes:", NUM_CLASSES)

except Exception as e:
    print("\n❌ Error loading EfficientNet model:")
    print("   ", repr(e))
    model = None
    class_labels = {}
    model_height, model_width = 224, 224

# ---------------------------------------------------------
#              LOAD SUPPLEMENT CSV
# ---------------------------------------------------------
try:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"data-info.csv not found at {CSV_PATH}")

    supplement_data = pd.read_csv(CSV_PATH)
    print("📄 Supplement data loaded successfully!")
except Exception as e:
    print("\n❌ Error loading supplement data:")
    print("   ", repr(e))
    supplement_data = pd.DataFrame()

# ---------------------------------------------------------
#              PREDICTION FUNCTION
# ---------------------------------------------------------
def predict_image_class(model, image_path):
    if model is None or not class_labels:
        return "Model not loaded"

    try:
        img = Image.open(image_path).convert('RGB').resize((model_width, model_height))
        img_array = np.array(img).astype('float32')

        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        pred_idx = int(np.argmax(preds[0]))

        pred_class = class_labels.get(str(pred_idx), "Unknown class")
        print("PREDICTION:", pred_class)
        return pred_class
    except Exception as e:
        print("\n❌ Error during prediction:")
        print("   ", repr(e))
        return "Prediction error"

# ---------------------------------------------------------
#              FLASK ROUTE
# ---------------------------------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    supplement_name = 'No supplement found'
    product_link = '#'
    prevention_tips = 'No prevention tips available'
    care_tips = 'No care tips available'

    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename != '':
            image_filename = file.filename
            image_full_path = os.path.join(UPLOAD_FOLDER, image_filename)
            file.save(image_full_path)

            prediction = predict_image_class(model, image_full_path)

            if not supplement_data.empty and prediction not in ["Model not loaded", "Prediction error", "Unknown class"]:
                info = supplement_data[supplement_data['disease_name'] == prediction]
                if not info.empty:
                    supplement_name = info.iloc[0].get('supplement_name', supplement_name)
                    product_link = info.iloc[0].get('product_link', product_link)
                    prevention_tips = info.iloc[0].get('prevention_tips', prevention_tips)
                    care_tips = info.iloc[0].get('care_tips', care_tips)

            image_path = url_for('static', filename=f'uploads/{image_filename}')

    return render_template(
        'index.html',
        prediction=prediction,
        image_path=image_path,
        supplement_name=supplement_name,
        product_link=product_link,
        prevention_tips=prevention_tips,
        care_tips=care_tips
    )

if __name__ == '__main__':
    app.run(debug=True)
