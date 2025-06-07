from __future__ import division, print_function
# coding=utf-8
import sys
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf

# Define the Flask app
app = Flask(__name__)

MODEL_PATH = 'Skin_Diseases.keras'  # Updated model path for .keras format
try:
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()

# Detailed suggestions and treatments for each disease
disease_treatments = {
    'Acne': {
        'suggestions': [
            "Wash your face twice daily with a gentle cleanser.",
            "Avoid touching your face and picking at pimples.",
            "Use non-comedogenic and oil-free cosmetics."
        ],
        'medications': [
            {"name": "Benzoyl Peroxide", "details": "Kills acne-causing bacteria and reduces inflammation."},
            {"name": "Salicylic Acid", "details": "Helps unclog pores and exfoliate dead skin cells."},
            {"name": "Topical Retinoids", "details": "Encourage skin cell turnover and reduce clogging."}
        ]
    },
    'Melanoma': {
        'suggestions': [
            "Seek immediate consultation with a dermatologist or oncologist.",
            "Avoid exposure to direct sunlight and use broad-spectrum sunscreen.",
            "Monitor for changes in moles or skin lesions."
        ],
        'medications': [
            {"name": "Immunotherapy", "details": "Boosts the body's natural defenses to fight cancer."},
            {"name": "Targeted Therapy", "details": "Focuses on specific genetic changes in melanoma cells."},
            {"name": "Surgical Excision", "details": "Removal of the melanoma along with some healthy tissue."}
        ]
    },
    'Psoriasis': {
        'suggestions': [
            "Keep your skin well-moisturized with heavy creams or ointments.",
            "Avoid known triggers like stress, infections, or certain medications.",
            "Use warm baths with Epsom salts to reduce itching and scaling."
        ],
        'medications': [
            {"name": "Corticosteroids", "details": "Reduce inflammation and suppress the immune response."},
            {"name": "Vitamin D Analogues", "details": "Slow skin cell growth and flatten psoriasis plaques."},
            {"name": "Biologic Drugs", "details": "Target specific parts of the immune system."}
        ]
    },
    'Rosacea': {
        'suggestions': [
            "Identify and avoid triggers such as spicy foods, alcohol, and extreme temperatures.",
            "Use gentle, fragrance-free skincare products.",
            "Protect your skin with sunscreen daily."
        ],
        'medications': [
            {"name": "Topical Metronidazole", "details": "Reduces redness and inflammation."},
            {"name": "Azelaic Acid", "details": "Improves rosacea symptoms and reduces swelling."},
            {"name": "Oral Antibiotics", "details": "Prescribed for severe cases to reduce inflammation."}
        ]
    },
    'Vitiligo': {
        'suggestions': [
            "Use sunscreen daily to protect against sunburn and reduce contrast between affected and unaffected skin.",
            "Consider cosmetic options like makeup or self-tanning products to cover affected areas.",
            "Discuss phototherapy options with a dermatologist."
        ],
        'medications': [
            {"name": "Topical Corticosteroids", "details": "Help repigment the skin in early stages."},
            {"name": "Calcineurin Inhibitors", "details": "Effective in small areas, especially on the face."},
            {"name": "Psoralen and UV-A Therapy (PUVA)", "details": "Combines a drug and light therapy to repigment skin."}
        ]
    }
}

# Route to display the main page (index.html)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route to handle the prediction logic
@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the uploaded image file
        f = request.files['file']

        # Save the file to ./uploads folder
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(f"Image saved to: {file_path}")

        try:
            # Preprocess the image for prediction
            img = image.load_img(file_path, target_size=(64, 64))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)

            # Normalize the image data (assuming your model was trained with normalization)
            x = x / 255.0

            # Predict with the model
            preds = model.predict(x)
            print(f"Raw predictions: {preds}")  # Print raw predictions for debugging

            # Map prediction to the disease names
            index = ['Acne', 'Melanoma', 'Psoriasis', 'Rosacea', 'Vitiligo']
            predicted_class = index[np.argmax(preds[0])]
            confidence = float(np.max(preds))

            # Get treatment details
            treatment = disease_treatments.get(predicted_class, {})
            suggestions = treatment.get('suggestions', [])
            medications = treatment.get('medications', [])

            # Add note
            note = "These treatments and suggestions are general recommendations. It is important to consult with a healthcare professional, such as a dermatologist, for accurate diagnosis and tailored treatment"

            # Return prediction result with suggestions, medications, and the note as JSON
            response = {
                "disease": predicted_class,
                "confidence": confidence,
                "suggestions": suggestions,
                "medications": medications,
                "note": note
            }

            return jsonify(response)

        except Exception as e:
            return jsonify({"error": str(e)})
        finally:
            # Clean up the uploaded image
            os.remove(file_path)
            print(f"Image removed from: {file_path}")

if __name__ == '__main__':
    app.run(debug=False, threaded=False)