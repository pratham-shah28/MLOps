from flask import Flask, request, jsonify
from predict import predict_wine
import os
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Accept JSON or form-encoded
    data = request.get_json(silent=True) or request.form

    required = [
        'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
        'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
        'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'
    ]

    # Basic validation
    missing = [k for k in required if k not in data]
    if missing:
        return jsonify({'error': f'missing fields: {missing}'}), 400

    try:
        features = [
            float(data['alcohol']),
            float(data['malic_acid']),
            float(data['ash']),
            float(data['alcalinity_of_ash']),
            float(data['magnesium']),
            float(data['total_phenols']),
            float(data['flavanoids']),
            float(data['nonflavanoid_phenols']),
            float(data['proanthocyanins']),
            float(data['color_intensity']),
            float(data['hue']),
            float(data['od280/od315_of_diluted_wines']),
            float(data['proline'])
        ]
    except (TypeError, ValueError) as e:
        return jsonify({'error': f'invalid input types: {str(e)}'}), 400

    pred = predict_wine(features)
    return jsonify({'prediction': int(pred)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
