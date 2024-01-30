import logging
import os
import json
from flask import Flask, request, jsonify
from catboost import CatBoostClassifier

VERSION = 0.99
app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
model_path = os.path.join("/app/var", f'model_v{VERSION}.cbm')
model = CatBoostClassifier().load_model(model_path)


@app.route('/model/info', methods=['GET'])
def list_models():
    try:
        with open(os.path.join("/app/var", 'DESCR'), "r") as f:
            response = f.read()
        return response
    except FileNotFoundError:
        return jsonify({'error': 'Model description file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model/predict', methods=['POST'])
def predict(model):
    try:
        data = request.get_json()
        features = data.get('features')
        feature_names = []
        features_list = [features[each] for each in feature_names]
        prediction = model.predict(features_list)
        response = {
            'model_version': VERSION,
            # Convert prediction to a list for JSON serialization
            'prediction': prediction.tolist()
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
