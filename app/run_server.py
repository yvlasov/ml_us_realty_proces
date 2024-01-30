import logging
import os
import json
from flask import Flask, request, jsonify
from catboost import CatBoostClassifier

VERSION = "0.99.0"
app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)
model_path = os.path.join("/app/var", f'model_v{VERSION}.cbm')
model = CatBoostClassifier().load_model(model_path)

try:
    with open(os.path.join("/app/var", 'features.json'), "r") as f:
        features_json = f.read()
except FileNotFoundError:
    print('error: Model features file not found')
    sys.exit(1)
except Exception as e:
    print(f'error: {str(e)}')
    sys.exit(1)

feature_names = json.loads(features_json)


def find_missing_keys(key_list, my_dict):
    # Use a list comprehension to find missing keys
    missing_keys = [key for key in key_list if key not in my_dict]

    return missing_keys


@app.route('/model/info', methods=['GET'])
def model_info():
    try:
        with open(os.path.join("/app/var", 'DESCR'), "r") as f:
            response = f.read()
    except FileNotFoundError:
        return jsonify({'error': 'Model description file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    return response


@app.route('/model/features', methods=['GET'])
def list_features():
    return features_json

@app.route('/model/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features')
        missing_features_list = find_missing_keys(feature_names, features)
        if len(missing_features_list) != 0:
            return jsonify(f"Missing features: {missing_features_list}"), 400
        
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
