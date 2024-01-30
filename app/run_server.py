from flask import Flask, request, jsonify
from catboost import CatBoostClassifier
import json
import logging
import os

app = Flask(__name__)
app.logger.setLevel(logging.DEBUG)

models = {}  # Dictionary to store loaded models
models_directory = "/app/var"

@app.route('/model/list', methods=['GET'])
def list_models():
    try:
        with open(os.path.join(models_directory, 'DESCR'), "r") as f:
            response = f.read()
        return response
    except FileNotFoundError:
        return jsonify({'error': 'Model description file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model/predict/<model_name>', methods=['POST'])
def predict(model_name):
    try:
        data = request.get_json()

        model_version = data.get('model_version')
        features = data.get('features')
        feature_names=[
            "education",
            "MonthProfit",
            "MonthExpense",
            "ChildCount",
            "Loan_amount",
            "Loan_term",
            "employment_loyalty",
            "employment_status_dekretnyj_otpusk",
            "employment_status_pensioner",
            "employment_status_rabotaju",
            "employment_status_rabotaju_najmu_nepolnyj_rabochij_den_",
            "employment_status_rabotaju_najmu_polnyj_rabochij_den_sluzhu",
            "employment_status_sobstvennoe_delo",
            "employment_status_student",
            "Family_status_brake_sostojal",
            "Family_status_grazhdanskij_brak_sovmestnoe_prozhivanie",
            "Family_status_nan",
            "Family_status_razveden_razvedena",
            "Family_status_vdovets_vdova",
            "Family_status_zhenat_zamuzhem",
            "age"]
        
        features_list = [features[each] for each in feature_names]

        if model_name not in models:
            models[model_name] = {}

        if model_version not in models[model_name]:
            model_path = os.path.join(models_directory, f'{model_name}_v{model_version}.cbm')
            models[model_name][model_version] = CatBoostClassifier().load_model(model_path)

        model = models[model_name][model_version]
        prediction = model.predict(features_list)

        response = {
            'model_name': model_name,
            'model_version': model_version,
            'prediction': prediction.tolist()  # Convert prediction to a list for JSON serialization
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
