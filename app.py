from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # ✅ Enables cross-origin requests from Android

# Load model and encoders
model = joblib.load("career_model.pkl")
le_interest = joblib.load("interest_encoder.pkl")
le_brain = joblib.load("brain_encoder.pkl")
le_career = joblib.load("career_encoder.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        iq = data.get('iq')
        interest = data.get('interest')
        brain = data.get('brain')

        # Encode inputs
        interest_encoded = le_interest.transform([interest])[0]
        brain_encoded = le_brain.transform([brain])[0]

        # Predict
        prediction = model.predict([[iq, interest_encoded, brain_encoded]])[0]
        career = le_career.inverse_transform([prediction])[0]

        return jsonify({'career': career})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
