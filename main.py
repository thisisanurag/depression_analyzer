import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for development

# Load the dataset
df = pd.read_excel('phq9.xlsx')

# Prepare the text data for MentalBERT
texts = df.iloc[:, 1:10].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Load MentalBERT model and tokenizer
# Please insert your HF_TOKEN here
HF_TOKEN = ""
tokenizer = AutoTokenizer.from_pretrained("mental/mental-bert-base-uncased", use_auth_token=HF_TOKEN)
model = AutoModel.from_pretrained("mental/mental-bert-base-uncased", use_auth_token=HF_TOKEN)

# Function to get MentalBERT embeddings
def get_bert_embeddings(texts):
    inputs = tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Use the [CLS] token embedding

# Get MentalBERT embeddings
embeddings = get_bert_embeddings(texts)

# Prepare the target variable
y = df['PHQ-9 Score'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train_scaled, y_train)

# Comment out test cases and evaluation for faster startup

# Make predictions
y_pred = rf_regressor.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"R-squared Score: {r2}")

# Feature importance
feature_importance = rf_regressor.feature_importances_
for i, v in enumerate(feature_importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

# If you want to see how the model performs on specific examples:
for i in range(3):  # Print predictions for first 5 test samples
    print(f"Actual: {y_test[i]}, Predicted: {y_pred[i]}")


def get_severity_level(score):
    if score < 5:
        return "Minimal"
    elif score < 10:
        return "Mild"
    elif score < 15:
        return "Moderate"
    elif score < 20:
        return "Moderately Severe"
    else:
        return "Severe"

def get_recommendations(score):
    if score < 5:
        return [
            "Monitor patient's mood",
            "Consider follow-up in 1-2 months",
            "Encourage healthy lifestyle habits"
        ]
    elif score < 10:
        return [
            "Consider watchful waiting",
            "Repeat assessment in 1 month",
            "Consider brief counseling or psychotherapy"
        ]
    elif score < 15:
        return [
            "Active treatment with psychotherapy",
            "Consider antidepressant medication",
            "Close follow-up within 2-4 weeks"
        ]
    elif score < 20:
        return [
            "Active treatment with medication",
            "Psychotherapy recommended",
            "Close follow-up within 1-2 weeks"
        ]
    else:
        return [
            "Immediate treatment with medication",
            "Psychotherapy strongly recommended",
            "Consider referral to mental health specialist",
            "Close follow-up within 1 week"
        ]

def predict_phq9_score(text):
    try:
        # Get MentalBERT embeddings for the input text
        inputs = tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        text_embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
        # Scale the embedding
        text_embedding_scaled = scaler.transform(text_embedding)
        
        # Make prediction
        score = rf_regressor.predict(text_embedding_scaled)[0]
        
        # Ensure score is between 0 and 27
        final_score = float(max(0, min(27, score)))
            
        return final_score
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return 0.0  # Return minimal score if prediction fails

@app.route('/assess', methods=['POST', 'OPTIONS'])
def assess():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        # Log the incoming request data for debugging
        print("Request headers:", request.headers)
        print("Request content type:", request.content_type)
        print("Raw request data:", request.get_data())
        
        # Try to parse JSON data
        try:
            data = request.get_json()
            print("Parsed JSON data:", data)
            
            # Handle the specific frontend format
            if isinstance(data, dict):
                if 'description' in data:
                    text = data['description']
                elif 'patientInfo' in data and 'description' in data['patientInfo']:
                    text = data['patientInfo']['description']
                elif 'text' in data:
                    text = data['text']
                else:
                    return jsonify({
                        'error': 'Invalid data format. Expected {description: "text"} or {patientInfo: {...}, description: "text"} or {text: "text"}',
                        'received_data': data
                    }), 400
            else:
                return jsonify({
                    'error': 'Invalid data format. Expected JSON object',
                    'received_data': data
                }), 400
            
            if not text:
                return jsonify({
                    'error': 'No description found in request',
                    'received_data': data
                }), 400
            
            # Get PHQ-9 score
            score = predict_phq9_score(text)
            
            # Get severity level and recommendations
            severity = get_severity_level(score)
            recommendations = get_recommendations(score)
            
            response = {
                'score': score,
                'severity': severity,
                'recommendations': recommendations
            }
            
            print("Sending response:", response)
            return jsonify(response)
            
        except Exception as e:
            print("Error during assessment:", str(e))
            return jsonify({
                'error': str(e),
                'request_data': request.get_data().decode('utf-8') if request.get_data() else None
            }), 400
            
    except Exception as e:
        print("Error in assess endpoint:", str(e))
        return jsonify({
            'error': str(e),
            'request_data': request.get_data().decode('utf-8') if request.get_data() else None
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)