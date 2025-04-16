# depression_analyzer

# Mental Health Assessment Tool

A machine learning-powered web application that assesses mental health using the PHQ-9 (Patient Health Questionnaire-9) scoring system. This tool leverages Natural Language Processing and Machine Learning to provide automated mental health assessments and recommendations.

## Features

- **Automated PHQ-9 Scoring**: Uses MentalBERT (a specialized BERT model for mental health) to analyze text input and predict PHQ-9 scores
- **Severity Classification**: Automatically categorizes mental health status into five levels:
  - Minimal
  - Mild
  - Moderate
  - Moderately Severe
  - Severe
- **Personalized Recommendations**: Provides tailored recommendations based on the severity level
- **RESTful API**: Built with Flask, offering easy integration with frontend applications
- **Cross-Origin Support**: Configured with CORS for seamless frontend integration

## Technical Stack

- **Backend**: Python, Flask
- **Machine Learning**: 
  - MentalBERT (Hugging Face Transformers)
  - Random Forest Regressor
  - Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Frontend**: HTML, CSS

## Use Cases

- Mental health screening in clinical settings
- Remote mental health assessment
- Research and data analysis
- Integration with healthcare systems

## Important Notes

- **Dataset Limitations**: The model is trained on a small dataset (~250 records), which may affect its accuracy and reliability
- **Potential Hallucinations**: Due to the limited training data, the model might produce inconsistent or unexpected results
- **Not Stress Tested**: The system hasn't undergone extensive stress testing in production environments
- **Input Requirements**: For accurate assessment, users must:
  - Answer all PHQ-9 questions completely
  - Provide the entire conversation transcript for analysis
  - Ensure responses are clear and detailed

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Flask server: `python main.py`
4. Access the web interface at `http://localhost:5001`

## Disclaimer

This tool is designed to assist healthcare professionals and should not be used as a substitute for professional medical advice, diagnosis, or treatment. The results should be interpreted with caution due to the limitations mentioned above.
