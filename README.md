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

## Sample Transcripts

### Transcript 1: Severe Depression
```
"I constantly feel like I am worthless and a burden to others. I constantly feel like I am worthless and a burden to others. I frequently think about ending everything, I don't see a way out. I constantly feel like I am worthless and a burden to others. I am always exhausted, even talking feels like too much effort. People tell me I seem lifeless or that I barely move anymore. I feel empty and disconnected from everything, like life is meaningless. I frequently think about ending everything, I don't see a way out. I barely sleep at night, and when I do, nightmares wake me up."
```

### Transcript 2: Moderate Depression
```
"I feel like I am not good enough, and it bothers me a lot. I don't enjoy activities like I used to, I force myself to do things. I feel tired even after a full night's sleep. I feel like I am not good enough, and it bothers me a lot. I sometimes think about disappearing, but I wouldn't act on it. Feeling down most days, like nothing makes me happy anymore. My eating habits have changed significantly, sometimes I skip meals. I struggle to concentrate on my assignments and lose track easily. Feeling down most days, like nothing makes me happy anymore."
```

### Transcript 3: Mild Depression
```
"I have slight trouble falling asleep but not every night. I get distracted easily but can regain focus quickly. I occasionally feel uninterested in activities I usually enjoy. I have slight trouble falling asleep but not every night. Sometimes I feel down, but it doesn't last long. I get distracted easily but can regain focus quickly. I get distracted easily but can regain focus quickly. Sometimes I feel down, but it doesn't last long. I get distracted easily but can regain focus quickly."
```

### Transcript 4: Minimal/No Depression
```
"I can focus well on my studies without distractions. I feel confident in myself and my abilities. No thoughts of self-harm or suicidal ideation. I feel confident in myself and my abilities. Sleeping well, no major issues with rest. Eating habits are normal, no major appetite changes. Energy levels are good, I can do my daily tasks easily. I feel fine and enjoy my hobbies as usual. Energy levels are good, I can do my daily tasks easily."
```

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Flask server: `python main.py`
4. Access the web interface at `http://localhost:5001`

## Disclaimer

This tool is designed to assist healthcare professionals and should not be used as a substitute for professional medical advice, diagnosis, or treatment. The results should be interpreted with caution due to the limitations mentioned above.
