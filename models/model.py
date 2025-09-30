# model.py - Run this file ONCE to train and save the model artifacts

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import pickle

# 1. Simulate a Small Dataset (Replace with your actual labeled dataset)
data = {
    'text': [
        "Drinking hot water with lime kills all cancer cells, share immediately!",
        "The new government guidelines for flu shots are available on the official health website.",
        "COVID-19 vaccine magnetizes your arm, doctors warn against it.",
        "A healthy diet including fruits and vegetables is crucial for a strong immune system.",
        "Garlic and ginger cure every disease, throw away your medicine.",
        "Always consult a physician before starting any new treatment or medication."
    ],
    # 1: Fake/Misinformation, 0: Real/Fact
    'label': [1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

# . Preprocessing and Feature Extraction
X = df['text']
y = df['label']

# Create and fit TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

#  Training the Classifier
classifier = LogisticRegression(solver='liblinear')
classifier.fit(X_vectorized, y)

# 4. Savng the Model and Vectorizer
with open('classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model (classifier.pkl) and Vectorizer (vectorizer.pkl) saved successfully.")
print("You can now run the Streamlit app: streamlit run app.py")