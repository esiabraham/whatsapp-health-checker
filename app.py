# app.py - Main Streamlit Application (Corrected for LinearSVC)

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy.special import expit # For sigmoid function

# LIME imports
from lime.lime_text import LimeTextExplainer

# --- Configuration ---
MODEL_PATH = 'classifier.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
CLASS_NAMES = ['REAL (0)', 'FAKE (1)'] # Ensure this aligns with your model labels

# --- Load Artifacts ---
@st.cache_resource
def load_model_artifacts():
    """Loads the trained model and vectorizer."""
    try:
        with open(MODEL_PATH, 'rb') as f:
            classifier = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        return classifier, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please run 'python model.py' first.")
        return None, None

classifier, vectorizer = load_model_artifacts()

# --- LIME Explainer Setup ---
def predictor(texts):
    """
    LIME requires a function that takes a list of strings and returns a prediction array.
    For LinearSVC, we use decision_function and convert it to pseudo-probabilities
    using the sigmoid (expit) function.
    """
    vectorized_texts = vectorizer.transform(texts)
    
    # Get the decision function scores from LinearSVC
    # For binary classification, decision_function returns a 1D array of scores.
    decision_scores = classifier.decision_function(vectorized_texts)
    
    # Convert decision scores to pseudo-probabilities using the sigmoid function (expit)
    # P(class=1) = 1 / (1 + exp(-score))
    prob_fake = expit(decision_scores) # expit is scipy's optimized sigmoid
    prob_real = 1 - prob_fake
    
    # LIME expects a 2D array: [[prob_real_1, prob_fake_1], [prob_real_2, prob_fake_2], ...]
    return np.column_stack((prob_real, prob_fake))

# Initialize the LIME Explainer
if classifier is not None:
    explainer = LimeTextExplainer(
        class_names=CLASS_NAMES,
        # TF-IDF uses words/ngrams, so we use 'word' for the default split
        split_expression=' ',
        # Feature selection to speed up LIME for demonstration
        bow=True
    )

# --- Streamlit UI ---
st.title("🛡️ WhatsApp Health News Verifier")
st.markdown("""
Enter a health-related message you received on WhatsApp to check its veracity.
The tool uses an Explainable AI (XAI) model to not only predict the outcome but also
show **why** it made that decision (the LIME explanation).
""")

# Input Area
news_text = st.text_area("Paste the health news text here:", height=150)

if st.button("Verify News"):
    if classifier is None:
        st.stop()

    if news_text:
        # 1. Prediction
        st.subheader("Prediction Result")
        
        # Get pseudo-probabilities from our custom predictor function
        probabilities = predictor([news_text])[0] # This now returns [prob_real, prob_fake]
        
        # Determine the predicted class based on the probabilities
        prediction_class_index = np.argmax(probabilities)
        prediction_label = CLASS_NAMES[prediction_class_index]
        fake_prob = probabilities[1] # Probability of being FAKE (index 1)

        if prediction_label == 'FAKE (1)':
            st.error(f"🚨 **PREDICTED: FAKE NEWS** (Confidence: {fake_prob*100:.2f}%)")
        else:
            st.success(f"✅ **PREDICTED: REAL NEWS** (Confidence: {probabilities[0]*100:.2f}%)")

        # 2. LIME Explanation
        st.subheader("Model Explanation (LIME)")
        st.info("Words highlighted in **red** contributed to the **FAKE** prediction. Words in **green** contributed to the **REAL** prediction.")
        
        # Generate the explanation
        with st.spinner('Generating LIME explanation...'):
            # num_features controls how many features (words) to highlight
            explanation = explainer.explain_instance(
                news_text, 
                predictor, # Use our adjusted predictor function
                num_features=10,
                labels=(0, 1) # Explicitly tell LIME to explain both classes if needed, or just the predicted one
            )

            # Display the explanation in the Streamlit app
            html_explanation = explanation.as_html()
            st.components.v1.html(html_explanation, height=500, scrolling=True)

    else:
        st.warning("Please enter some text to verify.")