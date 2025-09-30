# WhatsApp Health Checker  
# Explainable WhatsApp Health News Detector

This project uses the **LIME (Local Interpretable Model-agnostic Explanations)** technique alongside a basic machine learning classifier to verify the veracity of health-related news, simulating common WhatsApp misinformation. The application is built with **Streamlit** for a clean, interactive user interface.

##  Features

* **Prediction:** Classifies text as 'REAL' or 'FAKE'.
* **Explainable AI (XAI):** Uses LIME to highlight which words in the text most strongly influenced the model's prediction (words for 'FAKE' in red, words for 'REAL' in green).
* **Simple UI:** Streamlit provides an easy-to-use web interface.

## ⚙️ How to Run Locally

### Prerequisites

1.  Python 3.7+ installed.
2.  Git installed.

### Step-by-Step Guide

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YOUR_GITHUB_USERNAME/whatsapp-health-news-detector.git](https://github.com/YOUR_GITHUB_USERNAME/whatsapp-health-news-detector.git)
    cd whatsapp-health-news-detector
    ```

2.  **Create and Activate Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train and Save the Model:**
    You must run the `model.py` script once to generate the `classifier.pkl` and `vectorizer.pkl` files.
    ```bash
    python model.py
    ```

5.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    The application will open automatically in your web browser (usually at `http://localhost:8501`).

## 💻 Pushing to GitHub

After setting up and testing your code locally:

1.  **Initialize Git (if you haven't already):**
    ```bash
    git init
    ```

2.  **Create a `.gitignore` file** (to avoid committing large model files or virtual environments).
    *Create a file named `.gitignore` and add the following lines:*
    ```
    # Virtual environment
    venv/
    # Model artifacts (optional, but good practice if they are too large)
    *.pkl
    # Cache and Streamlit files
    .streamlit/
    ```

3.  **Commit the Code:**
    ```bash
    git add .
    git commit -m "Initial commit: Added Streamlit app, LIME, and model setup"
    ```

4.  **Create a Repository on GitHub:**
    Go to your GitHub profile and create a *New Repository* (e.g., `whatsapp-health-news-detector`). **Do not** initialize it with a README, license, or `.gitignore` as you already have these.

5.  **Link and Push:**
    Replace `YOUR_GITHUB_USERNAME` with your actual username.
    ```bash
    git remote add origin [https://github.com/YOUR_GITHUB_USERNAME/whatsapp-health-news-detector.git](https://github.com/YOUR_GITHUB_USERNAME/whatsapp-health-news-detector.git)
    git branch -M main
    git push -u origin main
    ```

