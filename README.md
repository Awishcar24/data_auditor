

# Data Health & Risk Auditor

[](https://www.google.com/search?q=https://YOUR-STREAMLIT-APP-URL.streamlit.app/)

An interactive web app built in Streamlit that audits datasets for quality, risk, and value. This tool solves the "data manure" problem by identifying toxic (poor quality), hazardous (PII), and noisy (irrelevant) data, allowing a user to clean it with one click.

The app features a full audit suite, interactive cleaning controls, and one-click reporting, all enhanced with Lottie animations for a professional user experience.

-----

## The Problem: "Data Manure"

In the age of AI, data is often called the "new oil," but in reality, it's become the "new manure." It's a byproduct of every digital action, and it piles up faster than organizations can manage. 
This raw, unfiltered data is:

  * **Toxic:** It's full of errors, missing values, and outliers that poison ML models.
  * **Hazardous:** It contains sensitive PII (Personally Identifiable Information), creating major legal and ethical risks.
  * **Noisy:** The valuable "signal" is often drowned out by a sea of irrelevant "noise," making it hard to find useful insights.

This tool is a "data auditor" that automatically inspects a dataset and provides a simple, non-technical report on its health.

-----

## Features

  * **Toxicity Report (Quality):**
      * Finds and reports all missing values.
      * Uses **Scikit-learn's `IsolationForest`** to detect statistical outliers.
  * **Hazard Report (Risk):**
      * Uses **Microsoft's `Presidio` (NLP)** to automatically scan for and flag 15+ types of PII (e.g., `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`).
  * **Signal Report (Value):**
      * Trains an **XGBoost** model on the fly to identify and rank the most predictive features in the dataset.
      * Intelligently drops "high-cardinality" text columns (like IDs or names) to prevent model-crashing.
  * **Final Recommendation:**
      * A unique audit that checks if the model's **most important features** are also **PII**, warning the user of a critical bias and security risk.
  * **Dynamic UI & Animations:**
      * Interactive header animation loaded via **Streamlit-Lottie**.
      * Technical "in-progress" and "complete" animations that play during the audit.
      * Celebration animation (using **Streamlit-Extras**) upon successful completion.
  * **Interactive Cleaning Pipeline:**
      * Users can toggle cleaning operations on/off (e.g., "Remove PII," "Drop Outliers").
      * Provides an "Advanced" menu to let the user select custom imputation strategies for each column (e.g., Fill with Median, Mean, or a Custom Value).
  * **One-Click Reporting:**
      * Download the fully cleaned, model-ready dataset as a `.csv` file.
      * Download a non-technical, user-friendly summary of all findings as a `.pdf` file (generated with **FPDF2**).

-----

## Tech Stack

  * **App Framework:** [Streamlit](https://streamlit.io/)
  * **Data Manipulation:** [Pandas](https://pandas.pydata.org/)
  * **Machine Learning (Quality & Signal):** [Scikit-learn](https://scikit-learn.org/) & [XGBoost](https://xgboost.ai/)
  * **PII Detection (NLP):** [Microsoft Presidio](https://microsoft.github.io/presidio/)
  * **PDF Generation:** [FPDF2](https://py-pdf.github.io/fpdf2/)
  * **Animations:** [Streamlit-Lottie](https://github.com/andfanilo/streamlit-lottie) & [Streamlit-Extras](https://github.com/arnaudmiribel/streamlit-extras)
  * **Web Requests:** [Requests](https://requests.readthedocs.io/en/latest/) (for loading header animation)
  * **Core Language:** [Python](https://www.python.org/)

-----

## How to Run Locally

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
    cd YOUR-REPO-NAME
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required libraries:**
    (Make sure you have run `pip freeze > requirements.txt` first to capture all libraries).

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the spaCy language model** (for Presidio):

    ```bash
    python -m spacy download en_core_web_lg
    ```

5.  **Run the Streamlit app:**

    ```bash
    streamlit run app.py
    ```
