from streamlit_extras.let_it_rain import rain
import streamlit as st
import pandas as pd
import requests  # <-- ADD THIS
from streamlit_lottie import st_lottie
from io import StringIO
import xgboost as xgb
#import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import numpy as np
from fpdf import FPDF
import datetime
import animations  # <-- ADD THIS (imports your new file)
import time

def load_lottieurl(url: str):
    """Fetches a Lottie JSON from a URL with error handling."""
    try:
        r = requests.get(url)
        r.raise_for_status() # Raises an error for bad responses (4xx or 5xx)
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading Lottie animation: {e}")
        return None
    except requests.exceptions.JSONDecodeError:
        st.error("Error: Failed to decode Lottie JSON. Is this a valid URL?")
        return None



def get_quality_report(df):
    st.header("1. Quality Report (Toxicity)")

    # Calculate missing data
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    missing_report = pd.DataFrame({
        'Missing Values': missing_data,
        'Percentage (%)': missing_percentage
    })

    st.subheader("Missing Data")
    st.dataframe(missing_report[missing_report['Missing Values'] > 0])
    return missing_report[missing_report['Missing Values'] > 0]

from sklearn.ensemble import IsolationForest

def get_outlier_report(df):
    st.subheader("Outlier Detection (Isolation Forest)")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # This will hold the indices of outlier rows
    outlier_indices = [] 
    
    if len(numerical_cols) > 0:
        # Create a simple report
        clf = IsolationForest(contamination=0.1, random_state=42)
        
        # We need to fillna just for the fit_predict to work
        numerical_data = df[numerical_cols].fillna(0) 
        outliers = clf.fit_predict(numerical_data)
        
        # Get the actual index numbers of the rows that are outliers
        outlier_indices = df.index[outliers == -1].tolist()
        
        outlier_percentage = len(outlier_indices) / len(df) * 100
        st.metric("Estimated Outlier Percentage", f"{outlier_percentage:.2f}%")
        
        if outlier_indices:
            st.warning(f"Found {len(outlier_indices)} outlier rows: {outlier_indices}")
    else:
        st.info("No numerical columns found for outlier detection.")
    
    # Return the list of outlier indices for the cleaner
    return outlier_indices

from presidio_analyzer import AnalyzerEngine

def get_risk_report(df):
    st.header("2. Risk Report (Hazard)")
    
    analyzer = AnalyzerEngine()
    pii_report = {}
    
    # We only scan a sample (e.g., first 50 rows) for performance
    sample_df = df.head(50)
    
    for col in sample_df.columns:
        if sample_df[col].dtype == 'object': # Only scan text columns
            all_text = " ".join(sample_df[col].dropna().astype(str))
            
            # Analyze the text
            analyzer_results = analyzer.analyze(text=all_text, language='en')
            
            found_entities = list(set([res.entity_type for res in analyzer_results]))
            
            if found_entities:
                pii_report[col] = found_entities

    if pii_report:
        st.warning("Found Potential PII (Personally Identifiable Information):")
        st.json(pii_report)
        return pii_report.keys() # Return list of risky columns
    else:
        st.success("No PII detected in the sampled data.")
        return []
    


# We need to tell Streamlit to allow Matplotlib to be used
#st.set_option('deprecation.showPyplotGlobalUse', False)

def get_signal_report(df, target_col):
    st.header("3. Signal Report (Value)")
    
    # --- 1. Preprocessing (Now much smarter!) ---
    proc_df = df.copy()

    # --- Step 1A: Handle NaNs in the TARGET column ---
    # LabelEncoder will crash if the target column has NaNs.
    # We'll fill any missing target values with the 'most_frequent' value.
    if proc_df[target_col].isnull().any():
        st.write(f"Note: Filling missing values in target '{target_col}' with most frequent value.")
        fill_val = proc_df[target_col].mode()[0]
        proc_df[target_col] = proc_df[target_col].fillna(fill_val)

    # Now, encode the target
    le = LabelEncoder()
    proc_df[target_col] = le.fit_transform(proc_df[target_col])
    
    
    # --- Step 1B: Handle High Cardinality in FEATURE columns ---
    st.write("Checking for high-cardinality text columns...")
    object_cols = proc_df.select_dtypes(include='object').columns
    CARDINALITY_THRESHOLD = 50 
    cols_to_drop = []
    
    for col in object_cols:
        if col == target_col: # Skip the target column
            continue
            
        num_unique = proc_df[col].nunique()
        if num_unique > CARDINALITY_THRESHOLD:
            cols_to_drop.append(col)
            
    if cols_to_drop:
        st.warning(f"Dropping high-cardinality columns (>{CARDINALITY_THRESHOLD} unique values) from model: {cols_to_drop}")
        proc_df = proc_df.drop(columns=cols_to_drop)
    else:
        st.success("No high-cardinality columns found. Good!")
        
    
    # --- Step 1C: One-Hot Encode the remaining categoricals ---
    st.write("One-hot encoding remaining categorical features...")
    proc_df = pd.get_dummies(proc_df, drop_first=True)
    
    # Handle any remaining missing values (e.g., in numerical columns)
    proc_df = proc_df.fillna(-999) 
    
    # --- 2. Train Model ---
    st.write("Training model...")
    X = proc_df.drop(target_col, axis=1)
    y = proc_df[target_col]
    
    # Make sure we still have features left after dropping
    if X.shape[1] == 0:
        st.error("No valid features left to train on after preprocessing. Cannot create Signal Report.")
        return []

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    
    # --- 3. Get Feature Importance ---
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    st.subheader("Top 10 Most Important Features")
    st.dataframe(importance_df.head(10))
    
    # --- 4. Get Feature Importance Plot ---
    st.subheader("Feature Importance Plot")
    st.info("This chart shows the relative 'importance' of each feature in predicting the target. Higher bars mean the model relies on that feature more.")

    chart_data = importance_df.head(10).set_index('Feature')
    
    st.bar_chart(chart_data)
    
    return importance_df['Feature'].head(10).tolist()

def get_final_recommendations(risky_cols, top_features):
    st.header("4. Final Audit Recommendations")
    
    # Find the intersection of risky and important columns
    # We must be careful: one-hot encoding (e.g., 'country_USA')
    # will make the feature name different from the PII column ('country').
    
    critical_features = []
    for risky_col in risky_cols:
        for feature in top_features:
            if feature.startswith(risky_col):
                critical_features.append(f"'{feature}' (derived from PII column '{risky_col}')")

    if critical_features:
        st.error("CRITICAL AUDIT WARNING")
        st.markdown("Your model is **highly dependent** on features that contain **sensitive PII**:")
        for feature in critical_features:
            st.markdown(f"- **{feature}**")
        st.markdown("This creates a major **privacy risk** and may lead to biased outcomes. **Recommendation:** Anonymize, hash, or remove these features before training.")
    else:
        st.success("Good news! Your model's top features do not appear to contain sensitive PII.")
    
    return critical_features

def clean_data(df, risky_cols, outlier_indices, 
               do_pii_removal, do_outlier_removal, do_missing_fix, 
               cleaning_strategies):
    """
    Cleans the dataframe based on user-selected operations.
    """
    st.header("5. Cleaning Data...")
    df_cleaned = df.copy()
    
    # 1. Drop PII columns (if user selected it)
    if do_pii_removal:
        st.write(f"Dropping PII columns: {risky_cols}")
        df_cleaned = df_cleaned.drop(columns=risky_cols, errors='ignore')
    else:
        st.write("Skipping PII column removal.")
        
    # 2. Drop outlier rows (if user selected it)
    if do_outlier_removal:
        st.write(f"Dropping {len(outlier_indices)} outlier rows.")
        df_cleaned = df_cleaned.drop(index=outlier_indices, errors='ignore')
    else:
        st.write("Skipping outlier row removal.")
        
    # 3. Apply missing data strategies (if user selected it)
    if do_missing_fix:
        st.write("Applying strategies for missing values...")
        cols_to_drop_rows = []
        
        for col, strategy_info in cleaning_strategies.items():
            strategy = strategy_info["strategy"]
            
            # Check if the column still exists (it might have been dropped as PII)
            if col not in df_cleaned.columns:
                st.warning(f"Skipping missing data strategy for '{col}' (column was already removed).")
                continue
                
            if strategy == "Drop Rows with Missing Data":
                cols_to_drop_rows.append(col)
                
            elif strategy == "Fill with Custom Value":
                custom_val = strategy_info["value"]
                try:
                    if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                        custom_val = pd.to_numeric(custom_val)
                    st.write(f"- Filling '{col}' with custom value: '{custom_val}'")
                    df_cleaned[col] = df_cleaned[col].fillna(custom_val)
                except ValueError:
                    st.error(f"Could not convert custom value '{custom_val}' to a number for column '{col}'. Filling with 0 instead.")
                    df_cleaned[col] = df_cleaned[col].fillna(0)
                    
            elif strategy == "Fill with Median":
                median_val = df_cleaned[col].median()
                st.write(f"- Filling '{col}' with median: {median_val}")
                df_cleaned[col] = df_cleaned[col].fillna(median_val)
                
            elif strategy == "Fill with Mean":
                mean_val = df_cleaned[col].mean()
                st.write(f"- Filling '{col}' with mean: {mean_val:.2f}")
                df_cleaned[col] = df_cleaned[col].fillna(mean_val)
                
            elif strategy == "Fill with Most Frequent":
                mode_val = df_cleaned[col].mode()[0]
                st.write(f"- Filling '{col}' with most frequent: '{mode_val}'")
                df_cleaned[col] = df_cleaned[col].fillna(mode_val)
                
        # Handle row drops at the end
        if cols_to_drop_rows:
            st.write(f"Dropping rows with missing data in columns: {cols_to_drop_rows}")
            df_cleaned = df_cleaned.dropna(subset=cols_to_drop_rows)
    else:
        st.write("Skipping missing data fixing.")
            
    st.success("Cleaning complete! Your file is ready for download.")
    st.dataframe(df_cleaned.head())
    return df_cleaned

@st.cache_data
def convert_df_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent re-running on every page load
    return df.to_csv(index=False).encode('utf-8')
st.set_page_config(layout="wide")
#st.title("Data Health & Risk Auditor 📈")

def get_missing_columns(df):
    """Finds columns with missing data and their types."""
    missing_cols = df.columns[df.isnull().any()].tolist()
    missing_info = {}
    for col in missing_cols:
        col_type = 'numerical' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical'
        missing_info[col] = col_type
    return missing_info


class PDF(FPDF):
    """Custom PDF class to add header and footer."""
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Data Health & Risk Auditor Report', 0, 1, 'C')
        self.set_font('Arial', '', 8)
        self.cell(0, 5, f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

def generate_simple_report(missing_df, outliers_list, risky_list, top_features_list, critical_list):
    """Generates a simple, non-technical summary of the audit."""
    
    report_text = ""
    
    # 1. Quality Summary
    report_text += "1. Data Quality (Toxicity)\n"
    report_text += "--------------------------------\n"
    if not missing_df.empty:
        report_text += f"Found {len(missing_df)} columns with missing data.\n"
        for col in missing_df.index:
            report_text += f"  - '{col}' has {missing_df.loc[col, 'Missing Values']} missing values.\n"
    else:
        report_text += "Good news! No missing data found.\n"
    
    if outliers_list:
        report_text += f"\nFound {len(outliers_list)} rows that appear to be outliers (statistically strange data).\n"
    else:
        report_text += "Good news! No major outliers were detected.\n"
    report_text += "\n\n"

    # 2. Risk Summary
    report_text += "2. Data Risk (Hazard)\n"
    report_text += "--------------------------------\n"
    if risky_list:
        report_text += f"Warning! Found {len(risky_list)} columns that may contain sensitive Personal Information (PII):\n"
        for col in risky_list:
            report_text += f"  - '{col}'\n"
        report_text += "This data should be handled with care.\n"
    else:
        report_text += "Good news! No sensitive PII was detected in the data sample.\n"
    report_text += "\n\n"

    # 3. Signal Summary
    report_text += "3. Data Value (Signal)\n"
    report_text += "--------------------------------\n"
    if top_features_list:
        report_text += "When trying to predict your target, the model found these features to be the most important:\n"
        for i, feature in enumerate(top_features_list[:5], 1): # Top 5
            report_text += f"  {i}. {feature}\n"
    else:
        report_text += "Could not determine the most important features.\n"
    report_text += "\n\n"
    
    # 4. Final Recommendation
    report_text += "4. Final Recommendation\n"
    report_text += "--------------------------------\n"
    if critical_list:
        report_text += "CRITICAL WARNING: Your model's most important features are directly linked to sensitive PII.\n"
        report_text += "This means the model is likely biased and creates a major security risk. It should not be used.\n"
        report_text += f"Problem features: {', '.join(critical_list)}\n"
    elif risky_list:
        report_text += "Your model's top features are clean, but the dataset still contains sensitive PII. Be sure to remove it before sharing.\n"
    else:
        report_text += "Excellent! Your data appears to be clean, non-hazardous, and the model is using safe features.\n"

    return report_text

def create_pdf(report_string):
    """Creates a PDF file from the report string."""
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_body(report_string)
    # NEW, FIXED LINE
    # NEW, FIXED LINE
    return bytes(pdf.output(dest='S'))
# --- 3. YOUR STREAMLIT APP LOGIC LAST ---

# Initialize session state to hold our cleaned data
# --- 3. YOUR STREAMLIT APP LOGIC LAST ---

# Initialize session state to hold our cleaned data
# --- 3. YOUR STREAMLIT APP LOGIC LAST ---

# Initialize session state
# --- 3. YOUR STREAMLIT APP LOGIC LAST ---

# Initialize session state for all our results
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
    st.session_state.audit_run = False
    st.session_state.risky_columns = []
    st.session_state.outlier_indices = []
    st.session_state.top_features = []
    st.session_state.critical_features = []
    st.session_state.missing_report = pd.DataFrame()

st.set_page_config(layout="wide")
# --- NEW TITLE & ANIMATION ---
# 1. Load the Lottie animation
# NEW, RELIABLE URL:
# --- NEW TITLE & ANIMATION ---
# 1. Load the Lottie animation
lottie_json = animations.load_lottieurl(animations.LOTTIE_URL_HEADER)

# 2. Create a two-column layout
col1, col2 = st.columns([1, 4])
# ... (rest of the code)

with col1:
    if lottie_json:
        st_lottie(
            lottie_json,
            speed=1,
            reverse=False,
            loop=True,
            quality="low", # Use "low" for better performance
            height=150,
            width=150,
            key="data_animation"
        )
    else:
        st.write("Loading animation...")

with col2:
    st.title("Data Health & Risk Auditor")
    st.markdown("*Your all-in-one tool for cleaning 'data manure'.*")
# --- END NEW TITLE ---
uploaded_file = st.file_uploader("Choose a CSV file")

if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.header("Data Preview")
    st.dataframe(dataframe.head())

    all_columns = dataframe.columns.tolist()
    target_column = st.selectbox("Select your Target Variable (for signal analysis)", all_columns)

    # --- Cleaning Operations Toggles ---
    st.header("Cleaning Operations")
    st.info("Choose which cleaning steps to perform.")
    
    do_pii_removal = st.checkbox("Remove PII Columns (e.g., Cast, Director)", value=True)
    do_outlier_removal = st.checkbox("Remove Outlier Rows", value=True)
    do_missing_fix = st.checkbox("Fix Missing Data", value=True)
    
    cleaning_strategies = {} # Initialize an empty dict
    missing_info = {} # Initialize
    
    if do_missing_fix:
        missing_info = get_missing_columns(dataframe)
        if missing_info:
            with st.expander("Advanced Missing Data Options", expanded=True):
                # ... (all your existing code for the missing data expander) ...
                st.info("Set your strategy for handling missing data in each column.")
                
                for col, col_type in missing_info.items():
                    if col_type == 'numerical':
                        options = ["Fill with Median", "Fill with Mean", "Fill with Custom Value", "Drop Rows with Missing Data"]
                        strategy = st.selectbox(f"Strategy for '{col}' (Numerical):", options, key=f"{col}_strategy")
                        
                        if strategy == "Fill with Custom Value":
                            custom_val = st.text_input(f"Custom value for '{col}':", key=f"{col}_custom", value="0")
                            cleaning_strategies[col] = {"strategy": strategy, "value": custom_val}
                        else:
                            cleaning_strategies[col] = {"strategy": strategy}
                            
                    else: # Categorical
                        options = ["Fill with Most Frequent", "Fill with Custom Value", "Drop Rows with Missing Data"]
                        strategy = st.selectbox(f"Strategy for '{col}' (Categorical):", options, key=f"{col}_strategy")
                        
                        if strategy == "Fill with Custom Value":
                            custom_val = st.text_input(f"Custom value for '{col}':", key=f"{col}_custom", value="Missing")
                            cleaning_strategies[col] = {"strategy": strategy, "value": custom_val}
                        else:
                            cleaning_strategies[col] = {"strategy": strategy}
        else:
            st.info("No missing data found to clean.")

    # --- Main Button ---
   # --- Main Button ---
    if st.button("Run Data Audit & Clean"):
        
        with st.status("Running Data Audit & Cleaning...", expanded=True) as status:
            
            # --- 1. Create a placeholder for our animations ---
            animation_placeholder = st.empty()
            
            # --- 2. Show the PROCESSING animation ---
            lottie_processing = animations.load_lottielocal(animations.PROCESSING_ANIMATION_PATH)
            if lottie_processing:
                with animation_placeholder.container():
                    st_lottie(lottie_processing, speed=1, loop=True, quality="low", height=150, width=150)
            
            # --- 3. Run all your existing steps (unchanged) ---
            st.write("Step 1/5: Analyzing data quality (toxicity)...")
            st.session_state.missing_report = get_quality_report(dataframe)
            
            st.write("Step 2/5: Detecting outliers...")
            st.session_state.outlier_indices = get_outlier_report(dataframe)
            
            st.write("Step 3/5: Scanning for PII (hazard)...")
            st.session_state.risky_columns = get_risk_report(dataframe)
            
            st.write("Step 4/5: Analyzing feature signal (value)...")
            st.session_state.top_features = get_signal_report(dataframe, target_column)
            
            st.write("Step 5/5: Generating final recommendations...")
            st.session_state.critical_features = get_final_recommendations(st.session_state.risky_columns, st.session_state.top_features)
            
            st.write("Applying cleaning operations...")
            cleaned_df = clean_data(dataframe, st.session_state.risky_columns, st.session_state.outlier_indices, 
                                    do_pii_removal, do_outlier_removal, do_missing_fix, 
                                    cleaning_strategies)
            
            st.session_state.cleaned_data = cleaned_df
            st.session_state.audit_run = True
            
            # --- 4. Show the "BUGS CLEARED" animation ---
            lottie_complete = animations.load_lottielocal(animations.COMPLETE_ANIMATION_PATH)
            if lottie_complete:
                with animation_placeholder.container():
                    st_lottie(lottie_complete, speed=1, loop=False, quality="low", height=150, width=150)
            
            # Pause for a moment to see the "complete" animation
            time.sleep(2) 
            
            # --- 5. Update the status box to "complete" ---
            status.update(label="Audit Complete!", state="complete")
            rain(
            emoji="👾",  # You can use any emoji here!
            font_size=54,
            falling_speed=2,
            animation_length="10s",
        )
# --- Show download buttons ONLY if the audit has been run ---
if st.session_state.audit_run:
    st.header("Download Results")
    
    # --- 1. CSV Download Button ---
    if st.session_state.cleaned_data is not None:
        st.info("This file has been cleaned based on your selected operations.")
        csv_data = convert_df_to_csv(st.session_state.cleaned_data)
        st.download_button(
            label="Download Cleaned_Data.csv",
            data=csv_data,
            file_name="cleaned_data.csv",
            mime="text/csv",
            key="csv_download"
        )
    
    # --- 2. PDF Download Button ---
    st.info("A non-technical summary of all the findings.")
    
    # Generate the report text
    report_string = generate_simple_report(
        st.session_state.missing_report,
        st.session_state.outlier_indices,
        st.session_state.risky_columns,
        st.session_state.top_features,
        st.session_state.critical_features
    )
    
    # Create the PDF
    pdf_data = create_pdf(report_string)
    
    st.download_button(
        label="Download Report.pdf",
        data=pdf_data,
        file_name="Data_Audit_Report.pdf",
        mime="application/pdf",
        key="pdf_download"
    )