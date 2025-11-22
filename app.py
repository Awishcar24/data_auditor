import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import seaborn as sns
import matplotlib.pyplot as plt
import io

# --- UI & Animation Libraries ---
from streamlit_lottie import st_lottie
from streamlit_extras.let_it_rain import rain
import animations

# --- Machine Learning & Data Processing ---
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp

# --- NLP & Reporting ---
from presidio_analyzer import AnalyzerEngine
from fpdf import FPDF
import datetime

# ==========================================
# 0. PAGE CONFIG & CSS
# ==========================================
st.set_page_config(layout="wide", page_title="Data Refinery")

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #FFFFFF 0%, #F0F2F6 100%) !important; }
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Roboto Mono', monospace !important; color: #31333F !important; }
    div[data-testid="stMetric"], div[data-testid="stExpander"], div[data-testid="stStatusWidget"] {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
    }
    div[data-testid="stMetricValue"] { color: #6a11cb !important; }
    div.stButton > button, div.stDownloadButton > button {
        background-image: linear-gradient(45deg, #6a11cb 0%, #2575fc 100%) !important;
        color: white !important;
        border-radius: 50px !important;
        border: none !important;
        padding: 12px 20px !important;
        font-weight: bold !important;
        width: 100% !important;
        box-shadow: 0 4px 10px rgba(106, 17, 203, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 20px rgba(37, 117, 252, 0.5) !important;
    }
    div[data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 1px dashed #6a11cb !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def add_download_button(fig, key_name):
    """Helper to add a download button for any matplotlib figure"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="💾 Download Plot",
        data=buf,
        file_name=f"{key_name}.png",
        mime="image/png",
        key=key_name
    )

def get_quality_report(df):
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    missing_report = pd.DataFrame({
        'Missing Values': missing_data,
        'Percentage (%)': missing_percentage
    })
    return missing_report[missing_report['Missing Values'] > 0]

def get_outlier_report(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    outlier_indices = [] 
    if len(numerical_cols) > 0:
        numerical_data = df[numerical_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numerical_data)
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_preds = iso_forest.fit_predict(X_scaled) 
        
        autoencoder = MLPRegressor(hidden_layer_sizes=(10, 5, 10), random_state=42, max_iter=500)
        autoencoder.fit(X_scaled, X_scaled)
        X_pred = autoencoder.predict(X_scaled)
        mse = np.mean(np.power(X_scaled - X_pred, 2), axis=1)
        error_threshold = np.percentile(mse, 90) 
        ae_preds = np.where(mse > error_threshold, -1, 1)
        
        final_outliers = (iso_preds == -1) | (ae_preds == -1)
        outlier_indices = df.index[final_outliers].tolist()
    return outlier_indices

def get_risk_report(df):
    analyzer = AnalyzerEngine()
    pii_report = {}
    sample_df = df.head(50)
    for col in sample_df.columns:
        if sample_df[col].dtype == 'object':
            all_text = " ".join(sample_df[col].dropna().astype(str))
            results = analyzer.analyze(text=all_text, language='en')
            found = list(set([res.entity_type for res in results]))
            if found:
                pii_report[col] = found
    return pii_report.keys()

def get_signal_report(df, target_col):
    proc_df = df.copy()
    if proc_df[target_col].isnull().any():
        proc_df[target_col] = proc_df[target_col].fillna(proc_df[target_col].mode()[0])
    le = LabelEncoder()
    proc_df[target_col] = le.fit_transform(proc_df[target_col])
    
    object_cols = proc_df.select_dtypes(include='object').columns
    cols_to_drop = [col for col in object_cols if col != target_col and proc_df[col].nunique() > 50]
    if cols_to_drop:
        proc_df = proc_df.drop(columns=cols_to_drop)
    
    proc_df = pd.get_dummies(proc_df, drop_first=True)
    proc_df = proc_df.fillna(-999)
    
    X = proc_df.drop(target_col, axis=1)
    y = proc_df[target_col]
    
    if X.shape[1] == 0:
        return [], pd.DataFrame()

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Plot
    st.subheader("Feature Importance Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    top_10 = importance_df.head(10).iloc[::-1] 
    ax.barh(top_10['Feature'], top_10['Importance'], color='#4A90E2')
    ax.set_title("Top 10 Predictive Features")
    ax.set_xlabel("Importance Score")
    st.pyplot(fig)
    add_download_button(fig, "signal_audit_chart") # Add download button
    
    return importance_df['Feature'].head(10).tolist(), importance_df

def get_drift_report(current_df, reference_df):
    current_cols = set(current_df.select_dtypes(include=[np.number]).columns)
    ref_cols = set(reference_df.select_dtypes(include=[np.number]).columns)
    common_cols = list(current_cols.intersection(ref_cols))
    drift_data = []
    for col in common_cols:
        stat, p_value = ks_2samp(current_df[col].dropna(), reference_df[col].dropna())
        has_drift = p_value < 0.05
        drift_data.append({
            "Feature": col,
            "P-Value": f"{p_value:.4f}",
            "Status": "🔴 Drift" if has_drift else "🟢 Stable"
        })
    return drift_data

def get_final_recommendations(risky_cols, top_features):
    critical_features = []
    for risky_col in risky_cols:
        for feature in top_features:
            if feature.startswith(risky_col):
                critical_features.append(f"'{feature}' (from '{risky_col}')")
    return critical_features

# ==========================================
# 2. GUIDE & VISUALIZATION FUNCTIONS
# ==========================================

def get_deep_eda_report(df):
    st.subheader("Feature Correlations")
    st.info("Darker colors mean a stronger relationship between features.")
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)
        add_download_button(fig, "eda_heatmap") # Add download button
    else:
        st.warning("Not enough numerical features for heatmap.")

def get_model_scout(df, target_col):
    st.subheader("AutoML Scout Results")
    proc_df = df.copy().dropna()
    le = LabelEncoder()
    proc_df[target_col] = le.fit_transform(proc_df[target_col])
    proc_df = pd.get_dummies(proc_df, drop_first=True)
    X = proc_df.drop(target_col, axis=1)
    y = proc_df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=50),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }
    results = []
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.append({"Model": name, "Accuracy": accuracy_score(y_test, y_pred)})
        except:
            results.append({"Model": name, "Accuracy": 0})
            
    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    best_model = results_df.iloc[0]
    st.success(f"🏆 Best Model: **{best_model['Model']}** ({best_model['Accuracy']:.2%} Accuracy)")
    st.dataframe(results_df)
    return results_df

def get_visualization_studio(df):
    st.header("Visualization Studio 🎨")
    st.info("Create custom charts to explore your data.")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    chart_type = st.selectbox("Choose Chart Type:", ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Correlation Heatmap"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_plot = False
    
    if chart_type == "Bar Chart":
        x, y = st.selectbox("X Axis", cat_cols), st.selectbox("Y Axis", num_cols)
        if x and y: 
            sns.barplot(data=df, x=x, y=y, ax=ax, palette="viridis")
            plt.xticks(rotation=45)
            valid_plot = True
    elif chart_type == "Scatter Plot":
        x, y = st.selectbox("X Axis", num_cols, key="sx"), st.selectbox("Y Axis", num_cols, key="sy")
        if x and y: 
            sns.scatterplot(data=df, x=x, y=y, ax=ax)
            valid_plot = True
    elif chart_type == "Histogram":
        x = st.selectbox("Column", num_cols)
        if x: 
            sns.histplot(data=df, x=x, kde=True, ax=ax)
            valid_plot = True
    elif chart_type == "Correlation Heatmap":
        if len(num_cols) > 1: 
            sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
            valid_plot = True
    elif chart_type == "Line Chart":
        x, y = st.selectbox("X Axis", df.columns), st.selectbox("Y Axis", num_cols)
        if x and y: 
            sns.lineplot(data=df, x=x, y=y, ax=ax)
            valid_plot = True

    if valid_plot:
        st.pyplot(fig)
        add_download_button(fig, f"custom_{chart_type}") # Add download button

# ==========================================
# 3. CLEANING & UTILS
# ==========================================

def clean_data(df, risky_cols, outlier_indices, do_pii, do_out, do_miss, strategies):
    df_cleaned = df.copy()
    if do_pii: df_cleaned = df_cleaned.drop(columns=risky_cols, errors='ignore')
    if do_out: df_cleaned = df_cleaned.drop(index=outlier_indices, errors='ignore')
    if do_miss:
        for col, info in strategies.items():
            if col not in df_cleaned.columns: continue
            strat = info["strategy"]
            if strat == "Drop Rows": df_cleaned = df_cleaned.dropna(subset=[col])
            elif strat == "Median": df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            elif strat == "Mean": df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
            elif strat == "Most Frequent": df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
            elif strat == "Custom Value": 
                val = info["value"]
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    try: val = pd.to_numeric(val)
                    except: val = 0
                df_cleaned[col] = df_cleaned[col].fillna(val)
    return df_cleaned

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def get_missing_columns(df):
    cols = df.columns[df.isnull().any()].tolist()
    return {c: 'numerical' if pd.api.types.is_numeric_dtype(df[c]) else 'categorical' for c in cols}

def generate_cleaning_script(risky_cols, outlier_indices, strategies):
    script_content = f"""
import pandas as pd
import numpy as np

def clean_dataset(df):
    print("Starting Cleaning Pipeline...")
    # 1. Drop PII
    pii_cols = {list(risky_cols)}
    if pii_cols: df = df.drop(columns=pii_cols, errors='ignore')
    # 2. Drop Outliers
    outlier_indices = {outlier_indices}
    if outlier_indices: df = df.drop(index=outlier_indices, errors='ignore')
    # 3. Impute
    """
    if strategies:
        for col, info in strategies.items():
            strat = info['strategy']
            val = info.get('value', None)
            script_content += f"\n    # Strategy for '{col}': {strat}"
            if strat == "Median": script_content += f"\n    df['{col}'] = df['{col}'].fillna(df['{col}'].median())"
            elif strat == "Mean": script_content += f"\n    df['{col}'] = df['{col}'].fillna(df['{col}'].mean())"
            elif strat == "Most Frequent": script_content += f"\n    df['{col}'] = df['{col}'].fillna(df['{col}'].mode()[0])"
            elif strat == "Custom Value":
                if str(val).replace('.','',1).isdigit(): script_content += f"\n    df['{col}'] = df['{col}'].fillna({val})"
                else: script_content += f"\n    df['{col}'] = df['{col}'].fillna('{val}')"
            elif strat == "Drop Rows": script_content += f"\n    df = df.dropna(subset=['{col}'])"
    script_content += "\n    return df"
    return script_content

# ==========================================
# 4. REPORTING
# ==========================================
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Data Health Audit Report', 0, 1, 'C')
        self.ln(5)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf_report(missing_df, outlier_count, risky_cols, top_features, critical_features, drift_data):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    safe_width = 190
    pdf.cell(0, 10, "1. Toxicity Audit", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(safe_width, 5, f"Outliers: {outlier_count}")
    if not missing_df.empty: pdf.multi_cell(safe_width, 5, f"Missing Cols: {len(missing_df)}")
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "2. Hazard Audit (PII)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Arial", size=10)
    if risky_cols: pdf.multi_cell(safe_width, 5, f"PII Found: {', '.join(risky_cols)}")
    else: pdf.multi_cell(safe_width, 5, "No PII detected.")
    pdf.ln(5)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, "3. Signal Audit", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Arial", size=10)
    if top_features: pdf.multi_cell(safe_width, 5, f"Top Features: {', '.join(top_features[:5])}")
    pdf.ln(5)
    return bytes(pdf.output(dest='S'))

# ==========================================
# 5. MAIN APP LOGIC
# ==========================================

if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
    st.session_state.audit_run = False
    st.session_state.importance_df = pd.DataFrame()
    st.session_state.drift_data = []
    st.session_state.risky_cols = []
    st.session_state.outlier_indices = []
    st.session_state.missing_report = pd.DataFrame()
    st.session_state.top_features = []
    st.session_state.critical_features = []
    st.session_state.strategies = {}

lottie_header = animations.load_lottieurl(animations.LOTTIE_URL_HEADER)
col1, col2 = st.columns([1, 4])
with col1:
    if lottie_header: st_lottie(lottie_header, height=150, key="header")
with col2:
    st.title("The Data Refinery")
    st.markdown("*Automated Health & Risk Auditor*")

st.info("Step 1: Upload your data.")
uploaded_file = st.file_uploader("1. Current Dataset (Required)", type="csv")
uploaded_ref = st.file_uploader("2. Reference Dataset (Optional - for Drift)", type="csv")

if uploaded_file is not None:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
    ref_df = None
    if uploaded_ref is not None:
        uploaded_ref.seek(0)
        try: ref_df = pd.read_csv(uploaded_ref)
        except: pass

    with st.container():
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Rows", df.shape[0])
        k2.metric("Columns", df.shape[1])
        k3.metric("Num Features", len(df.select_dtypes(include=np.number).columns))
        k4.metric("Cat Features", len(df.select_dtypes(include='object').columns))
        with st.expander("View Raw Data"):
            st.dataframe(df.head(), use_container_width=True)
    st.markdown("---")
    target_col = st.selectbox("Target Variable (What to predict?)", df.columns)
    
    st.header("Cleaning Operations")
    c1, c2, c3 = st.columns(3)
    do_pii = c1.checkbox("Remove PII", value=True)
    do_out = c2.checkbox("Remove Outliers", value=True)
    do_miss = c3.checkbox("Fix Missing Data", value=True)
    
    strategies = {}
    if do_miss:
        miss_info = get_missing_columns(df)
        if miss_info:
            with st.expander("Advanced Missing Data Options"):
                for col, dtype in miss_info.items():
                    opt = st.selectbox(f"Fix '{col}' ({dtype})", ["Median" if dtype=='numerical' else "Most Frequent", "Drop Rows"])
                    strategies[col] = {"strategy": opt}

    if st.button("Run Audit & Clean"):
        with st.status("Refining Data...", expanded=True) as status:
            anim_ph = st.empty()
            l_proc = animations.load_lottielocal(animations.PROCESSING_ANIMATION_PATH)
            if l_proc:
                with anim_ph.container():
                    st_lottie(l_proc, height=150, key="proc")
            
            st.session_state.strategies = strategies
            st.write("Running Toxicity Audit...")
            st.session_state.missing_report = get_quality_report(df)
            st.session_state.outlier_indices = get_outlier_report(df)
            st.write("Running Hazard Audit...")
            st.session_state.risky_cols = get_risk_report(df)
            st.write("Running Signal Audit...")
            top_feat, imp_df = get_signal_report(df, target_col) # Plots & returns DF
            st.session_state.top_features = top_feat
            st.session_state.importance_df = imp_df
            st.session_state.critical_features = get_final_recommendations(st.session_state.risky_cols, top_feat)
            
            if ref_df is not None:
                st.write("Checking Data Drift...")
                st.session_state.drift_data = get_drift_report(df, ref_df)
            
            st.write("Running AutoML Scout...")
            get_model_scout(df, target_col)
            
            st.write("Cleaning Data...")
            st.session_state.cleaned_data = clean_data(df, st.session_state.risky_cols, st.session_state.outlier_indices, do_pii, do_out, do_miss, strategies)
            st.session_state.audit_run = True
            
            l_done = animations.load_lottielocal(animations.COMPLETE_ANIMATION_PATH)
            if l_done:
                with anim_ph.container():
                    st_lottie(l_done, height=150, loop=False, key="done")
            time.sleep(1)
            status.update(label="Audit Complete!", state="complete")
        
        rain(emoji="✨", font_size=54, falling_speed=5, animation_length="1s")

if st.session_state.audit_run:
    st.markdown("---")
    st.header("🔎 Audit Results & Studio")
    
    # ADDED "Deep Dive EDA" back to the tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Signal & Scout", "🔍 Deep Dive EDA", "⚠️ Risk & Toxicity", "📉 Data Drift", "🎨 Viz Studio", "📥 Downloads"])
    
    with tab1:
        st.subheader("Feature Importance")
        if not st.session_state.importance_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            top10 = st.session_state.importance_df.head(10).iloc[::-1]
            ax.barh(top10['Feature'], top10['Importance'], color='#4A90E2')
            st.pyplot(fig)
            add_download_button(fig, "final_signal_chart") # Download button for result tab
        get_model_scout(df, target_col)
        
    with tab2:
        get_deep_eda_report(df) # Now accessible
        
    with tab3:
        st.subheader("Toxicity")
        if not st.session_state.missing_report.empty: st.dataframe(st.session_state.missing_report)
        st.metric("Outliers Removed", len(st.session_state.outlier_indices))
        st.subheader("Hazard")
        if st.session_state.risky_cols: st.error(f"PII Removed: {list(st.session_state.risky_cols)}")
        else: st.success("No PII Found.")
        
    with tab4:
        if st.session_state.drift_data: st.dataframe(pd.DataFrame(st.session_state.drift_data))
        else: st.info("No reference dataset provided.")
            
    with tab5:
        get_visualization_studio(df)
        
    with tab6:
        st.subheader("Download Results")
        with st.container():
            if st.session_state.cleaned_data is not None:
                csv = convert_df_to_csv(st.session_state.cleaned_data)
                st.download_button("📥 Download Cleaned_Data.csv", csv, "cleaned.csv", "text/csv")
            pdf = create_pdf_report(
                st.session_state.missing_report,
                len(st.session_state.outlier_indices),
                st.session_state.risky_cols,
                st.session_state.top_features,
                st.session_state.critical_features,
                st.session_state.drift_data
            )
            st.download_button("📄 Download Report.pdf", pdf, "report.pdf", "application/pdf")
            
            current_strategies = st.session_state.get('strategies', {})
            py_script = generate_cleaning_script(st.session_state.risky_cols, st.session_state.outlier_indices, current_strategies)
            st.download_button("📜 Download Pipeline Code (.py)", py_script, "cleaning_pipeline.py", "text/plain")