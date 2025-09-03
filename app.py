"""
DataMind - Platform Analisis & Data Mining Otomatis
Level 1 MVP - Main Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import time
import json
import joblib
from io import BytesIO
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix, roc_curve, auc
from scipy import stats
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="DataMind - Auto Data Analysis & Mining",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2563EB 0%, #F97316 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2563EB;
        margin: 0.5rem 0;
    }
    
    .success-card {
        background: #f0f9ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 0.5rem 0;
    }
    
    .warning-card {
        background: #fffbeb;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'statistical_tests' not in st.session_state:
    st.session_state.statistical_tests = {}
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 DataMind</h1>
        <h3>Platform Analisis & Data Mining Otomatis</h3>
        <p>Analisis Data End-to-End dalam Sekali Klik</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📋 DataMind Navigation")
    st.sidebar.markdown("---")
    
    # Main category selection
    main_category = st.sidebar.selectbox(
        "📋 Select Category:",
        [
            "🏠 Home & Upload",
            "🔍 Data Processing",
            "📈 Analysis & Visualization",
            "🤖 Machine Learning",
            "🛠️ Advanced Tools",
            "💾 Management & Help"
        ]
    )
    
    st.sidebar.markdown("---")
    
    # Add category description
    category_descriptions = {
        "🏠 Home & Upload": "Start here: Welcome page and data upload",
        "🔍 Data Processing": "Clean and prepare your data",
        "📈 Analysis & Visualization": "Explore and visualize your data",
        "🤖 Machine Learning": "Train and compare ML models",
        "🛠️ Advanced Tools": "Advanced automation features",
        "💾 Management & Help": "Save sessions and get help"
    }
    
    st.sidebar.info(f"💡 **{main_category}**\n\n{category_descriptions[main_category]}")
    
    # Sub-menu based on category
    if main_category == "🏠 Home & Upload":
        menu = st.sidebar.radio(
            "Features:",
            ["🏠 Home", "📂 Upload Data"]
        )
    
    elif main_category == "🔍 Data Processing":
        menu = st.sidebar.radio(
            "Data Processing:",
            [
                "🔍 EDA & Data Cleaning",
                "🛠️ Feature Engineering",
                "⚙️ Data Pipeline Automation"
            ]
        )
    
    elif main_category == "📈 Analysis & Visualization":
        menu = st.sidebar.radio(
            "Analysis & Visualization:",
            [
                "📊 Interactive Visualizations",
                "📈 Statistical Tests",
                "📅 Time Series Analysis",
                "📋 Dashboard Builder"
            ]
        )
    
    elif main_category == "🤖 Machine Learning":
        menu = st.sidebar.radio(
            "Machine Learning:",
            [
                "🤖 Auto Data Mining",
                "🔮 AutoML Pipeline",
                "🏆 Model Comparison",
                "🧪 A/B Testing Framework"
            ]
        )
    
    elif main_category == "🛠️ Advanced Tools":
        menu = st.sidebar.radio(
            "Advanced Tools:",
            [
                "🔮 AutoML Pipeline",
                "📋 Dashboard Builder",
                "⚙️ Data Pipeline Automation"
            ]
        )
    
    else:  # Management & Help
        menu = st.sidebar.radio(
            "Management & Help:",
            [
                "💾 Session Management",
                "ℹ️ Algorithm Guide"
            ]
        )
    
    # Handle menu selections
    if menu == "🏠 Home":
        show_home()
    elif menu == "📂 Upload Data":
        show_upload()
    elif menu == "🔍 EDA & Data Cleaning":
        show_eda_cleaning()
    elif menu == "🛠️ Feature Engineering":
        show_feature_engineering()
    elif menu == "📊 Interactive Visualizations":
        show_visualization()
    elif menu == "📈 Statistical Tests":
        show_statistical_tests()
    elif menu == "🤖 Auto Data Mining":
        show_auto_mining()
    elif menu == "🏆 Model Comparison":
        show_model_comparison()
    elif menu == "🔮 AutoML Pipeline":
        show_automl_pipeline()
    elif menu == "📋 Dashboard Builder":
        show_dashboard_builder()
    elif menu == "⚙️ Data Pipeline Automation":
        show_data_pipeline_automation()
    elif menu == "📅 Time Series Analysis":
        show_time_series_analysis()
    elif menu == "🧪 A/B Testing Framework":
        show_ab_testing()
    elif menu == "💾 Session Management":
        show_session_management()
    elif menu == "ℹ️ Algorithm Guide":
        show_guidance()

def show_home():
    st.title("Selamat Datang di DataMind! 🚀")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Apa itu DataMind?
        
        **DataMind** adalah platform berbasis web yang memudahkan Anda dalam melakukan analisis data end-to-end:
        
        - 🔍 **Exploratory Data Analysis (EDA)** - Pahami data Anda dengan mudah
        - 🧹 **Data Cleaning** - Bersihkan data dari missing values dan outliers  
        - 🛠️ **Feature Engineering** - Siapkan data untuk machine learning
        - 📊 **Visualisasi Interaktif** - Buat grafik yang menarik dan informatif
        - 🤖 **Auto Data Mining** - Machine learning otomatis tanpa coding
        - 📥 **Download Results** - Unduh data bersih untuk penggunaan lanjutan
        
        ### 🚀 Cara Memulai:
        1. Upload file CSV Anda (maksimal 20MB)
        2. Lakukan eksplorasi dan pembersihan data
        3. Pilih metode machine learning yang sesuai
        4. Lihat hasil analisis dan unduh data bersih
        """)
        
        if st.button("🚀 Mulai Analisis Data", type="primary", use_container_width=True):
            st.sidebar.selectbox(
                "Pilih Fitur:",
                [
                    "📂 Upload Data",
                    "🏠 Beranda",
                    "🔍 EDA & Data Cleaning", 
                    "🛠️ Feature Engineering",
                    "📊 Visualisasi Interaktif",
                    "🤖 Auto Data Mining",
                    "ℹ️ Panduan Algoritma"
                ],
                key="nav_from_home"
            )
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: black;">✨ Fitur Unggulan</h4>
            <ul style="color: black;">
                <li>Interface yang user-friendly</li>
                <li>Analisis otomatis tanpa coding</li>
                <li>Visualisasi interaktif</li>
                <li>Machine learning otomatis</li>
                <li>Export data hasil pembersihan</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-card">
            <h4 style="color: black;">🎓 Cocok untuk:</h4>
            <ul style="color: black;">
                <li>Mahasiswa & Peneliti</li>
                <li>Data Analyst Pemula</li>
                <li>Business Intelligence</li>
                <li>Academic Research</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_upload():
    st.title("📂 Upload Dataset")
    
    # Create tabs for different upload options
    tab1, tab2 = st.tabs(["📁 Upload Your File", "🧪 Try Sample Data"])
    
    with tab1:
        st.markdown("""
        ### 📋 File Requirements:
        - Format: CSV (.csv) or Excel (.xlsx, .xls)
        - Maximum size: 20MB
        - Encoding: UTF-8 (recommended for CSV)
        """)
        
        uploaded_file = st.file_uploader(
            "Choose your dataset file:",
            type=['csv', 'xlsx', 'xls'],
            help="Upload CSV or Excel file to start data analysis"
        )
    
    with tab2:
        st.markdown("""
        ### 🧪 Test DataMind with Sample Datasets
        
        Try our platform with carefully crafted sample datasets that demonstrate different machine learning scenarios:
        """)
        
        # Sample data selection
        sample_datasets = {
            "🏦 Loan Approval (Classification)": {
                "file": "sample_classification_data.csv",
                "description": "**Predict loan approval** - Binary classification with customer demographics, income, credit score, and loan details. Contains missing values and mixed data types.",
                "features": "13 features including age, income, credit_score, employment_years, debt_to_income, etc.",
                "target": "loan_approved (0/1)",
                "use_case": "Perfect for testing classification algorithms, missing value handling, and feature engineering"
            },
            "🏠 Real Estate Prices (Regression)": {
                "file": "sample_regression_data.csv",
                "description": "**Predict property prices** - Regression with property characteristics, location data, and neighborhood scores. Includes outliers and missing values.",
                "features": "16 features including size_sqft, bedrooms, location, neighborhood_score, crime_rate, etc.",
                "target": "price (continuous)",
                "use_case": "Ideal for testing regression algorithms, outlier detection, and correlation analysis"
            },
            "👥 Customer Segmentation (Clustering)": {
                "file": "sample_clustering_data.csv",
                "description": "**Customer behavior analysis** - Unsupervised learning with spending patterns, demographics, and online behavior data.",
                "features": "15 features including spending categories, online_activity_hours, brand_loyalty_score, etc.",
                "target": "No target (unsupervised)",
                "use_case": "Great for testing K-means clustering, data preprocessing, and customer segmentation"
            },
            "🎯 User Behavior Tracking (RL Data)": {
                "file": "sample_rl_data.csv",
                "description": "**Web analytics data** - User sessions, actions, and conversion tracking for recommendation systems and sequential analysis.",
                "features": "20 features including session_duration, click_count, conversion, device_type, etc.",
                "target": "conversion/revenue (for RL simulation)",
                "use_case": "Perfect for testing multi-armed bandit simulation and sequential decision analysis"
            }
        }
        
        selected_sample = st.selectbox(
            "Choose a sample dataset:",
            list(sample_datasets.keys()),
            help="Select a sample dataset to explore DataMind's features"
        )
        
        if selected_sample:
            sample_info = sample_datasets[selected_sample]
            
            # Display dataset information
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Description:** {sample_info['description']}")
                st.markdown(f"**Features:** {sample_info['features']}")
                st.markdown(f"**Target:** {sample_info['target']}")
                st.markdown(f"**Use Case:** {sample_info['use_case']}")
            
            with col2:
                if st.button(f"🚀 Load {selected_sample.split()[0]} Dataset", type="primary"):
                    try:
                        # Load the selected sample dataset
                        sample_file_path = f"c:\\Users\\GC\\Desktop\\Auto Analis + ML\\{sample_info['file']}"
                        data = pd.read_csv(sample_file_path)
                        
                        # Store data in session state
                        st.session_state.data = data
                        st.session_state.cleaned_data = None  # Reset cleaned data
                        st.session_state.processed_data = None  # Reset processed data
                        
                        # Add to analysis history
                        history_entry = {
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'action': 'Sample Data Loaded',
                            'details': f"Dataset: {selected_sample}, Shape: {data.shape}"
                        }
                        st.session_state.analysis_history.append(history_entry)
                        
                        st.success(f"✅ Sample dataset loaded successfully! {data.shape[0]} rows and {data.shape[1]} columns.")
                        
                        # Show basic info
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Rows", data.shape[0])
                        with col2:
                            st.metric("Columns", data.shape[1])
                        with col3:
                            st.metric("Missing Values", data.isnull().sum().sum())
                        with col4:
                            st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                        
                        # Preview data
                        st.subheader("👀 Preview Data")
                        st.dataframe(data.head(10), use_container_width=True)
                        
                        st.info("🎯 **Next Steps:** Navigate to 'EDA & Data Cleaning' to start exploring this dataset!")
                        
                        # Auto navigate suggestion
                        if st.button("➡️ Start EDA & Cleaning", type="secondary"):
                            st.info("Please select 'EDA & Data Cleaning' from the sidebar menu to continue.")
                        
                        return  # Exit function after loading sample data
                        
                    except Exception as e:
                        st.error(f"❌ Error loading sample dataset: {str(e)}")
                        st.info("💡 Make sure all sample data files are in the correct directory.")
        
        # Additional information about sample datasets
        with st.expander("ℹ️ About Sample Datasets"):
            st.markdown("""
            **These sample datasets are specifically designed to:**
            
            - ✅ **Test all DataMind features** - Each dataset contains realistic data quality issues
            - 🧪 **Demonstrate best practices** - Learn proper data analysis workflows
            - 🎯 **Compare algorithms** - See how different ML models perform
            - 📊 **Explore visualizations** - Create meaningful charts and insights
            - 🔍 **Practice data cleaning** - Handle missing values, outliers, and inconsistencies
            
            **Data Quality Features:**
            - Missing values in strategic locations
            - Mixed data types (numerical, categorical, text)
            - Realistic outliers and noise
            - Typos and inconsistencies for testing cleaning features
            """)
    
    # Only process uploaded file if we're in the upload tab and a file is uploaded
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try different encodings for CSV
                try:
                    data = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)  # Reset file pointer
                    try:
                        data = pd.read_csv(uploaded_file, encoding='latin-1')
                        st.warning("⚠️ File read using latin-1 encoding")
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        data = pd.read_csv(uploaded_file, encoding='cp1252')
                        st.warning("⚠️ File read using cp1252 encoding")
            
            elif file_extension in ['xlsx', 'xls']:
                # Read Excel file
                excel_file = pd.ExcelFile(uploaded_file)
                
                if len(excel_file.sheet_names) > 1:
                    st.info(f"📄 Excel file has {len(excel_file.sheet_names)} sheets")
                    selected_sheet = st.selectbox(
                        "Choose sheet to analyze:",
                        excel_file.sheet_names
                    )
                    data = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                else:
                    data = pd.read_excel(uploaded_file)
            
            # Store data in session state
            st.session_state.data = data
            st.session_state.cleaned_data = None  # Reset cleaned data
            st.session_state.processed_data = None  # Reset processed data
            
            # Add to analysis history
            history_entry = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'action': 'Data Upload',
                'details': f"File: {uploaded_file.name}, Shape: {data.shape}"
            }
            st.session_state.analysis_history.append(history_entry)
            
            st.success(f"✅ File uploaded successfully! Dataset has {data.shape[0]} rows and {data.shape[1]} columns.")
            
            # Show basic info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", data.shape[0])
            with col2:
                st.metric("Columns", data.shape[1])
            with col3:
                st.metric("Missing Values", data.isnull().sum().sum())
            with col4:
                st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # Preview data
            st.subheader("👀 Preview Data")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Data types and missing values analysis
            st.subheader("📊 Dataset Info")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Types:**")
                dtype_df = pd.DataFrame({
                    'Column': data.dtypes.index,
                    'Data Type': data.dtypes.values,
                    'Non-Null Count': data.count().values
                })
                st.dataframe(dtype_df, use_container_width=True)
            
            with col2:
                st.write("**Missing Values Analysis:**")
                missing_df = pd.DataFrame({
                    'Column': data.columns,
                    'Missing Count': data.isnull().sum().values,
                    'Missing %': (data.isnull().sum() / len(data) * 100).round(2).values
                })
                missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
                
                if len(missing_df) > 0:
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("✅ No missing values found!")
            
            if st.button("➡️ Continue to EDA & Cleaning", type="primary"):
                st.success("Data ready for analysis! Please select 'EDA & Data Cleaning' from the sidebar menu.")
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("💡 Tips: Make sure your file is valid and not corrupted. For CSV files, try using UTF-8 encoding.")

def show_eda_cleaning():
    st.title("🔍 EDA & Data Cleaning")
    
    if st.session_state.data is None:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu!")
        return
    
    # Use cleaned data if available, otherwise use original data
    if st.session_state.cleaned_data is not None:
        data = st.session_state.cleaned_data.copy()
        st.info("🧹 Menggunakan data yang sudah dibersihkan")
    else:
        data = st.session_state.data.copy()
        st.info("📂 Menggunakan data original")
    
    tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis", "🧹 Data Cleaning", "💾 Download Data"])
    
    with tab1:
        st.subheader("📈 Statistik Deskriptif")
        
        # Numeric columns statistics
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.write("**Kolom Numerik:**")
            st.dataframe(data[numeric_cols].describe(), use_container_width=True)
            
            # Distribution plots
            st.subheader("📊 Distribusi Data Numerik")
            selected_col = st.selectbox("Pilih kolom untuk visualisasi distribusi:", numeric_cols)
            
            if selected_col:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_hist = px.histogram(data, x=selected_col, title=f"Histogram - {selected_col}")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    fig_box = px.box(data, y=selected_col, title=f"Box Plot - {selected_col}")
                    st.plotly_chart(fig_box, use_container_width=True)
        
        # Categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            st.subheader("📋 Analisis Data Kategorikal")
            selected_cat = st.selectbox("Pilih kolom kategorikal:", categorical_cols)
            
            if selected_cat:
                value_counts = data[selected_cat].value_counts()
                st.write(f"**Distribusi {selected_cat}:**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(value_counts.head(10), use_container_width=True)
                
                with col2:
                    fig_bar = px.bar(x=value_counts.index[:10], y=value_counts.values[:10], 
                                   title=f"Top 10 Values - {selected_cat}")
                    st.plotly_chart(fig_bar, use_container_width=True)
        
        # Correlation matrix
        if len(numeric_cols) > 1:
            st.subheader("🔗 Matriks Korelasi")
            corr_matrix = data[numeric_cols].corr()
            fig_corr = px.imshow(corr_matrix, 
                               text_auto=True, 
                               aspect="auto",
                               title="Correlation Heatmap")
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab2:
        st.subheader("🧹 Pembersihan Data")
        
        # Column deletion feature
        st.markdown("### 🗑️ Hapus Kolom")
        cols_to_delete = st.multiselect(
            "Pilih kolom yang ingin dihapus:",
            data.columns.tolist(),
            help="Pilih kolom yang tidak diperlukan untuk analisis"
        )
        
        if cols_to_delete:
            if st.button("🗑️ Hapus Kolom Terpilih", type="secondary"):
                data = data.drop(columns=cols_to_delete)
                st.success(f"✅ {len(cols_to_delete)} kolom berhasil dihapus: {', '.join(cols_to_delete)}")
                st.session_state.cleaned_data = data
                st.rerun()
        
        # Typo correction feature
        st.markdown("### ✏️ Perbaiki Typo")
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            col_to_fix = st.selectbox("Pilih kolom untuk memperbaiki typo:", categorical_cols)
            
            if col_to_fix:
                # Get current data (use cleaned data if available)
                current_data = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else data
                unique_values = current_data[col_to_fix].unique()
                
                st.write(f"**Nilai unik dalam kolom '{col_to_fix}':**")
                
                # Show current values
                value_counts = current_data[col_to_fix].value_counts()
                st.dataframe(value_counts.head(20), use_container_width=True)
                
                # Typo correction interface
                st.markdown("**Koreksi Typo:**")
                
                # Use form to prevent immediate execution
                with st.form(f"typo_form_{col_to_fix}"):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        old_value = st.selectbox("Nilai yang salah:", unique_values)
                    
                    with col2:
                        new_value = st.text_input("Nilai yang benar:", value="")
                    
                    with col3:
                        submit_typo = st.form_submit_button("🔄 Ganti")
                    
                    if submit_typo:
                        if old_value and new_value and old_value != new_value:
                            # Work with the current data
                            working_data = current_data.copy()
                            count_replaced = (working_data[col_to_fix] == old_value).sum()
                            working_data[col_to_fix] = working_data[col_to_fix].replace(old_value, new_value)
                            
                            # Update session state
                            st.session_state.cleaned_data = working_data
                            
                            st.success(f"✅ '{old_value}' berhasil diganti dengan '{new_value}' pada {count_replaced} baris")
                            st.info("🔄 Halaman akan refresh untuk menampilkan perubahan. Klik tombol lagi jika diperlukan.")
                            
                            # Use a more compatible rerun method
                            time.sleep(0.1)  # Small delay for better UX
                            st.rerun()
                        else:
                            st.warning("⚠️ Pastikan nilai lama dan baru berbeda dan tidak kosong")
                
                # Bulk typo correction
                st.markdown("**Koreksi Massal (Advanced):**")
                with st.expander("🔧 Koreksi Otomatis"):
                    if st.button("🧹 Bersihkan Spasi Berlebih"):
                        original_data = data[col_to_fix].copy()
                        data[col_to_fix] = data[col_to_fix].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
                        changes = (original_data != data[col_to_fix]).sum()
                        if changes > 0:
                            st.success(f"✅ {changes} nilai dibersihkan dari spasi berlebih")
                            st.session_state.cleaned_data = data
                        else:
                            st.info("ℹ️ Tidak ada spasi berlebih yang ditemukan")
                    
                    if st.button("📝 Standardisasi Kapitalisasi"):
                        original_data = data[col_to_fix].copy()
                        capitalization = st.radio(
                            "Pilih format:",
                            ["Title Case", "UPPER CASE", "lower case"],
                            horizontal=True,
                            key="cap_format"
                        )
                        
                        if capitalization == "Title Case":
                            data[col_to_fix] = data[col_to_fix].astype(str).str.title()
                        elif capitalization == "UPPER CASE":
                            data[col_to_fix] = data[col_to_fix].astype(str).str.upper()
                        else:
                            data[col_to_fix] = data[col_to_fix].astype(str).str.lower()
                        
                        changes = (original_data != data[col_to_fix]).sum()
                        st.success(f"✅ {changes} nilai distandarisasi ke {capitalization}")
                        st.session_state.cleaned_data = data
        else:
            st.info("📊 Dataset tidak memiliki kolom kategorikal untuk diperbaiki typo-nya.")
        
        # Missing values handling
        st.markdown("### 🔍 Handle Missing Values")
        missing_summary = data.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0].index.tolist()
        
        if len(missing_cols) > 0:
            st.write(f"**Kolom dengan missing values:** {len(missing_cols)}")
            
            for col in missing_cols:
                st.write(f"**{col}** - {missing_summary[col]} missing values ({(missing_summary[col]/len(data)*100):.1f}%)")
                
                col1, col2 = st.columns(2)
                with col1:
                    if data[col].dtype in ['int64', 'float64']:
                        method = st.selectbox(f"Metode untuk {col}:", 
                                            ["Drop rows", "Mean", "Median", "Mode", "Forward Fill", "Backward Fill"], 
                                            key=f"missing_{col}")
                    else:
                        method = st.selectbox(f"Metode untuk {col}:", 
                                            ["Drop rows", "Mode", "Fill with 'Unknown'", "Forward Fill", "Backward Fill"], 
                                            key=f"missing_{col}")
                
                with col2:
                    if st.button(f"Apply to {col}", key=f"apply_{col}"):
                        original_count = len(data)
                        if method == "Drop rows":
                            data = data.dropna(subset=[col])
                            dropped_count = original_count - len(data)
                            st.success(f"✅ {dropped_count} baris dengan missing values dihapus")
                        elif method == "Mean":
                            mean_val = data[col].mean()
                            data[col].fillna(mean_val, inplace=True)
                            st.success(f"✅ Missing values diisi dengan mean: {mean_val:.2f}")
                        elif method == "Median":
                            median_val = data[col].median()
                            data[col].fillna(median_val, inplace=True)
                            st.success(f"✅ Missing values diisi dengan median: {median_val:.2f}")
                        elif method == "Mode":
                            if not data[col].mode().empty:
                                mode_val = data[col].mode()[0]
                                data[col].fillna(mode_val, inplace=True)
                                st.success(f"✅ Missing values diisi dengan mode: {mode_val}")
                            else:
                                st.error("❌ Tidak dapat menghitung mode untuk kolom ini")
                        elif method == "Forward Fill":
                            data[col].fillna(method='ffill', inplace=True)
                            st.success(f"✅ Missing values diisi dengan forward fill")
                        elif method == "Backward Fill":
                            data[col].fillna(method='bfill', inplace=True)
                            st.success(f"✅ Missing values diisi dengan backward fill")
                        elif method == "Fill with 'Unknown'":
                            data[col].fillna('Unknown', inplace=True)
                            st.success(f"✅ Missing values diisi dengan 'Unknown'")
                        
                        st.session_state.cleaned_data = data
        else:
            st.success("✅ Tidak ada missing values dalam dataset!")
        
        # Outlier detection
        st.markdown("### 📊 Deteksi Outlier")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            outlier_col = st.selectbox("Pilih kolom untuk deteksi outlier:", numeric_cols)
            outlier_method = st.selectbox("Metode deteksi:", ["Z-Score", "IQR"])
            
            if outlier_col:
                if outlier_method == "Z-Score":
                    threshold = st.slider("Z-Score threshold:", 1.0, 5.0, 3.0, 0.1)
                    z_scores = np.abs((data[outlier_col] - data[outlier_col].mean()) / data[outlier_col].std())
                    outliers = data[z_scores > threshold]
                    st.write(f"**Outliers detected (Z-score > {threshold}):** {len(outliers)}")
                    
                    if len(outliers) > 0:
                        st.write("**Preview outliers:**")
                        st.dataframe(outliers[[outlier_col]].head(), use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🗑️ Remove Outliers"):
                                data = data[z_scores <= threshold]
                                st.success(f"✅ {len(outliers)} outliers removed!")
                                st.session_state.cleaned_data = data
                        
                        with col2:
                            if st.button("🔄 Cap Outliers"):
                                # Cap outliers to threshold values
                                mean_val = data[outlier_col].mean()
                                std_val = data[outlier_col].std()
                                lower_bound = mean_val - threshold * std_val
                                upper_bound = mean_val + threshold * std_val
                                
                                data[outlier_col] = data[outlier_col].clip(lower=lower_bound, upper=upper_bound)
                                st.success(f"✅ Outliers capped to range [{lower_bound:.2f}, {upper_bound:.2f}]")
                                st.session_state.cleaned_data = data
                    
                elif outlier_method == "IQR":
                    Q1 = data[outlier_col].quantile(0.25)
                    Q3 = data[outlier_col].quantile(0.75)
                    IQR = Q3 - Q1
                    multiplier = st.slider("IQR multiplier:", 1.0, 3.0, 1.5, 0.1)
                    
                    outliers = data[(data[outlier_col] < (Q1 - multiplier * IQR)) | 
                                  (data[outlier_col] > (Q3 + multiplier * IQR))]
                    st.write(f"**Outliers detected (IQR method, multiplier={multiplier}):** {len(outliers)}")
                    
                    if len(outliers) > 0:
                        st.write("**Preview outliers:**")
                        st.dataframe(outliers[[outlier_col]].head(), use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("🗑️ Remove Outliers", key="iqr_remove"):
                                data = data[(data[outlier_col] >= (Q1 - multiplier * IQR)) & 
                                          (data[outlier_col] <= (Q3 + multiplier * IQR))]
                                st.success(f"✅ {len(outliers)} outliers removed!")
                                st.session_state.cleaned_data = data
                        
                        with col2:
                            if st.button("🔄 Cap Outliers", key="iqr_cap"):
                                lower_bound = Q1 - multiplier * IQR
                                upper_bound = Q3 + multiplier * IQR
                                data[outlier_col] = data[outlier_col].clip(lower=lower_bound, upper=upper_bound)
                                st.success(f"✅ Outliers capped to range [{lower_bound:.2f}, {upper_bound:.2f}]")
                                st.session_state.cleaned_data = data
        else:
            st.info("📊 Dataset tidak memiliki kolom numerik untuk deteksi outlier.")
    
    with tab3:
        st.subheader("💾 Download Data Bersih")
        
        cleaned_data = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else data
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Baris Asli", st.session_state.data.shape[0])
        with col2:
            st.metric("Baris Setelah Cleaning", cleaned_data.shape[0])
        with col3:
            rows_removed = st.session_state.data.shape[0] - cleaned_data.shape[0]
            st.metric("Baris Dihapus", rows_removed)
        
        st.subheader("👀 Preview Data Bersih")
        st.dataframe(cleaned_data.head(), use_container_width=True)
        
        # Download button
        csv = cleaned_data.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv",
            type="primary"
        )
        
        st.session_state.cleaned_data = cleaned_data

def show_feature_engineering():
    st.title("🔧 Feature Engineering")
    
    if st.session_state.data is None:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu!")
        return
    
    # Use cleaned data if available, otherwise use original data
    data = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data.copy()
    
    tab1, tab2, tab3 = st.tabs(["🔖 Encoding", "📊 Scaling", "📁 Export Processed Data"])
    
    with tab1:
        st.subheader("🔖 Categorical Encoding")
        
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) > 0:
            st.write(f"**Kolom kategorikal ditemukan:** {len(categorical_cols)}")
            
            for col in categorical_cols:
                st.write(f"**{col}** - {data[col].nunique()} unique values")
                
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    encoding_method = st.selectbox(f"Encoding method for {col}:", 
                                                 ["None", "Label Encoding", "One-Hot Encoding"], 
                                                 key=f"encoding_{col}")
                
                with col2:
                    if data[col].nunique() <= 10:
                        st.write("Values:", list(data[col].unique()[:5]), "..." if data[col].nunique() > 5 else "")
                    else:
                        st.write(f"Too many unique values ({data[col].nunique()})")
                
                with col3:
                    if st.button(f"Apply", key=f"apply_encoding_{col}"):
                        if encoding_method == "Label Encoding":
                            le = LabelEncoder()
                            data[f"{col}_encoded"] = le.fit_transform(data[col].astype(str))
                            st.success(f"✅ Label encoding applied to {col}")
                            
                        elif encoding_method == "One-Hot Encoding":
                            if data[col].nunique() <= 10:  # Limit one-hot to avoid too many columns
                                dummies = pd.get_dummies(data[col], prefix=col)
                                data = pd.concat([data, dummies], axis=1)
                                st.success(f"✅ One-hot encoding applied to {col}")
                            else:
                                st.warning(f"⚠️ Too many unique values for one-hot encoding. Use label encoding instead.")
                
                st.session_state.processed_data = data
        else:
            st.info("📊 Dataset tidak memiliki kolom kategorikal.")
    
    with tab2:
        st.subheader("📊 Feature Scaling")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            st.write(f"**Kolom numerik ditemukan:** {len(numeric_cols)}")
            
            scaling_method = st.selectbox("Pilih metode scaling:", 
                                        ["None", "StandardScaler (Z-score)", "MinMaxScaler (0-1)"])
            
            cols_to_scale = st.multiselect("Pilih kolom untuk di-scale:", numeric_cols)
            
            if scaling_method != "None" and len(cols_to_scale) > 0:
                if st.button("🚀 Apply Scaling"):
                    if scaling_method == "StandardScaler (Z-score)":
                        scaler = StandardScaler()
                        data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
                        st.success(f"✅ StandardScaler applied to {len(cols_to_scale)} columns")
                        
                    elif scaling_method == "MinMaxScaler (0-1)":
                        scaler = MinMaxScaler()
                        data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])
                        st.success(f"✅ MinMaxScaler applied to {len(cols_to_scale)} columns")
                    
                    st.session_state.processed_data = data
            
            # Show before/after comparison
            if len(cols_to_scale) > 0:
                st.subheader("🔄 Before vs After Scaling")
                comparison_col = st.selectbox("Pilih kolom untuk perbandingan:", cols_to_scale)
                
                if comparison_col:
                    original_data = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Before Scaling:**")
                        st.write(f"Mean: {original_data[comparison_col].mean():.3f}")
                        st.write(f"Std: {original_data[comparison_col].std():.3f}")
                        st.write(f"Min: {original_data[comparison_col].min():.3f}")
                        st.write(f"Max: {original_data[comparison_col].max():.3f}")
                    
                    with col2:
                        st.write("**After Scaling:**")
                        st.write(f"Mean: {data[comparison_col].mean():.3f}")
                        st.write(f"Std: {data[comparison_col].std():.3f}")
                        st.write(f"Min: {data[comparison_col].min():.3f}")
                        st.write(f"Max: {data[comparison_col].max():.3f}")
        else:
            st.info("📊 Dataset tidak memiliki kolom numerik.")
    
    with tab3:
        st.subheader("📁 Export Processed Data")
        
        processed_data = st.session_state.processed_data if st.session_state.processed_data is not None else data
        
        st.write(f"**Shape:** {processed_data.shape[0]} rows, {processed_data.shape[1]} columns")
        
        # Show summary of changes
        original_cols = st.session_state.data.shape[1]
        processed_cols = processed_data.shape[1]
        new_cols = processed_cols - original_cols
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Columns", original_cols)
        with col2:
            st.metric("Processed Columns", processed_cols)
        with col3:
            st.metric("New Features", new_cols)
        
        st.subheader("👀 Preview Processed Data")
        st.dataframe(processed_data.head(), use_container_width=True)
        
        # Download button
        csv = processed_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Processed CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv",
            type="primary"
        )

def show_visualization():
    st.title("📊 Visualisasi Interaktif")
    
    if st.session_state.data is None:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu!")
        return
    
    # Use the most processed version of data available
    if st.session_state.processed_data is not None:
        data = st.session_state.processed_data
        st.info("📊 Menggunakan data yang sudah diproses")
    elif st.session_state.cleaned_data is not None:
        data = st.session_state.cleaned_data
        st.info("🧹 Menggunakan data yang sudah dibersihkan")
    else:
        data = st.session_state.data
        st.info("📂 Menggunakan data original")
    
    tab1, tab2, tab3 = st.tabs(["📈 Basic Plots", "🔗 Relationship Analysis", "🔮 Advanced Plots"])
    
    with tab1:
        st.subheader("📈 Visualisasi Dasar")
        
        plot_type = st.selectbox("Pilih jenis plot:", 
                                ["Histogram", "Box Plot", "Bar Chart", "Scatter Plot", "Violin Plot"])
        
        if plot_type == "Histogram":
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col = st.selectbox("Pilih kolom:", numeric_cols)
                bins = st.slider("Jumlah bins:", 10, 100, 30)
                
                fig = px.histogram(data, x=col, nbins=bins, title=f"Histogram - {col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Box Plot":
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col = st.selectbox("Pilih kolom:", numeric_cols)
                fig = px.box(data, y=col, title=f"Box Plot - {col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Bar Chart":
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                col = st.selectbox("Pilih kolom:", categorical_cols)
                top_n = st.slider("Tampilkan top N values:", 5, 20, 10)
                value_counts = data[col].value_counts().head(top_n)
                fig = px.bar(x=value_counts.index, y=value_counts.values,
                           title=f"Bar Chart - Top {top_n} {col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Scatter Plot":
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("X-axis:", numeric_cols)
                y_col = st.selectbox("Y-axis:", [col for col in numeric_cols if col != x_col])
                fig = px.scatter(data, x=x_col, y=y_col, title=f"Scatter Plot - {y_col} vs {x_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_type == "Violin Plot":
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col = st.selectbox("Pilih kolom:", numeric_cols)
                fig = px.violin(data, y=col, title=f"Violin Plot - {col}")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔗 Analisis Hubungan")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            selected_cols = st.multiselect("Pilih kolom untuk korelasi:", 
                                          numeric_cols, 
                                          default=numeric_cols[:min(5, len(numeric_cols))])
            
            if len(selected_cols) >= 2:
                corr_matrix = data[selected_cols].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                              title="Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔮 Advanced Visualizations")
        
        advanced_type = st.selectbox("Pilih visualisasi:", 
                                   ["PCA Visualization", "Distribution Comparison", "3D Scatter"])
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if advanced_type == "PCA Visualization":
            if len(numeric_cols) >= 3:
                st.markdown("### Principal Component Analysis (PCA)")
                
                # Select features for PCA
                selected_features = st.multiselect(
                    "Select features for PCA:",
                    numeric_cols,
                    default=numeric_cols[:min(5, len(numeric_cols))]
                )
                
                if len(selected_features) >= 2:
                    # Prepare data
                    pca_data = data[selected_features].dropna()
                    
                    if len(pca_data) > 0:
                        # Standardize data
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(pca_data)
                        
                        # Apply PCA
                        pca = PCA(n_components=min(3, len(selected_features)))
                        pca_result = pca.fit_transform(scaled_data)
                        
                        # Create PCA dataframe
                        pca_df = pd.DataFrame({
                            'PC1': pca_result[:, 0],
                            'PC2': pca_result[:, 1]
                        })
                        
                        if pca_result.shape[1] >= 3:
                            pca_df['PC3'] = pca_result[:, 2]
                        
                        # Add categorical variable for coloring
                        if categorical_cols:
                            color_var = st.selectbox("Color by (optional):", ["None"] + categorical_cols)
                            if color_var != "None":
                                pca_df[color_var] = data[color_var].values[:len(pca_df)]
                        
                        # Plot PCA results
                        if pca_result.shape[1] >= 3 and 'PC3' in pca_df.columns:
                            if categorical_cols and color_var != "None":
                                fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', 
                                                  color=color_var, title="PCA 3D Visualization")
                            else:
                                fig = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', 
                                                  title="PCA 3D Visualization")
                        else:
                            if categorical_cols and color_var != "None":
                                fig = px.scatter(pca_df, x='PC1', y='PC2', 
                                               color=color_var, title="PCA 2D Visualization")
                            else:
                                fig = px.scatter(pca_df, x='PC1', y='PC2', 
                                               title="PCA 2D Visualization")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show explained variance
                        st.subheader("Explained Variance Ratio")
                        variance_df = pd.DataFrame({
                            'Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                            'Explained Variance': pca.explained_variance_ratio_,
                            'Cumulative Variance': np.cumsum(pca.explained_variance_ratio_)
                        })
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.dataframe(variance_df)
                        
                        with col2:
                            fig_var = px.bar(variance_df, x='Component', y='Explained Variance',
                                           title="Explained Variance by Component")
                            st.plotly_chart(fig_var, use_container_width=True)
            
            else:
                st.warning("⚠️ PCA requires at least 3 numeric columns")
        
        elif advanced_type == "Distribution Comparison":
            if numeric_cols and categorical_cols:
                numeric_col = st.selectbox("Kolom numerik:", numeric_cols, key="dist_num")
                categorical_col = st.selectbox("Kolom kategorikal:", categorical_cols, key="dist_cat")
                
                fig = px.histogram(data, x=numeric_col, color=categorical_col,
                                 title=f"Distribution of {numeric_col} by {categorical_col}",
                                 marginal="box")
                st.plotly_chart(fig, use_container_width=True)
        
        elif advanced_type == "3D Scatter":
            if len(numeric_cols) >= 3:
                x_col = st.selectbox("X-axis:", numeric_cols, key="3d_x")
                y_col = st.selectbox("Y-axis:", [col for col in numeric_cols if col != x_col], key="3d_y")
                z_col = st.selectbox("Z-axis:", [col for col in numeric_cols if col not in [x_col, y_col]], key="3d_z")
                
                # Optional color coding
                if categorical_cols:
                    color_col = st.selectbox("Color by (optional):", ["None"] + categorical_cols)
                    if color_col != "None":
                        fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col, color=color_col,
                                          title=f"3D Scatter Plot")
                    else:
                        fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col,
                                          title=f"3D Scatter Plot")
                else:
                    fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col,
                                      title=f"3D Scatter Plot")
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ 3D scatter plot requires at least 3 numeric columns")

def show_auto_mining():
    st.title("🤖 Auto Data Mining")
    
    if st.session_state.data is None:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu!")
        return
    
    # Use the most processed version of data available
    if st.session_state.processed_data is not None:
        data = st.session_state.processed_data
        st.info("📊 Menggunakan data yang sudah diproses")
    elif st.session_state.cleaned_data is not None:
        data = st.session_state.cleaned_data
        st.info("🧹 Menggunakan data yang sudah dibersihkan")
    else:
        data = st.session_state.data
        st.warning("⚠️ Menggunakan data original")
    
    tab1, tab2, tab3 = st.tabs(["📊 Supervised", "🔍 Unsupervised", "🎮 Reinforcement"])
    
    with tab1:
        st.subheader("📊 Supervised Learning")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("❌ Need at least 2 numeric columns. Please do feature engineering first.")
            return
        
        problem_type = st.selectbox("Problem type:", ["Classification", "Regression"])
        target_col = st.selectbox("Target column:", numeric_cols)
        
        # Auto-suggest problem type based on target variable
        if target_col:
            target_data = data[target_col].dropna()
            unique_vals = target_data.nunique()
            total_vals = len(target_data)
            
            if unique_vals <= 10:
                suggested_type = "Classification"
                reason = f"Target has {unique_vals} unique values (≤ 10)"
            elif unique_vals / total_vals < 0.05:
                suggested_type = "Classification"
                reason = f"Target has {unique_vals} unique values ({unique_vals/total_vals:.1%} of total)"
            else:
                suggested_type = "Regression"
                reason = f"Target has {unique_vals} unique values ({unique_vals/total_vals:.1%} of total)"
            
            if suggested_type != problem_type:
                st.info(f"💡 **Suggestion**: Consider using **{suggested_type}** - {reason}")
            else:
                st.success(f"✅ **Good choice**: {problem_type} is appropriate - {reason}")
        feature_cols = st.multiselect("Feature columns:", 
                                    [col for col in numeric_cols if col != target_col],
                                    default=[col for col in numeric_cols if col != target_col][:3])
        
        if len(feature_cols) > 0 and target_col:
            if st.button("🚀 Train Models", type="primary"):
                try:
                    X = data[feature_cols].dropna()
                    y = data[target_col].dropna()
                    
                    # Align X and y
                    common_idx = X.index.intersection(y.index)
                    X = X.loc[common_idx]
                    y = y.loc[common_idx]
                    
                    if len(X) < 10:
                        st.error("❌ Not enough data after cleaning.")
                        return
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    if problem_type == "Classification":
                        # Better handling for classification target preparation
                        unique_values = y.nunique()
                        
                        if unique_values > 20:
                            # Convert to binary classification
                            st.info(f"🔄 Converting continuous target to binary classification using median threshold")
                            median_val = y.median()
                            y_binary = (y > median_val).astype(int)
                            
                            # Update both train and test sets
                            y_train_binary = (y_train > median_val).astype(int)
                            y_test_binary = (y_test > median_val).astype(int)
                            
                            # Verify we have both classes
                            if len(np.unique(y_train_binary)) < 2:
                                st.error("❌ Cannot create binary classification - all values fall on one side of median. Try regression instead.")
                                return
                            
                            y_train = y_train_binary
                            y_test = y_test_binary
                            y = y_binary
                            
                            st.success(f"✅ Binary classification: {len(y[y==0])} samples below median, {len(y[y==1])} samples above median")
                            
                        elif unique_values > 10:
                            # Try to convert to discrete classes
                            st.info(f"🔄 Converting to discrete classes using quantile binning")
                            
                            # Use quantile-based binning for moderate number of unique values
                            y_binned = pd.qcut(y, q=min(5, unique_values), labels=False, duplicates='drop')
                            
                            # Update train and test
                            y_train_binned = pd.qcut(y_train, q=min(5, unique_values), labels=False, duplicates='drop')
                            y_test_binned = pd.qcut(y_test, q=min(5, unique_values), labels=False, duplicates='drop')
                            
                            # Check for NaN values after binning
                            if y_train_binned.isna().any() or y_test_binned.isna().any():
                                st.warning("⚠️ Quantile binning created NaN values. Falling back to binary classification.")
                                median_val = y.median()
                                y_train = (y_train > median_val).astype(int)
                                y_test = (y_test > median_val).astype(int)
                                y = (y > median_val).astype(int)
                            else:
                                y_train = y_train_binned.astype(int)
                                y_test = y_test_binned.astype(int)
                                y = y_binned.astype(int)
                            
                            st.success(f"✅ Multi-class classification: {unique_values} classes created")
                        
                        else:
                            # Already discrete, just ensure integer type
                            y_train = y_train.astype(int)
                            y_test = y_test.astype(int)
                            y = y.astype(int)
                            
                            st.success(f"✅ Using original discrete values: {unique_values} classes")
                        
                        # Final validation
                        train_classes = len(np.unique(y_train))
                        test_classes = len(np.unique(y_test))
                        
                        if train_classes < 2:
                            st.error("❌ Training set has less than 2 classes. Cannot perform classification.")
                            return
                        
                        # Enhanced model selection with advanced algorithms
                        models = {
                            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                            "SVM": SVC(random_state=42, probability=True),
                            "Neural Network": MLPClassifier(random_state=42, max_iter=500)
                        }
                        
                        # Add XGBoost if available
                        if XGBOOST_AVAILABLE:
                            models["XGBoost"] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                        
                        # Add LightGBM if available
                        if LIGHTGBM_AVAILABLE:
                            models["LightGBM"] = lgb.LGBMClassifier(random_state=42, verbose=-1)
                        
                        results = {}
                        model_objects = {}
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, (name, model) in enumerate(models.items()):
                            status_text.text(f"Training {name}...")
                            
                            try:
                                # Train model
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                
                                # Calculate metrics
                                accuracy = accuracy_score(y_test, y_pred)
                                
                                # Cross-validation score with error handling
                                try:
                                    cv_scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                                    cv_mean = cv_scores.mean()
                                    cv_std = cv_scores.std()
                                except Exception as cv_error:
                                    st.warning(f"⚠️ Cross-validation failed for {name}: {str(cv_error)}")
                                    cv_mean = accuracy  # Use test accuracy as fallback
                                    cv_std = 0.0
                                
                                results[name] = {
                                    'accuracy': accuracy,
                                    'cv_mean': cv_mean,
                                    'cv_std': cv_std
                                }
                                
                                model_objects[name] = model
                                
                                # Store in session state
                                st.session_state.model_results[name] = results[name]
                                
                            except Exception as e:
                                st.warning(f"⚠️ {name} failed: {str(e)[:100]}...")
                                continue
                            
                            progress_bar.progress((i + 1) / len(models))
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if results:
                            st.success("✅ Classification models trained successfully!")
                            
                            # Display results
                            results_df = pd.DataFrame(results).T
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Best model
                            best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
                            st.success(f"🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.3f})")
                            
                            # Visualization
                            fig = px.bar(x=list(results.keys()), 
                                       y=[results[model]['accuracy'] for model in results.keys()],
                                       title="Model Accuracy Comparison")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Feature importance (if available)
                            best_model_obj = model_objects[best_model[0]]
                            if hasattr(best_model_obj, 'feature_importances_'):
                                st.subheader("📊 Feature Importance")
                                importance_df = pd.DataFrame({
                                    'Feature': feature_cols,
                                    'Importance': best_model_obj.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                
                                fig_imp = px.bar(importance_df, x='Importance', y='Feature', 
                                               orientation='h', title="Feature Importance")
                                st.plotly_chart(fig_imp, use_container_width=True)
                    
                    else:  # Regression
                        # Enhanced regression models
                        models = {
                            "Linear Regression": LinearRegression(),
                            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                            "SVR": SVR(),
                            "Neural Network": MLPRegressor(random_state=42, max_iter=500)
                        }
                        
                        # Add XGBoost if available
                        if XGBOOST_AVAILABLE:
                            models["XGBoost"] = xgb.XGBRegressor(random_state=42)
                        
                        # Add LightGBM if available
                        if LIGHTGBM_AVAILABLE:
                            models["LightGBM"] = lgb.LGBMRegressor(random_state=42, verbose=-1)
                        
                        results = {}
                        model_objects = {}
                        
                        # Progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, (name, model) in enumerate(models.items()):
                            status_text.text(f"Training {name}...")
                            
                            try:
                                # Train model
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                
                                # Calculate metrics
                                r2 = r2_score(y_test, y_pred)
                                mse = mean_squared_error(y_test, y_pred)
                                
                                # Cross-validation score with error handling
                                try:
                                    cv_scores = cross_val_score(model, X, y, cv=3, scoring='r2')
                                    cv_mean = cv_scores.mean()
                                    cv_std = cv_scores.std()
                                except Exception as cv_error:
                                    st.warning(f"⚠️ Cross-validation failed for {name}: {str(cv_error)[:50]}...")
                                    cv_mean = r2  # Use test R2 as fallback
                                    cv_std = 0.0
                                
                                results[name] = {
                                    'r2_score': r2,
                                    'mse': mse,
                                    'cv_mean': cv_mean,
                                    'cv_std': cv_std
                                }
                                
                                model_objects[name] = model
                                
                                # Store in session state
                                st.session_state.model_results[name] = results[name]
                                
                            except Exception as e:
                                st.warning(f"⚠️ {name} failed: {str(e)[:100]}...")
                                continue
                            
                            progress_bar.progress((i + 1) / len(models))
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if results:
                            st.success("✅ Regression models trained successfully!")
                            
                            # Display results
                            results_df = pd.DataFrame(results).T
                            st.dataframe(results_df, use_container_width=True)
                            
                            # Best model
                            best_model = max(results.items(), key=lambda x: x[1]['r2_score'])
                            st.success(f"🏆 Best Model: {best_model[0]} (R²: {best_model[1]['r2_score']:.3f})")
                            
                            # Visualization
                            fig = px.bar(x=list(results.keys()), 
                                       y=[results[model]['r2_score'] for model in results.keys()],
                                       title="Model R² Score Comparison")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Feature importance (if available)
                            best_model_obj = model_objects[best_model[0]]
                            if hasattr(best_model_obj, 'feature_importances_'):
                                st.subheader("📊 Feature Importance")
                                importance_df = pd.DataFrame({
                                    'Feature': feature_cols,
                                    'Importance': best_model_obj.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                
                                fig_imp = px.bar(importance_df, x='Importance', y='Feature', 
                                               orientation='h', title="Feature Importance")
                                st.plotly_chart(fig_imp, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with tab2:
        st.subheader("🔍 Unsupervised Learning")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("❌ Need at least 2 numeric columns.")
            return
        
        feature_cols = st.multiselect("Features for clustering:", 
                                    numeric_cols,
                                    default=numeric_cols[:min(3, len(numeric_cols))])
        
        if len(feature_cols) >= 2:
            n_clusters = st.slider("Number of clusters:", 2, 8, 3)
            
            if st.button("🔍 Run K-Means", type="primary"):
                try:
                    X = data[feature_cols].dropna()
                    
                    if len(X) < 10:
                        st.error("❌ Not enough data.")
                        return
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    st.success(f"✅ K-Means completed with {n_clusters} clusters!")
                    
                    # Show cluster distribution
                    cluster_counts = pd.Series(clusters).value_counts().sort_index()
                    st.bar_chart(cluster_counts)
                    
                    # Visualize if we have 2+ features
                    if len(feature_cols) >= 2:
                        result_df = X.copy()
                        result_df['Cluster'] = clusters
                        fig = px.scatter(result_df, x=feature_cols[0], y=feature_cols[1], 
                                       color='Cluster', title="K-Means Clustering Results")
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with tab3:
        st.subheader("🎮 Reinforcement Learning Demo")
        
        st.markdown("""
        ### Multi-Armed Bandit Simulation
        Demonstrasi sederhana RL dimana agent belajar memilih "arm" terbaik.
        """)
        
        n_arms = st.slider("Number of arms:", 2, 6, 3)
        n_episodes = st.slider("Number of episodes:", 100, 1000, 500)
        
        if st.button("🎮 Run Simulation", type="primary"):
            np.random.seed(42)
            true_rewards = np.random.uniform(0.2, 0.8, n_arms)
            
            q_values = np.zeros(n_arms)
            action_counts = np.zeros(n_arms)
            rewards_history = []
            
            epsilon = 0.1
            
            for episode in range(n_episodes):
                if np.random.random() < epsilon:
                    action = np.random.randint(n_arms)
                else:
                    action = np.argmax(q_values)
                
                reward = 1 if np.random.random() < true_rewards[action] else 0
                
                action_counts[action] += 1
                q_values[action] += (reward - q_values[action]) / action_counts[action]
                
                rewards_history.append(reward)
            
            st.success("✅ Simulation completed!")
            
            results_df = pd.DataFrame({
                'Arm': range(n_arms),
                'True Rate': true_rewards.round(3),
                'Learned Q': q_values.round(3),
                'Selections': action_counts.astype(int)
            })
            st.dataframe(results_df)
            
            cumulative_rewards = np.cumsum(rewards_history)
            fig = px.line(x=range(len(cumulative_rewards)), y=cumulative_rewards,
                        title="Cumulative Rewards")
            st.plotly_chart(fig, use_container_width=True)

def show_automl_pipeline():
    st.title("🔮 AutoML Pipeline Builder")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload dataset first!")
        return
    
    # Use the most processed version of data available
    if st.session_state.processed_data is not None:
        data = st.session_state.processed_data
        st.info("📊 Using processed data")
    elif st.session_state.cleaned_data is not None:
        data = st.session_state.cleaned_data
        st.info("🧙 Using cleaned data")
    else:
        data = st.session_state.data
        st.warning("⚠️ Using original data")
    
    tab1, tab2, tab3 = st.tabs(["🤖 Auto Pipeline", "⚙️ Custom Pipeline", "📈 Pipeline Results"])
    
    with tab1:
        st.subheader("🤖 Automated Machine Learning Pipeline")
        
        st.markdown("""
        **AutoML** automatically handles:
        - 🎯 Problem type detection (Classification/Regression)
        - 🛠️ Feature preprocessing and engineering
        - 🔍 Algorithm selection and comparison
        - ⚙️ Hyperparameter optimization
        - 📊 Model evaluation and validation
        """)
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("❌ Need at least 2 numeric columns for AutoML")
            return
        
        # Target selection
        target_col = st.selectbox("Select target variable:", numeric_cols)
        
        if target_col:
            # Automatic problem type detection
            target_data = data[target_col].dropna()
            unique_vals = target_data.nunique()
            total_vals = len(target_data)
            
            if unique_vals <= 10:
                auto_problem_type = "Classification"
                confidence = "High"
                reason = f"Target has {unique_vals} unique values (≤ 10)"
            elif unique_vals / total_vals < 0.05:
                auto_problem_type = "Classification"
                confidence = "Medium"
                reason = f"Target has {unique_vals} unique values ({unique_vals/total_vals:.1%} of total)"
            else:
                auto_problem_type = "Regression"
                confidence = "High"
                reason = f"Target has {unique_vals} unique values ({unique_vals/total_vals:.1%} of total)"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 Detected Problem Type", auto_problem_type)
            with col2:
                st.metric("📊 Confidence", confidence)
            with col3:
                st.metric("🔢 Unique Values", unique_vals)
            
            st.info(f"🧠 **Reasoning**: {reason}")
            
            # Feature selection (automatic)
            feature_cols = [col for col in numeric_cols if col != target_col]
            st.write(f"**Selected Features ({len(feature_cols)}):** {', '.join(feature_cols[:5])}{', ...' if len(feature_cols) > 5 else ''}")
            
            # AutoML Configuration
            st.subheader("⚙️ AutoML Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                max_models = st.slider("Maximum models to train:", 3, 10, 5)
                cross_validation_folds = st.slider("Cross-validation folds:", 3, 10, 5)
            
            with col2:
                hyperparameter_tuning = st.checkbox("Enable hyperparameter tuning", value=True)
                feature_selection = st.checkbox("Automatic feature selection", value=True)
            
            if st.button("🚀 Run AutoML Pipeline", type="primary"):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Data preparation
                    status_text.text("🛠️ Preparing data...")
                    X = data[feature_cols].dropna()
                    y = data[target_col].dropna()
                    
                    # Align X and y
                    common_idx = X.index.intersection(y.index)
                    X = X.loc[common_idx]
                    y = y.loc[common_idx]
                    
                    if len(X) < 10:
                        st.error("❌ Not enough data after cleaning")
                        return
                    
                    progress_bar.progress(0.1)
                    
                    # Feature selection if enabled
                    if feature_selection and len(feature_cols) > 5:
                        status_text.text("🔍 Selecting best features...")
                        from sklearn.feature_selection import SelectKBest, f_classif, f_regression
                        
                        if auto_problem_type == "Classification":
                            # Prepare target for classification
                            if unique_vals > 10:
                                y = (y > y.median()).astype(int)
                            else:
                                y = y.astype(int)
                            selector = SelectKBest(score_func=f_classif, k=min(5, len(feature_cols)))
                        else:
                            selector = SelectKBest(score_func=f_regression, k=min(5, len(feature_cols)))
                        
                        X_selected = selector.fit_transform(X, y)
                        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
                        X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
                        
                        st.info(f"🔍 **Selected Features**: {', '.join(selected_features)}")
                    
                    progress_bar.progress(0.2)
                    
                    # Split data
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Model selection based on problem type
                    if auto_problem_type == "Classification":
                        models = {
                            "Random Forest": RandomForestClassifier(random_state=42),
                            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                            "SVM": SVC(random_state=42, probability=True),
                            "Neural Network": MLPClassifier(random_state=42, max_iter=500)
                        }
                        
                        if XGBOOST_AVAILABLE:
                            models["XGBoost"] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
                        
                        scoring_metric = 'accuracy'
                    else:
                        models = {
                            "Random Forest": RandomForestRegressor(random_state=42),
                            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                            "Linear Regression": LinearRegression(),
                            "SVR": SVR(),
                            "Neural Network": MLPRegressor(random_state=42, max_iter=500)
                        }
                        
                        if XGBOOST_AVAILABLE:
                            models["XGBoost"] = xgb.XGBRegressor(random_state=42)
                        
                        scoring_metric = 'r2'
                    
                    # Limit to max_models
                    models = dict(list(models.items())[:max_models])
                    
                    # Train and evaluate models
                    automl_results = {}
                    
                    for i, (name, model) in enumerate(models.items()):
                        status_text.text(f"🤖 Training {name}...")
                        
                        try:
                            # Hyperparameter tuning if enabled
                            if hyperparameter_tuning:
                                status_text.text(f"⚙️ Tuning {name} hyperparameters...")
                                
                                if "Random Forest" in name:
                                    param_grid = {
                                        'n_estimators': [50, 100, 200],
                                        'max_depth': [None, 10, 20]
                                    }
                                elif "Gradient Boosting" in name:
                                    param_grid = {
                                        'n_estimators': [50, 100, 200],
                                        'learning_rate': [0.01, 0.1, 0.2]
                                    }
                                else:
                                    param_grid = {}  # Use default parameters for other models
                                
                                if param_grid:
                                    grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scoring_metric, n_jobs=-1)
                                    grid_search.fit(X_train, y_train)
                                    model = grid_search.best_estimator_
                                else:
                                    model.fit(X_train, y_train)
                            else:
                                model.fit(X_train, y_train)
                            
                            # Predictions and evaluation
                            y_pred = model.predict(X_test)
                            
                            if auto_problem_type == "Classification":
                                from sklearn.metrics import accuracy_score, classification_report
                                score = accuracy_score(y_test, y_pred)
                                
                                # Cross-validation
                                try:
                                    cv_scores = cross_val_score(model, X, y, cv=cross_validation_folds, scoring=scoring_metric)
                                    cv_mean = cv_scores.mean()
                                    cv_std = cv_scores.std()
                                except:
                                    cv_mean = score
                                    cv_std = 0.0
                                
                                automl_results[name] = {
                                    'accuracy': score,
                                    'cv_mean': cv_mean,
                                    'cv_std': cv_std,
                                    'model': model
                                }
                            else:
                                from sklearn.metrics import r2_score, mean_squared_error
                                score = r2_score(y_test, y_pred)
                                mse = mean_squared_error(y_test, y_pred)
                                
                                # Cross-validation
                                try:
                                    cv_scores = cross_val_score(model, X, y, cv=cross_validation_folds, scoring=scoring_metric)
                                    cv_mean = cv_scores.mean()
                                    cv_std = cv_scores.std()
                                except:
                                    cv_mean = score
                                    cv_std = 0.0
                                
                                automl_results[name] = {
                                    'r2_score': score,
                                    'mse': mse,
                                    'cv_mean': cv_mean,
                                    'cv_std': cv_std,
                                    'model': model
                                }
                        
                        except Exception as e:
                            st.warning(f"⚠️ {name} failed: {str(e)[:50]}...")
                            continue
                        
                        progress_bar.progress(0.3 + (i + 1) * 0.6 / len(models))
                    
                    progress_bar.progress(1.0)
                    status_text.empty()
                    
                    if automl_results:
                        st.success("✅ AutoML Pipeline completed successfully!")
                        
                        # Store results
                        st.session_state.automl_results = automl_results
                        st.session_state.automl_problem_type = auto_problem_type
                        
                        # Display results
                        results_df = pd.DataFrame({k: {key: val for key, val in v.items() if key != 'model'} for k, v in automl_results.items()}).T
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Best model recommendation
                        if auto_problem_type == "Classification":
                            best_model_name = max(automl_results.items(), key=lambda x: x[1]['accuracy'])[0]
                            best_score = automl_results[best_model_name]['accuracy']
                            st.success(f"🏆 **Best Model**: {best_model_name} (Accuracy: {best_score:.3f})")
                        else:
                            best_model_name = max(automl_results.items(), key=lambda x: x[1]['r2_score'])[0]
                            best_score = automl_results[best_model_name]['r2_score']
                            st.success(f"🏆 **Best Model**: {best_model_name} (R²: {best_score:.3f})")
                        
                        # Model comparison chart
                        if auto_problem_type == "Classification":
                            fig = px.bar(
                                x=list(automl_results.keys()),
                                y=[automl_results[model]['accuracy'] for model in automl_results.keys()],
                                title="AutoML Model Accuracy Comparison",
                                color=[automl_results[model]['accuracy'] for model in automl_results.keys()],
                                color_continuous_scale='viridis'
                            )
                        else:
                            fig = px.bar(
                                x=list(automl_results.keys()),
                                y=[automl_results[model]['r2_score'] for model in automl_results.keys()],
                                title="AutoML Model R² Score Comparison",
                                color=[automl_results[model]['r2_score'] for model in automl_results.keys()],
                                color_continuous_scale='viridis'
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.error("❌ No models were successfully trained")
                
                except Exception as e:
                    st.error(f"❌ AutoML Pipeline failed: {str(e)}")
    
    with tab2:
        st.subheader("⚙️ Custom Pipeline Configuration")
        
        st.markdown("""
        **Build your own ML pipeline** with custom configurations:
        """)
        
        # Custom pipeline builder would go here
        # This is a placeholder for future expansion
        st.info("🚧 Custom pipeline builder - Coming soon!")
        
        st.markdown("""
        **Planned Features:**
        - 🔧 Custom preprocessing steps
        - 🎯 Algorithm-specific parameter tuning
        - 🗗️ Pipeline templates and presets
        - 💾 Save and reuse custom pipelines
        """)
    
    with tab3:
        st.subheader("📈 AutoML Pipeline Results")
        
        if 'automl_results' in st.session_state and st.session_state.automl_results:
            automl_results = st.session_state.automl_results
            problem_type = st.session_state.get('automl_problem_type', 'Unknown')
            
            st.write(f"**Problem Type**: {problem_type}")
            st.write(f"**Models Trained**: {len(automl_results)}")
            
            # Detailed results table
            results_df = pd.DataFrame({k: {key: val for key, val in v.items() if key != 'model'} for k, v in automl_results.items()}).T
            st.dataframe(results_df, use_container_width=True)
            
            # Export options
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Export Results CSV"):
                    csv_data = results_df.to_csv(index=True)
                    st.download_button(
                        label="📈 Download AutoML Results",
                        data=csv_data,
                        file_name="automl_results.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("🗜️ Generate Report"):
                    # Generate a comprehensive report
                    report = f"""
# AutoML Pipeline Report

## Problem Type: {problem_type}
## Models Trained: {len(automl_results)}

## Results Summary:
{results_df.to_string()}

## Best Model Recommendation:
{max(automl_results.items(), key=lambda x: x[1].get('accuracy', x[1].get('r2_score', 0)))[0]}
                    """
                    
                    st.download_button(
                        label="📁 Download Report",
                        data=report,
                        file_name="automl_report.md",
                        mime="text/markdown"
                    )
            
            # Clear results
            if st.button("🗑️ Clear AutoML Results", type="secondary"):
                if 'automl_results' in st.session_state:
                    del st.session_state.automl_results
                if 'automl_problem_type' in st.session_state:
                    del st.session_state.automl_problem_type
                st.success("✅ AutoML results cleared!")
                st.rerun()
        
        else:
            st.info("🤖 No AutoML results available. Run AutoML pipeline first.")



    st.title("ℹ️ Panduan Algoritma")
    
    tab1, tab2, tab3 = st.tabs(["📚 Overview", "📊 Supervised", "🔍 Unsupervised"])
    
    with tab1:
        st.markdown("""
        ## Machine Learning Categories
        
        ### 📊 Supervised Learning
        - **Data:** Has target/labels
        - **Goal:** Predict outcomes
        - **Examples:** Email spam, price prediction
        - **When to use:** You have historical data with known results
        
        ### 🔍 Unsupervised Learning  
        - **Data:** No target/labels
        - **Goal:** Find hidden patterns
        - **Examples:** Customer segmentation, anomaly detection
        - **When to use:** Explore data structure, group similar items
        
        ### 🎮 Reinforcement Learning
        - **Data:** Actions and rewards
        - **Goal:** Learn optimal decisions
        - **Examples:** Game AI, recommendation systems
        - **When to use:** Sequential decision making problems
        """)
    
    with tab2:
        st.markdown("""
        ## Supervised Learning Guide
        
        ### Classification vs Regression
        
        **Classification** (Predict categories):
        - Email spam/not spam
        - Disease diagnosis
        - Customer will churn/stay
        - **Metrics:** Accuracy, Precision, Recall
        
        **Regression** (Predict numbers):
        - House prices
        - Stock prices
        - Temperature forecast
        - **Metrics:** R², MSE, MAE
        
        ### Algorithm Selection
        
        | Situation | Recommended Algorithm |
        |-----------|----------------------|
        | Small dataset | Linear/Logistic Regression |
        | Large dataset | Random Forest, XGBoost |
        | Need interpretability | Linear/Logistic Regression |
        | Mixed data types | Random Forest |
        | High accuracy priority | Ensemble methods |
        """)
    
    with tab3:
        st.markdown("""
        ## Unsupervised Learning Guide
        
        ### Clustering Algorithms
        
        **K-Means:**
        - **Best for:** Well-separated, spherical clusters
        - **Pros:** Simple, fast
        - **Cons:** Need to specify number of clusters
        - **Use case:** Customer segmentation
        
        **DBSCAN:**
        - **Best for:** Irregular shapes, noisy data
        - **Pros:** Finds arbitrary shapes, handles outliers
        - **Cons:** Sensitive to parameters
        - **Use case:** Anomaly detection
        
        ### How to Choose Clusters
        
        1. **Domain knowledge:** Business requirements
        2. **Elbow method:** Plot within-cluster sum of squares
        3. **Silhouette analysis:** Measure cluster quality
        4. **Trial and error:** Test different numbers
        """)

def show_statistical_tests():
    st.title("📈 Statistical Tests")
    
    if st.session_state.data is None:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu!")
        return
    
    # Use the most processed version of data available
    if st.session_state.processed_data is not None:
        data = st.session_state.processed_data
        st.info("📊 Menggunakan data yang sudah diproses")
    elif st.session_state.cleaned_data is not None:
        data = st.session_state.cleaned_data
        st.info("🧹 Menggunakan data yang sudah dibersihkan")
    else:
        data = st.session_state.data
        st.info("📂 Menggunakan data original")
    
    tab1, tab2, tab3 = st.tabs(["📊 Normality Tests", "🔗 Correlation Tests", "📎 Comparative Tests"])
    
    with tab1:
        st.subheader("📊 Normality Tests")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            col_for_normality = st.selectbox("Pilih kolom untuk test normalitas:", numeric_cols)
            
            if col_for_normality:
                sample_data = data[col_for_normality].dropna()
                
                if len(sample_data) > 5000:
                    sample_data = sample_data.sample(5000)
                    st.info("📊 Menggunakan sample 5000 data untuk test")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Shapiro-Wilk Test
                    try:
                        shapiro_stat, shapiro_p = stats.shapiro(sample_data)
                        st.write("**Shapiro-Wilk Test:**")
                        st.write(f"Statistic: {shapiro_stat:.4f}")
                        st.write(f"P-value: {shapiro_p:.4f}")
                        
                        if shapiro_p > 0.05:
                            st.success("✅ Data berdistribusi normal (p > 0.05)")
                        else:
                            st.warning("⚠️ Data tidak berdistribusi normal (p ≤ 0.05)")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                
                with col2:
                    # Kolmogorov-Smirnov Test
                    try:
                        ks_stat, ks_p = stats.kstest(sample_data, 'norm', 
                                                    args=(sample_data.mean(), sample_data.std()))
                        st.write("**Kolmogorov-Smirnov Test:**")
                        st.write(f"Statistic: {ks_stat:.4f}")
                        st.write(f"P-value: {ks_p:.4f}")
                        
                        if ks_p > 0.05:
                            st.success("✅ Data berdistribusi normal (p > 0.05)")
                        else:
                            st.warning("⚠️ Data tidak berdistribusi normal (p ≤ 0.05)")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.info("📊 Tidak ada kolom numerik untuk test normalitas")
    
    with tab2:
        st.subheader("🔗 Correlation Tests")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            st.markdown("### Correlation Significance Test")
            
            col1, col2 = st.columns(2)
            with col1:
                var1 = st.selectbox("Variable 1:", numeric_cols, key="corr_var1")
            with col2:
                var2 = st.selectbox("Variable 2:", [col for col in numeric_cols if col != var1], key="corr_var2")
            
            if var1 and var2:
                clean_data = data[[var1, var2]].dropna()
                
                if len(clean_data) > 3:
                    try:
                        # Pearson correlation
                        pearson_r, pearson_p = stats.pearsonr(clean_data[var1], clean_data[var2])
                        
                        # Spearman correlation
                        spearman_r, spearman_p = stats.spearmanr(clean_data[var1], clean_data[var2])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Pearson Correlation:**")
                            st.write(f"Correlation: {pearson_r:.4f}")
                            st.write(f"P-value: {pearson_p:.4f}")
                            
                            if pearson_p < 0.05:
                                st.success("✅ Korelasi signifikan (p < 0.05)")
                            else:
                                st.info("ℹ️ Korelasi tidak signifikan (p ≥ 0.05)")
                        
                        with col2:
                            st.write("**Spearman Correlation:**")
                            st.write(f"Correlation: {spearman_r:.4f}")
                            st.write(f"P-value: {spearman_p:.4f}")
                            
                            if spearman_p < 0.05:
                                st.success("✅ Korelasi signifikan (p < 0.05)")
                            else:
                                st.info("ℹ️ Korelasi tidak signifikan (p ≥ 0.05)")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Chi-square test for categorical variables
        if len(categorical_cols) >= 2:
            st.markdown("### Chi-Square Test (Categorical Association)")
            
            col1, col2 = st.columns(2)
            with col1:
                cat_var1 = st.selectbox("Categorical Variable 1:", categorical_cols, key="chi_var1")
            with col2:
                cat_var2 = st.selectbox("Categorical Variable 2:", [col for col in categorical_cols if col != cat_var1], key="chi_var2")
            
            if cat_var1 and cat_var2:
                try:
                    contingency_table = pd.crosstab(data[cat_var1], data[cat_var2])
                    chi2_stat, chi2_p, chi2_dof, chi2_expected = stats.chi2_contingency(contingency_table)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Contingency Table:**")
                        st.dataframe(contingency_table)
                    
                    with col2:
                        st.write("**Chi-Square Test Results:**")
                        st.write(f"Chi-square: {chi2_stat:.4f}")
                        st.write(f"P-value: {chi2_p:.4f}")
                        st.write(f"Degrees of freedom: {chi2_dof}")
                        
                        if chi2_p < 0.05:
                            st.success("✅ Ada hubungan signifikan (p < 0.05)")
                        else:
                            st.info("ℹ️ Tidak ada hubungan signifikan (p ≥ 0.05)")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab3:
        st.subheader("📎 Comparative Tests")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if numeric_cols and categorical_cols:
            st.markdown("### T-Test / ANOVA")
            
            numeric_var = st.selectbox("Numeric Variable:", numeric_cols, key="ttest_num")
            categorical_var = st.selectbox("Grouping Variable:", categorical_cols, key="ttest_cat")
            
            if numeric_var and categorical_var:
                try:
                    groups = data.groupby(categorical_var)[numeric_var].apply(lambda x: x.dropna().values)
                    group_names = groups.index.tolist()
                    group_data = [group for group in groups.values if len(group) > 1]
                    
                    if len(group_data) == 2:
                        # T-test for two groups
                        ttest_stat, ttest_p = stats.ttest_ind(group_data[0], group_data[1])
                        
                        st.write("**Independent T-Test Results:**")
                        st.write(f"T-statistic: {ttest_stat:.4f}")
                        st.write(f"P-value: {ttest_p:.4f}")
                        
                        if ttest_p < 0.05:
                            st.warning("⚠️ Rata-rata berbeda signifikan (p < 0.05)")
                        else:
                            st.success("✅ Rata-rata tidak berbeda signifikan (p ≥ 0.05)")
                    
                    elif len(group_data) > 2:
                        # ANOVA for multiple groups
                        anova_stat, anova_p = stats.f_oneway(*group_data)
                        
                        st.write("**One-Way ANOVA Results:**")
                        st.write(f"F-statistic: {anova_stat:.4f}")
                        st.write(f"P-value: {anova_p:.4f}")
                        
                        if anova_p < 0.05:
                            st.warning("⚠️ Ada perbedaan rata-rata signifikan (p < 0.05)")
                        else:
                            st.success("✅ Tidak ada perbedaan rata-rata signifikan (p ≥ 0.05)")
                    
                    # Show descriptive statistics by group
                    st.markdown("### Descriptive Statistics by Group")
                    desc_stats = data.groupby(categorical_var)[numeric_var].describe()
                    st.dataframe(desc_stats)
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

def show_model_comparison():
    st.title("🏆 Model Comparison Dashboard")
    
    if not st.session_state.model_results:
        st.info("📊 Belum ada hasil model. Jalankan model di 'Auto Data Mining' terlebih dahulu.")
        return
    
    st.subheader("📈 Model Performance Comparison")
    
    # Display all model results
    results_df = pd.DataFrame(st.session_state.model_results).T
    
    if not results_df.empty:
        st.dataframe(results_df, use_container_width=True)
        
        # Create comparison charts
        if 'accuracy' in results_df.columns:
            fig = px.bar(x=results_df.index, y=results_df['accuracy'], 
                        title="Model Accuracy Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        if 'r2_score' in results_df.columns:
            fig = px.bar(x=results_df.index, y=results_df['r2_score'], 
                        title="Model R² Score Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        # Best model recommendation
        if 'accuracy' in results_df.columns:
            best_model = results_df['accuracy'].idxmax()
            st.success(f"🏆 Best Classification Model: {best_model} (Accuracy: {results_df.loc[best_model, 'accuracy']:.3f})")
        
        if 'r2_score' in results_df.columns:
            best_model = results_df['r2_score'].idxmax()
            st.success(f"🏆 Best Regression Model: {best_model} (R²: {results_df.loc[best_model, 'r2_score']:.3f})")
    
    # Clear results button
    if st.button("🗑️ Clear All Results"):
        st.session_state.model_results = {}
        st.rerun()

def show_session_management():
    st.title("💾 Session Management")
    
    tab1, tab2, tab3 = st.tabs(["💾 Save Session", "📁 Load Session", "📅 Analysis History"])
    
    with tab1:
        st.subheader("💾 Save Current Session")
        
        if st.session_state.data is not None:
            session_name = st.text_input("Session Name:", value=f"session_{time.strftime('%Y%m%d_%H%M%S')}")
            
            session_data = {
                'original_data': st.session_state.data.to_dict() if st.session_state.data is not None else None,
                'cleaned_data': st.session_state.cleaned_data.to_dict() if st.session_state.cleaned_data is not None else None,
                'processed_data': st.session_state.processed_data.to_dict() if st.session_state.processed_data is not None else None,
                'model_results': st.session_state.model_results,
                'analysis_history': st.session_state.analysis_history,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            session_json = json.dumps(session_data, indent=2, default=str)
            
            st.download_button(
                label="📥 Download Session",
                data=session_json,
                file_name=f"{session_name}.json",
                mime="application/json",
                type="primary"
            )
            
            st.info("📊 Session includes: original data, cleaned data, processed data, model results, and analysis history.")
        else:
            st.warning("⚠️ No data to save. Please upload data first.")
    
    with tab2:
        st.subheader("📁 Load Session")
        
        uploaded_session = st.file_uploader(
            "Upload Session File:",
            type=['json'],
            help="Load a previously saved session"
        )
        
        if uploaded_session is not None:
            try:
                session_data = json.load(uploaded_session)
                
                # Restore data
                if session_data.get('original_data'):
                    st.session_state.data = pd.DataFrame(session_data['original_data'])
                if session_data.get('cleaned_data'):
                    st.session_state.cleaned_data = pd.DataFrame(session_data['cleaned_data'])
                if session_data.get('processed_data'):
                    st.session_state.processed_data = pd.DataFrame(session_data['processed_data'])
                if session_data.get('model_results'):
                    st.session_state.model_results = session_data['model_results']
                if session_data.get('analysis_history'):
                    st.session_state.analysis_history = session_data['analysis_history']
                
                st.success("✅ Session loaded successfully!")
                st.info(f"Session timestamp: {session_data.get('timestamp', 'Unknown')}")
                
                if st.session_state.data is not None:
                    st.write(f"**Data shape:** {st.session_state.data.shape}")
                    st.dataframe(st.session_state.data.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error loading session: {str(e)}")
    
    with tab3:
        st.subheader("📅 Analysis History")
        
        if st.session_state.analysis_history:
            history_df = pd.DataFrame(st.session_state.analysis_history)
            st.dataframe(history_df, use_container_width=True)
            
            if st.button("🗑️ Clear History"):
                st.session_state.analysis_history = []
                st.rerun()
        else:
            st.info("📊 No analysis history available.")


def show_dashboard_builder():
    st.title("📋 Interactive Dashboard Builder")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload dataset first!")
        return
    
    st.info("🚧 Dashboard Builder - Advanced feature coming soon!")
    st.markdown("""
    **Planned Features:**
    - 🎨 Drag-and-drop chart builder
    - 📈 Interactive KPI widgets
    - 🗟️ Custom dashboard templates
    - 💾 Export dashboards as HTML/PDF
    """)


def show_data_pipeline_automation():
    st.title("⚙️ Data Pipeline Automation")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload dataset first!")
        return
    
    st.info("🚧 Pipeline Automation - Advanced feature coming soon!")
    st.markdown("""
    **Planned Features:**
    - 🔄 Save and replay data cleaning workflows
    - 📁 Batch processing for multiple files
    - ⏰ Scheduled data processing tasks
    - 🗟️ Template-based workflows
    """)


def show_time_series_analysis():
    st.title("📅 Time Series Analysis")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload dataset first!")
        return
    
    st.info("🚧 Time Series Analysis - Advanced feature coming soon!")
    st.markdown("""
    **Planned Features:**
    - 📈 Trend and seasonality decomposition
    - 🔮 ARIMA, Prophet, LSTM forecasting
    - ⚡ Anomaly detection in time series
    - 📉 Interactive time series visualizations
    """)


def show_guidance():
    st.title("ℹ️ Algorithm Guide")
    
    tab1, tab2, tab3 = st.tabs(["📚 Overview", "📊 Supervised", "🔍 Unsupervised"])
    
    with tab1:
        st.markdown("""
        ## Machine Learning Categories
        
        ### 📊 Supervised Learning
        - **Data:** Has target/labels
        - **Goal:** Predict outcomes
        - **Examples:** Email spam, price prediction
        - **When to use:** You have historical data with known results
        
        ### 🔍 Unsupervised Learning  
        - **Data:** No target/labels
        - **Goal:** Find hidden patterns
        - **Examples:** Customer segmentation, anomaly detection
        - **When to use:** Explore data structure, group similar items
        
        ### 🎮 Reinforcement Learning
        - **Data:** Actions and rewards
        - **Goal:** Learn optimal decisions
        - **Examples:** Game AI, recommendation systems
        - **When to use:** Sequential decision making problems
        """)
    
    with tab2:
        st.markdown("""
        ## Supervised Learning Guide
        
        ### Classification vs Regression
        
        **Classification** (Predict categories):
        - Email spam/not spam
        - Disease diagnosis
        - Customer will churn/stay
        - **Metrics:** Accuracy, Precision, Recall
        
        **Regression** (Predict numbers):
        - House prices
        - Stock prices
        - Temperature forecast
        - **Metrics:** R², MSE, MAE
        
        ### Algorithm Selection
        
        | Situation | Recommended Algorithm |
        |-----------|----------------------|
        | Small dataset | Linear/Logistic Regression |
        | Large dataset | Random Forest, XGBoost |
        | Need interpretability | Linear/Logistic Regression |
        | Mixed data types | Random Forest |
        | High accuracy priority | Ensemble methods |
        """)
    
    with tab3:
        st.markdown("""
        ## Unsupervised Learning Guide
        
        ### Clustering Algorithms
        
        **K-Means:**
        - **Best for:** Well-separated, spherical clusters
        - **Pros:** Simple, fast
        - **Cons:** Need to specify number of clusters
        - **Use case:** Customer segmentation
        
        **DBSCAN:**
        - **Best for:** Irregular shapes, noisy data
        - **Pros:** Finds arbitrary shapes, handles outliers
        - **Cons:** Sensitive to parameters
        - **Use case:** Anomaly detection
        
        ### How to Choose Clusters
        
        1. **Domain knowledge:** Business requirements
        2. **Elbow method:** Plot within-cluster sum of squares
        3. **Silhouette analysis:** Measure cluster quality
        4. **Trial and error:** Test different numbers
        """)



    st.title("🧪 A/B Testing Framework")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload dataset first!")
        return
    
    st.info("🚧 A/B Testing Framework - Advanced feature coming soon!")
    st.markdown("""
    **Planned Features:**
    - 📊 Sample size calculators
    - 📍 Statistical significance testing
    - 📈 Conversion rate analysis
    - 📁 Experiment result interpretation
    """)

if __name__ == "__main__":
    main()