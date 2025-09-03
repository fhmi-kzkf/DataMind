# 🧠 DataMind - Comprehensive Data Analysis & Machine Learning Platform

**DataMind** - A comprehensive web-based data analysis & ML platform built with Streamlit. Upload CSV/Excel files, perform advanced EDA, clean data, create visualizations, train multiple ML models, and compare results. Features AutoML pipeline, statistical tests, PCA analysis, session management & interactive dashboards. Perfect for students, researchers & data professionals. No coding required!

## 🌟 Platform Overview

DataMind transforms complex data science workflows into intuitive, guided experiences. The platform features a clean, organized navigation system with six logical categories that follow natural data analysis workflows: Home & Upload, Data Processing, Analysis & Visualization, Machine Learning, Advanced Tools, and Management & Help.

The platform excels in intelligent automation, featuring advanced AutoML capabilities that automatically detect problem types (classification vs regression), perform feature selection, and compare multiple machine learning algorithms including Random Forest, XGBoost, Neural Networks, and Support Vector Machines. Users can upload CSV or Excel files and immediately begin exploring their data through interactive visualizations, statistical tests, and comprehensive exploratory data analysis.

Key strengths include robust data cleaning tools with typo correction, missing value handling, and outlier detection; advanced visualization capabilities including PCA analysis and 3D plotting; and comprehensive statistical testing modules. The platform provides sample datasets with realistic noise for testing all features, ensuring users can immediately experience the full capabilities.

DataMind bridges the gap between technical complexity and practical usability, offering professional-grade machine learning tools through an intuitive interface. Whether you're a researcher analyzing experimental data, a business analyst exploring market trends, or a student learning data science concepts, DataMind provides the tools and guidance needed to extract meaningful insights from your data efficiently and effectively.

The platform's session management system allows users to save and restore their work, while the model comparison dashboard provides clear performance metrics and recommendations. With support for supervised learning, unsupervised clustering, and even reinforcement learning simulations, DataMind serves as a comprehensive solution for modern data analysis challenges.

---

## ✨ Fitur Utama (Enhanced)

### 📂 **Data Upload & Management**
- **Multi-format Support**: CSV (.csv), Excel (.xlsx, .xls)
- **Smart Encoding Detection**: Automatic encoding detection for CSV files
- **Excel Sheet Selection**: Choose specific sheets from multi-sheet Excel files
- **Enhanced File Info**: Memory usage, detailed data type analysis
- **Analysis History Tracking**: Track all data operations and changes

### 🔍 **Enhanced EDA & Data Cleaning**
- **Advanced Missing Value Handling**: Forward fill, backward fill, custom methods
- **Column Deletion**: Multi-select interface for removing unwanted columns
- **Intelligent Typo Correction**: 
  - Individual value replacement with form-based interface
  - Bulk operations (space cleaning, capitalization standardization)
  - Real-time preview of changes
- **Enhanced Outlier Detection**: Customizable thresholds, capping options
- **Improved Data Quality Metrics**: Comprehensive data profiling

### 🛠️ **Advanced Feature Engineering**
- **Enhanced Encoding**: Label encoding, One-hot encoding with smart limits
- **Advanced Scaling**: StandardScaler, MinMaxScaler with before/after comparison
- **Feature Transformation**: Automatic feature creation and validation
- **Data Pipeline Tracking**: Monitor all transformations applied

### 📊 **Advanced Interactive Visualizations**
- **Basic Plots**: Histogram, Box Plot, Bar Chart, Scatter Plot, Violin Plot
- **Relationship Analysis**: Enhanced correlation analysis with significance testing
- **Advanced Visualizations**:
  - **PCA Visualization**: 2D/3D principal component analysis with variance explanation
  - **Distribution Comparison**: Multi-group distribution analysis with marginal plots
  - **3D Scatter Plots**: Interactive 3D visualizations with color coding

### 📈 **Statistical Testing Suite**
- **Normality Tests**: Shapiro-Wilk, Kolmogorov-Smirnov with Q-Q plots
- **Correlation Tests**: Pearson, Spearman with significance testing
- **Association Tests**: Chi-square test for categorical variables
- **Comparative Tests**: T-tests, ANOVA with descriptive statistics
- **Automated Interpretation**: Clear explanations of statistical results

### 🤖 **Advanced Machine Learning**
- **Enhanced Supervised Learning**:
  - **Classification**: Logistic Regression, Random Forest, Gradient Boosting, SVM, Neural Networks
  - **Advanced Models**: XGBoost, LightGBM (when available)
  - **Cross-validation**: 3-fold CV with mean and standard deviation reporting
  - **Feature Importance**: Automatic feature importance analysis
  - **Progress Tracking**: Real-time training progress with status updates
  
- **Enhanced Unsupervised Learning**: 
  - K-Means, DBSCAN clustering with advanced parameters
  - Cluster visualization and distribution analysis
  
- **Reinforcement Learning**: Multi-armed bandit simulation with visualization

### 🏆 **Model Comparison Dashboard**
- **Comprehensive Comparison**: Side-by-side model performance comparison
- **Interactive Charts**: Bar charts comparing accuracy, R², and other metrics
- **Best Model Recommendation**: Automatic identification of top-performing models
- **Performance Tracking**: Store and compare results across sessions
- **Model History**: Track all trained models with timestamps

### 💾 **Session Management**
- **Save Sessions**: Export complete analysis sessions as JSON
- **Load Sessions**: Import and restore previous work
- **Analysis History**: Detailed log of all operations performed
- **Data Versioning**: Track original, cleaned, and processed data versions
- **Reproducible Analysis**: Restore exact state of previous sessions

### 📱 **Enhanced UI/UX**
- **Modern Interface**: Clean design with blue (#2563EB) and orange (#F97316) theme
- **Responsive Layout**: Optimized for different screen sizes
- **Progress Indicators**: Real-time feedback for long-running operations
- **Smart Navigation**: Intuitive menu system with clear categorization
- **Error Handling**: Comprehensive error messages with helpful tips

## 🚀 Cara Menjalankan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi

```bash
streamlit run app.py
```

### 3. Buka Browser

Aplikasi akan terbuka di `http://localhost:8501`

## 🎯 Enhanced Workflow

1. **📂 Upload Data** - Support CSV/Excel dengan deteksi encoding otomatis
2. **🔍 Explore & Clean** - EDA mendalam dengan statistical tests
3. **🛠️ Engineer Features** - Advanced preprocessing dengan tracking
4. **📊 Visualize** - Create advanced plots termasuk PCA dan 3D scatter
5. **📈 Test Statistics** - Comprehensive statistical testing suite
6. **🤖 Train Models** - Advanced ML dengan multiple algorithms
7. **🏆 Compare Results** - Dashboard perbandingan model
8. **💾 Save Session** - Export/import untuk reproducibility

## 🛠️ Enhanced Technology Stack

- **Framework**: Streamlit (enhanced with advanced features)
- **Data Processing**: Pandas, NumPy (with advanced operations)
- **Visualization**: Plotly (with 3D and advanced plots), Matplotlib, Seaborn
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Statistics**: SciPy, statsmodels
- **File Handling**: CSV, Excel (multi-format support)
- **Session Management**: JSON-based serialization

## 📊 Supported File Formats

| Format | Extension | Features |
|--------|-----------|----------|
| CSV | .csv | Multi-encoding support, automatic detection |
| Excel | .xlsx, .xls | Multi-sheet selection, advanced parsing |

## 🤖 Machine Learning Algorithms

### Classification
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM)
- Multi-layer Perceptron (Neural Network)
- XGBoost Classifier (optional)
- LightGBM Classifier (optional)

### Regression
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regression (SVR)
- Multi-layer Perceptron Regressor
- XGBoost Regressor (optional)
- LightGBM Regressor (optional)

### Clustering
- K-Means Clustering
- DBSCAN

## 📈 Statistical Tests Available

- **Normality**: Shapiro-Wilk, Kolmogorov-Smirnov
- **Correlation**: Pearson, Spearman
- **Association**: Chi-square test
- **Comparison**: Independent t-test, One-way ANOVA
- **Distribution**: Two-sample Kolmogorov-Smirnov, Mann-Whitney U

## 📋 Enhanced Requirements

- Python 3.8+
- Memory: 4GB+ (recommended for large datasets)
- Storage: 1GB untuk dependencies
- Browser: Modern browser dengan JavaScript support

## 🎓 Target Pengguna (Expanded)

- **Students & Researchers**: Academic analysis dengan statistical rigor
- **Data Scientists**: Professional-grade analysis tools
- **Business Analysts**: Advanced insights dengan easy-to-use interface
- **Machine Learning Engineers**: Model comparison dan evaluation
- **Statisticians**: Comprehensive statistical testing suite

## 🆘 Enhanced Troubleshooting

### Installation Issues
```bash
# Update pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

# For XGBoost/LightGBM issues
pip install xgboost lightgbm
```

### Memory Issues
- Gunakan dataset < 50MB untuk performa optimal
- Enable data sampling untuk dataset besar
- Monitor memory usage di dashboard

### Performance Optimization
- Close unused browser tabs
- Use Chrome/Firefox untuk performa terbaik
- Enable hardware acceleration jika tersedia

## 🚀 Advanced Features

### Session Management
- **Auto-save**: Automatic saving setiap major operation
- **Version Control**: Track changes dengan timestamps
- **Export Options**: JSON format untuk portability
- **Collaboration**: Share sessions dengan team members

### Model Comparison
- **Cross-validation**: 3-fold CV untuk robust evaluation
- **Feature Importance**: Automatic feature ranking
- **Performance Metrics**: Comprehensive evaluation metrics
- **Visual Comparison**: Interactive charts dan graphs

### Advanced Visualization
- **PCA Analysis**: Dimensionality reduction dengan explained variance
- **3D Plotting**: Interactive 3D scatter plots
- **Statistical Plots**: Q-Q plots, distribution comparisons
- **Network Visualization**: Correlation networks

## 📈 Performance Metrics

- **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Regression**: R², MSE, MAE, RMSE
- **Clustering**: Silhouette Score, Inertia
- **Cross-validation**: Mean ± Standard Deviation

## 🔮 Future Enhancements (Roadmap)

- [ ] **Deep Learning**: TensorFlow/PyTorch integration
- [ ] **Time Series**: Advanced forecasting models
- [ ] **NLP Support**: Text analysis capabilities
- [ ] **AutoML**: Automated model selection dan hyperparameter tuning
- [ ] **Cloud Integration**: Deploy models ke cloud platforms
- [ ] **Real-time Data**: Streaming data analysis
- [ ] **Collaborative Features**: Multi-user support
- [ ] **API Integration**: REST API untuk external systems

## 📄 License

MIT License - Feel free to use and modify for educational and commercial purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### Development Setup
```bash
git clone <repository>
cd datamind
pip install -r requirements.txt
streamlit run app.py
```

---

**DataMind v2.0 (Enhanced)** - Professional Data Analysis & Machine Learning Platform! 🚀✨