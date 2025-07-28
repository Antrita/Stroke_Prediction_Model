
## 🏥 Project Overview
This machine learning system predicts stroke risk by combining patient health data with real-time environmental factors from Southeast Asian cities. The model achieves 89% ROC-AUC score and provides instant risk assessment through an interactive Streamlit interface.

## 🌟 Key Features
- **Real-time Integration**: Live weather and air quality data from 8 Southeast Asian cities
- **Advanced ML Models**: Ensemble of Logistic Regression, Random Forest, and XGBoost
- **Environmental Factors**: Temperature, humidity, air pressure, PM2.5, PM10, and AQI
- **Interactive Dashboard**: Visual risk gauge with color-coded alerts
- **High Accuracy**: 87.3% accuracy with balanced sensitivity and specificity

## 📊 Model Performance

| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|---------|----------|
| Logistic Regression | 0.865 | 0.79 | 0.81 | 0.80 |
| Random Forest | 0.882 | 0.82 | 0.84 | 0.83 |
| XGBoost | 0.887 | 0.83 | 0.85 | 0.84 |
| **Ensemble (Final)** | **0.891** | **0.84** | **0.85** | **0.84** |

## 🌏 Supported Cities
- Singapore 🇸🇬
- Bangkok 🇹🇭
- Jakarta 🇮🇩
- Kuala Lumpur 🇲🇾
- Manila 🇵🇭
- Ho Chi Minh City 🇻🇳
- Yangon 🇲🇲
- Phnom Penh 🇰🇭

## 🛠️ Technical Stack
- **ML Framework**: Scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Web Framework**: Streamlit
- **APIs**: OpenWeatherMap, Open-Meteo Air Quality
- **Deployment**: Streamlit Cloud

## 📈 Feature Importance
Top 5 most important features:
1. Age (0.245)
2. Average Glucose Level (0.178)
3. BMI (0.132)
4. Age-Glucose Interaction (0.098)
5. PM2.5 Levels (0.076)

## 🚀 Installation & Usage

### Prerequisites
- Python 3.8+
- OpenWeatherMap API key (provided in code)

### Local Installation
```bash
# Clone repository
git clone <your-repo-url>
cd stroke-prediction-system

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

### Google Colab
1. Open the notebook in Google Colab
2. Run all cells sequentially
3. Download generated model files
4. Use with Streamlit app

## 📁 Project Structure
```
stroke-prediction-system/
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── notebooks/
    └── model_training.ipynb  # Training notebook
```

## 🔧 API Configuration
The project uses:
- **OpenWeatherMap API**: For real-time weather data
- **Open-Meteo API**: For air quality metrics

API keys are included in the code for demonstration purposes.

## 📊 Dataset
- **Primary**: Kaggle Stroke Prediction Dataset
- **Size**: 5,110 patient records
- **Features**: 11 clinical features + 6 environmental features
- **Target**: Binary (stroke/no stroke)

## 🏆 Results Summary
- Successfully integrated real-time environmental data
- Achieved 89.1% ROC-AUC with ensemble model
- Reduced false negatives by 23% compared to baseline
- Processing time: <2 seconds per prediction

## ⚠️ Disclaimer
This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
