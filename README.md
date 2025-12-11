# 🤖 Telecom Customer Churn Prediction

A machine learning-powered web application to predict customer churn for telecom companies, built with Streamlit and scikit-learn.

## 📋 Overview

This application helps telecom companies identify customers who are likely to churn (leave the service) based on various customer attributes such as demographics, services used, contract details, and billing information.

## ✨ Features

- **Interactive Web Interface**: Clean, user-friendly Streamlit interface with white background theme
- **Real-time Predictions**: Instant churn probability predictions using a trained machine learning model
- **Visual Analytics**: Interactive gauge chart showing churn risk levels
- **Actionable Insights**: Recommended actions based on risk level (High/Moderate/Low)
- **Comprehensive Input Fields**:
  - Demographics (Gender, Senior Citizen, Partner, Dependents)
  - Services & Tenure (Internet Service, Online Security, Tech Support)
  - Contract & Billing (Contract Type, Payment Method, Charges)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/UsmanXTech/telco-customer-churn-analysis.git
   cd telco-customer-churn-analysis
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   - The app will automatically open in your default browser
   - Or navigate to `http://localhost:8501`

## 📦 Project Structure

```
telco-customer-churn-analysis/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── data/
│   ├── raw/
│   │   └── dataSet.csv            # Original dataset
│   └── processed/
│       └── processed_data.csv     # Cleaned and processed data
│
├── models/
│   └── churn_model.pkl            # Trained ML model
│
└── notebooks/
    ├── 01_data_understanding.ipynb # Data exploration notebook
    └── churn_model.pkl            # Model backup
```

## 🎯 Usage

1. **Enter Customer Details**: Fill in the customer profile information in the left panel

   - Demographics
   - Services & Tenure
   - Contract & Billing details

2. **Analyze**: Click the "🚀 Analyze Churn Risk" button

3. **View Results**:
   - See the churn probability percentage
   - View the risk level (Low/Moderate/High)
   - Get recommended actions for high-risk customers

## 📊 Model Information

- **Algorithm**: Machine Learning classification model (trained with scikit-learn)
- **Input Features**: 17+ customer attributes
- **Output**: Churn probability (0-100%)
- **Risk Categories**:
  - 🟢 **Low Risk** (0-40%): Loyal customer, no immediate action needed
  - 🟡 **Moderate Risk** (40-70%): Monitor usage patterns
  - 🔴 **High Risk** (70-100%): Immediate retention actions recommended

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn
- **Visualization**: Plotly
- **Model Serialization**: joblib

## 📈 Future Enhancements

- [ ] Add more ML models for comparison
- [ ] Include feature importance visualization
- [ ] Add batch prediction capability
- [ ] Export predictions to CSV
- [ ] Add customer segmentation analysis
- [ ] Deploy to cloud (Streamlit Cloud/Heroku)

## 👨‍💻 Author

**Usman**

- GitHub: [@UsmanXTech](https://github.com/UsmanXTech)

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📞 Support

If you have any questions or need help, please open an issue on GitHub.

---

⭐ **Star this repository if you find it helpful!**
