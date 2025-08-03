# detecting-fraud-mobile-wallets

ML-based fraud detection system for Pakistani mobile wallet transactions

# Mobile Wallet Fraud Detection

This project uses machine learning to detect fraudulent transactions in mobile wallets. It includes a trained model, Streamlit web app, and preprocessing pipeline to help flag suspicious activity in real-time.

## Features

- Detects fraud using a trained Random Forest classifier
- Cleaned and preprocessed real-world transaction data
- Web interface built with Streamlit
- Scalable and easy to extend

## Machine Learning Model

- **Algorithm:** Random Forest Classifier
- **Features:** PCA-based features (`V1` to `V28`), `Amount`, `Time`
- **Target:** Binary classification (fraud vs. not fraud)

## Tech Stack

- Python
- Scikit-learn
- Streamlit
- Pandas, NumPy
- Jupyter Notebooks

## Folder Structure

``plaintext
notebooks/
│
├── 01_EDA.ipynb # Exploratory Data Analysis
├── 02_Model_Training.ipynb # Feature engineering, SMOTE, and model training
├── app.py # Streamlit deployment script
├── rf_model_smote.pkl # Trained model
├── scaler.pkl # Feature scaler
├── creditcard.csv # Original dataset
├── wallet_fraud_readable.csv # Preprocessed readable format
├── X_encoded.csv, y.csv # Final feature-label datasets
├── requirements.txt # Dependencies

## Model

Model Used: Random Forest Classifier
Sampling: SMOTE (Synthetic Minority Oversampling)
Scaler: StandardScaler
The model predicts whether a transaction is fraudulent or genuine based on anonymized features (V1–V28), Amount, and Time.

## Features

V1 to V28: PCA-transformed features for security
Time: Time since the first transaction
Amount: Transaction amount
Operator:Jazzcash,Easypaisa,SadaPay
Region:Punjab,sindh,blochistan,KPK
Txn_Type:bill payment,money transfer,mobile load,bank transfer
Class: 1 for Fraud, 0 for Legit

## Model Performance

Accuracy: 99%
F1 Score: 87%
Balanced Dataset via SMOTE

## Notebooks

EDA: Visualizations, feature distributions, class imbalance checks
Model Training: Data cleaning → oversampling → model training → evaluation
Deployment App: Upload CSV → Predict fraud probabilities → Display fraud count

## License

MIT

## Contact

For queries or collaborations: moizzahr@gmail.com
