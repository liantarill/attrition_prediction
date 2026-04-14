from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the trained SVM model package
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model_svm.pkl')
with open(MODEL_PATH, 'rb') as f:
    model_package = pickle.load(f)

# Extract components from the model package
svm_model = model_package['model']           # SVC classifier
scaler = model_package['scaler']             # StandardScaler
label_encoders = model_package['label_encoders']  # LabelEncoders for categorical features
feature_names = model_package['feature_names']     # List of feature names in order
model_metrics = model_package['metrics']           # Model performance metrics

# Load CSV for dataset info
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'final_attrition.csv')
df = pd.read_csv(CSV_PATH)

# Define feature columns and their types for the prediction form
CATEGORICAL_FEATURES = {
    'BusinessTravel': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
    'Department': ['Human Resources', 'Research & Development', 'Sales'],
    'EducationField': ['Human Resources', 'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree'],
    'Gender': ['Female', 'Male'],
    'JobRole': ['Healthcare Representative', 'Human Resources', 'Laboratory Technician',
                'Manager', 'Manufacturing Director', 'Research Director',
                'Research Scientist', 'Sales Executive', 'Sales Representative'],
    'MaritalStatus': ['Divorced', 'Married', 'Single'],
    'OverTime': ['No', 'Yes'],
}

NUMERIC_FEATURES = {
    'Age': {'min': 18, 'max': 60, 'default': 30},
    'DailyRate': {'min': 100, 'max': 1500, 'default': 800},
    'DistanceFromHome': {'min': 1, 'max': 29, 'default': 9},
    'Education': {'min': 1, 'max': 5, 'default': 3},
    'EnvironmentSatisfaction': {'min': 1, 'max': 4, 'default': 3},
    'HourlyRate': {'min': 30, 'max': 100, 'default': 66},
    'JobInvolvement': {'min': 1, 'max': 4, 'default': 3},
    'JobLevel': {'min': 1, 'max': 5, 'default': 2},
    'JobSatisfaction': {'min': 1, 'max': 4, 'default': 3},
    'MonthlyIncome': {'min': 1000, 'max': 20000, 'default': 5000},
    'MonthlyRate': {'min': 2000, 'max': 27000, 'default': 14000},
    'NumCompaniesWorked': {'min': 0, 'max': 9, 'default': 2},
    'PercentSalaryHike': {'min': 11, 'max': 25, 'default': 14},
    'PerformanceRating': {'min': 1, 'max': 4, 'default': 3},
    'RelationshipSatisfaction': {'min': 1, 'max': 4, 'default': 3},
    'StockOptionLevel': {'min': 0, 'max': 3, 'default': 1},
    'TotalWorkingYears': {'min': 0, 'max': 40, 'default': 10},
    'TrainingTimesLastYear': {'min': 0, 'max': 6, 'default': 3},
    'WorkLifeBalance': {'min': 1, 'max': 4, 'default': 3},
    'YearsAtCompany': {'min': 0, 'max': 40, 'default': 5},
    'YearsInCurrentRole': {'min': 0, 'max': 18, 'default': 3},
    'YearsSinceLastPromotion': {'min': 0, 'max': 15, 'default': 1},
    'YearsWithCurrManager': {'min': 0, 'max': 17, 'default': 3},
}

# Labels for satisfaction/education/etc. scales
SCALE_LABELS = {
    'Education': {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'},
    'EnvironmentSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
    'JobInvolvement': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
    'JobSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
    'PerformanceRating': {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'},
    'RelationshipSatisfaction': {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'},
    'WorkLifeBalance': {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'},
}

# Dataset summary stats
total_employees = len(df)
attrition_count = df['Attrition'].sum()
attrition_rate = round(attrition_count / total_employees * 100, 1)
avg_age = round(df['Age'].mean(), 1)
avg_income = round(df['MonthlyIncome'].mean(), 0)


@app.route('/')
def home():
    """Home page"""
    return render_template('home.html',
                           total_employees=total_employees,
                           attrition_count=int(attrition_count),
                           attrition_rate=attrition_rate,
                           avg_age=avg_age,
                           avg_income=int(avg_income))


@app.route('/dashboard')
def dashboard():
    """Dashboard page with embedded Looker Studio"""
    return render_template('dashboard.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    """Prediction page"""
    result = None
    if request.method == 'POST':
        try:
            # Build feature values in the correct order
            input_data = {}
            
            # Collect all feature values from the form
            for feat in feature_names:
                if feat in CATEGORICAL_FEATURES:
                    input_data[feat] = request.form.get(feat, CATEGORICAL_FEATURES[feat][0])
                elif feat in NUMERIC_FEATURES:
                    val = request.form.get(feat, NUMERIC_FEATURES[feat]['default'])
                    input_data[feat] = float(val)
                else:
                    input_data[feat] = float(request.form.get(feat, 0))
            
            # Create DataFrame with correct column order
            input_df = pd.DataFrame([input_data], columns=feature_names)
            
            # Apply label encoding for categorical features (same as training)
            for col, le in label_encoders.items():
                if col in input_df.columns:
                    # Handle unseen labels gracefully
                    val = input_df[col].iloc[0]
                    if val in le.classes_:
                        input_df[col] = le.transform(input_df[col])
                    else:
                        # Default to 0 if label not seen during training
                        input_df[col] = 0
            
            # Apply scaling
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            pred = svm_model.predict(input_scaled)[0]
            
            # Try to get probability
            if hasattr(svm_model, 'predict_proba'):
                prob = svm_model.predict_proba(input_scaled)[0]
                prob_attrition = round(prob[1] * 100, 1) if len(prob) > 1 else round(prob[0] * 100, 1)
            elif hasattr(svm_model, 'decision_function'):
                decision = svm_model.decision_function(input_scaled)[0]
                # Convert decision function to pseudo-probability using sigmoid
                prob_attrition = round(1 / (1 + np.exp(-decision)) * 100, 1)
            else:
                prob_attrition = 100.0 if pred == 1 else 0.0
            
            result = {
                'prediction': int(pred),
                'probability': prob_attrition,
                'label': 'Yes — Employee is likely to leave' if pred == 1 else 'No — Employee is likely to stay',
                'risk_level': 'High Risk' if prob_attrition >= 70 else ('Medium Risk' if prob_attrition >= 40 else 'Low Risk'),
                'risk_color': '#e74c3c' if prob_attrition >= 70 else ('#f39c12' if prob_attrition >= 40 else '#27ae60'),
            }
        except Exception as e:
            result = {'error': str(e)}
    
    return render_template('prediction.html',
                           categorical_features=CATEGORICAL_FEATURES,
                           numeric_features=NUMERIC_FEATURES,
                           scale_labels=SCALE_LABELS,
                           result=result)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
