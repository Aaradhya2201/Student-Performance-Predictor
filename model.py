import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
from preprocess import preprocess_data
import streamlit as st

MODEL_PATH = "student_performance_model.pkl"
ENCODERS_PATH = "label_encoders.pkl"

def train_model(df):
    df, encoders = preprocess_data(df)
    X = df.drop(columns=['Grade'])
    y = df['Grade']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    return model, encoders

def load_or_train_model():
    try:
        df = pd.read_csv('student_data.csv')
    except FileNotFoundError:
        st.error("Dataset 'student_data.csv' not found. Please upload the dataset.")
        return None, None
    
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
    else:
        model, encoders = train_model(df)
    return model, encoders

def predict_performance(model, encoders, input_data):
    input_df = pd.DataFrame([input_data])
    
    for col, le in encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = le.transform(input_df[col])
            except ValueError as e:
                st.warning(f"Unseen label in {col}: {input_df[col].values}. Mapping to default value.")
                unseen_labels = set(input_df[col]) - set(le.classes_)
                if unseen_labels:
                    input_df[col] = input_df[col].apply(lambda x: le.classes_[0] if x in unseen_labels else x)
                    input_df[col] = le.transform(input_df[col])
    
    expected_cols = model.feature_names_in_
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    
    prediction = model.predict(input_df[expected_cols])[0]
    return encoders['Grade'].inverse_transform([prediction])[0]
