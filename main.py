# Complete ML Pipeline for Gas Fire Detection System
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=== Gas Fire Detection ML Pipeline ===")
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv('safehome_sensor_data.csv')
    print(f"Dataset loaded! Shape: {df.shape}")
    print(f"Target distribution:\n{df['status'].value_counts()}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    df = df.dropna()
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['room_type_encoded'] = label_encoder.fit_transform(df['room_type'])
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['status', 'room_type']]
    X = df[feature_cols]
    y = df['status']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    print("\nSaving model...")
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': X.columns.tolist()
    }
    
    with open('gas_fire_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved as 'gas_fire_model.pkl'")
    print(f"\n=== Pipeline Complete! Final Accuracy: {accuracy:.4f} ===")

if __name__ == "__main__":
    main()