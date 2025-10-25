# Flask Web Application for Gas Fire Detection System
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import numpy as np
from predict import GasFirePredictor
import os
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

app = Flask(__name__)
app.secret_key = 'gas_fire_detection_secret_key'

# Initialize predictor
try:
    predictor = GasFirePredictor()
    model_loaded = True
except:
    predictor = None
    model_loaded = False

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Single prediction page"""
    if not model_loaded:
        flash('Model not loaded. Please train the model first.', 'error')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        try:
            # Get form data
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            co2 = float(request.form['co2'])
            methane = float(request.form['methane'])
            smoke = float(request.form['smoke'])
            air_quality = float(request.form['air_quality'])
            room_type = request.form['room_type']
            
            # Validate inputs
            if not (-50 <= temperature <= 100):
                raise ValueError("Temperature should be between -50°C and 100°C")
            if not (0 <= humidity <= 100):
                raise ValueError("Humidity should be between 0% and 100%")
            if co2 < 0 or co2 > 10000:
                raise ValueError("CO2 should be between 0 and 10000 ppm")
            if methane < 0:
                raise ValueError("Methane should be non-negative")
            if smoke < 0:
                raise ValueError("Smoke density should be non-negative")
            if air_quality < 0:
                raise ValueError("Air quality index should be non-negative")
            
            # Make prediction
            prediction, probabilities = predictor.predict_from_values(
                temperature_c=temperature,
                humidity_percent=humidity,
                co2_ppm=co2,
                methane_ppm=methane,
                smoke_density=smoke,
                air_quality_index=air_quality,
                room_type=room_type
            )
            
            # Get interpretation
            result = predictor.get_risk_interpretation(prediction, probabilities)
            
            return render_template('predict.html', 
                                 result=result, 
                                 form_data=request.form,
                                 model_loaded=model_loaded)
            
        except ValueError as e:
            flash(str(e), 'error')
        except Exception as e:
            flash(f'Prediction error: {str(e)}', 'error')
    
    return render_template('predict.html', model_loaded=model_loaded)





def generate_plots():
    """Generate model evaluation plots"""
    try:
        # Load test data and model
        df = pd.read_csv('safehome_sensor_data.csv')
        with open('gas_fire_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        
        # Prepare data
        df = df.dropna()
        df['room_type_encoded'] = label_encoder.transform(df['room_type'])
        feature_cols = [col for col in df.columns if col not in ['status', 'room_type']]
        X = df[feature_cols]
        y = df['status']
        X_scaled = scaler.transform(X)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        plots = {}
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Danger', 'Safe', 'Warning'],
                   yticklabels=['Danger', 'Safe', 'Warning'])
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plots['confusion_matrix'] = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # ROC Curve
        plt.figure(figsize=(8, 6))
        y_test_bin = label_binarize(y_test, classes=['Danger', 'Safe', 'Warning'])
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(y_test_bin.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        colors = ['red', 'green', 'orange']
        labels = ['Danger', 'Safe', 'Warning']
        for i in range(y_test_bin.shape[1]):
            plt.plot(fpr[i], tpr[i], color=colors[i], 
                    label=f'{labels[i]} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plots['roc_curve'] = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Feature Importance
        plt.figure(figsize=(10, 6))
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plots['feature_importance'] = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return plots, accuracy, report, cm
        
    except Exception as e:
        print(f"Error generating plots: {e}")
        return {}, 0, {}, None

@app.route('/status')
def status():
    """System status page with test results"""
    status_info = {
        'model_loaded': model_loaded,
        'data_file_exists': os.path.exists('safehome_sensor_data.csv'),
        'model_file_exists': os.path.exists('gas_fire_model.pkl'),
    }
    
    plots = {}
    accuracy = 0
    report = {}
    confusion_matrix_data = None
    
    if model_loaded and status_info['data_file_exists']:
        try:
            plots, accuracy, report, confusion_matrix_data = generate_plots()
            status_info['model_working'] = True
        except:
            status_info['model_working'] = False
    else:
        status_info['model_working'] = False
    
    return render_template('status.html', 
                         status=status_info, 
                         plots=plots, 
                         accuracy=accuracy, 
                         report=report,
                         confusion_matrix=confusion_matrix_data)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)