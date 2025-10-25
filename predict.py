# Prediction module for Gas Fire Detection System
import pandas as pd
import numpy as np
import pickle

class GasFirePredictor:
    def __init__(self, model_path='gas_fire_model.pkl'):
        """Initialize the predictor with saved model"""
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessors"""
        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_names = model_data['feature_names']
            print("Model loaded successfully!")
            
        except FileNotFoundError:
            print(f"Model file {self.model_path} not found. Please train the model first.")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict_from_values(self, temperature_c, humidity_percent, co2_ppm, methane_ppm, 
                           smoke_density, air_quality_index, room_type, heat_index=None,
                           gas_ratio=None, fire_risk_score=None, air_degradation=None,
                           hour=12, day_of_week=1):
        """Make prediction from individual sensor values"""
        
        # Calculate derived features if not provided
        if heat_index is None:
            heat_index = temperature_c + 2.5  # Simple approximation
        
        if gas_ratio is None:
            gas_ratio = methane_ppm / (co2_ppm + 1) if co2_ppm > 0 else 0
        
        if fire_risk_score is None:
            fire_risk_score = (temperature_c * 0.1 + smoke_density * 0.05 + methane_ppm * 0.02)
        
        if air_degradation is None:
            air_degradation = air_quality_index + smoke_density * 0.5
        
        # Encode room type
        try:
            room_type_encoded = self.label_encoder.transform([room_type])[0]
        except ValueError:
            print(f"Unknown room type: {room_type}. Using 'living_room' as default.")
            room_type_encoded = self.label_encoder.transform(['living_room'])[0]
        
        # Create feature array
        features = np.array([[
            temperature_c, humidity_percent, co2_ppm, methane_ppm, smoke_density,
            air_quality_index, room_type_encoded, heat_index, gas_ratio,
            fire_risk_score, air_degradation, hour, day_of_week
        ]])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Get class probabilities
        classes = self.model.classes_
        prob_dict = dict(zip(classes, probabilities))
        
        return prediction, prob_dict
    
    def predict_from_csv(self, csv_path, output_path=None):
        """Make predictions from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} records from {csv_path}")
            
            # Prepare features (assuming same structure as training data)
            required_cols = ['temperature_c', 'humidity_percent', 'co2_ppm', 'methane_ppm',
                           'smoke_density', 'air_quality_index', 'room_type']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Missing required columns: {missing_cols}")
                return None
            
            predictions = []
            probabilities = []
            
            for idx, row in df.iterrows():
                try:
                    pred, prob = self.predict_from_values(
                        temperature_c=row['temperature_c'],
                        humidity_percent=row['humidity_percent'],
                        co2_ppm=row['co2_ppm'],
                        methane_ppm=row['methane_ppm'],
                        smoke_density=row['smoke_density'],
                        air_quality_index=row['air_quality_index'],
                        room_type=row['room_type'],
                        heat_index=row.get('heat_index'),
                        gas_ratio=row.get('gas_ratio'),
                        fire_risk_score=row.get('fire_risk_score'),
                        air_degradation=row.get('air_degradation'),
                        hour=row.get('hour', 12),
                        day_of_week=row.get('day_of_week', 1)
                    )
                    predictions.append(pred)
                    probabilities.append(prob)
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
                    predictions.append('Error')
                    probabilities.append({})
            
            # Add predictions to dataframe
            df['predicted_status'] = predictions
            df['danger_prob'] = [p.get('Danger', 0) for p in probabilities]
            df['safe_prob'] = [p.get('Safe', 0) for p in probabilities]
            df['warning_prob'] = [p.get('Warning', 0) for p in probabilities]
            
            # Save results
            if output_path:
                df.to_csv(output_path, index=False)
                print(f"Results saved to {output_path}")
            
            return df
            
        except Exception as e:
            print(f"Error processing CSV: {e}")
            return None
    
    def get_risk_interpretation(self, prediction, probabilities):
        """Get human-readable risk interpretation"""
        interpretations = {
            'Safe': {
                'level': 'SAFE',
                'description': 'Normal conditions detected. No immediate action required.',
                'recommendations': [
                    'Continue regular monitoring',
                    'Maintain current safety protocols',
                    'Check sensors periodically'
                ]
            },
            'Warning': {
                'level': 'WARNING',
                'description': 'Elevated risk detected. Increased monitoring recommended.',
                'recommendations': [
                    'Increase monitoring frequency',
                    'Check ventilation systems',
                    'Investigate potential sources',
                    'Prepare for possible escalation'
                ]
            },
            'Danger': {
                'level': 'DANGER',
                'description': 'HIGH RISK! Immediate action required.',
                'recommendations': [
                    'EVACUATE AREA IMMEDIATELY',
                    'Contact emergency services',
                    'Do not ignore this warning',
                    'Ensure all personnel are safe'
                ]
            }
        }
        
        confidence = max(probabilities.values()) * 100
        
        result = {
            'prediction': prediction,
            'confidence': f"{confidence:.1f}%",
            'risk_level': interpretations[prediction]['level'],
            'description': interpretations[prediction]['description'],
            'recommendations': interpretations[prediction]['recommendations'],
            'probabilities': {k: f"{v*100:.1f}%" for k, v in probabilities.items()}
        }
        
        return result