#!/usr/bin/env python3
"""
Simple test script for Gas Fire Detection System
"""
from predict import GasFirePredictor

def test_predictions():
    """Test the prediction system with sample data"""
    print("Testing Gas Fire Detection System")
    print("=" * 40)
    
    try:
        # Initialize predictor
        predictor = GasFirePredictor()
        print("Model loaded successfully!")
        
        # Test safe conditions
        print("\nTest 1: Safe Conditions")
        print("-" * 30)
        prediction, probabilities = predictor.predict_from_values(
            temperature_c=22.5,
            humidity_percent=45.0,
            co2_ppm=400,
            methane_ppm=0.0,
            smoke_density=2.0,
            air_quality_index=50,
            room_type='living_room'
        )
        result = predictor.get_risk_interpretation(prediction, probabilities)
        print(f"Prediction: {result['risk_level']}")
        print(f"Confidence: {result['confidence']}")
        
        # Test warning conditions
        print("\nTest 2: Warning Conditions")
        print("-" * 30)
        prediction, probabilities = predictor.predict_from_values(
            temperature_c=35.0,
            humidity_percent=70.0,
            co2_ppm=1200,
            methane_ppm=15.0,
            smoke_density=25.0,
            air_quality_index=200,
            room_type='kitchen'
        )
        result = predictor.get_risk_interpretation(prediction, probabilities)
        print(f"Prediction: {result['risk_level']}")
        print(f"Confidence: {result['confidence']}")
        
        # Test danger conditions
        print("\nTest 3: Danger Conditions")
        print("-" * 30)
        prediction, probabilities = predictor.predict_from_values(
            temperature_c=55.0,
            humidity_percent=80.0,
            co2_ppm=3000,
            methane_ppm=80.0,
            smoke_density=90.0,
            air_quality_index=400,
            room_type='basement'
        )
        result = predictor.get_risk_interpretation(prediction, probabilities)
        print(f"Prediction: {result['risk_level']}")
        print(f"Confidence: {result['confidence']}")
        
        print("\nAll tests completed successfully!")
        print("You can now start the web application with: python run_web.py")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Make sure the model is trained: python main.py")

if __name__ == "__main__":
    test_predictions()