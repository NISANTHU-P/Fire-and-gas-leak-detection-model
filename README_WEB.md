# 🔥 Gas Fire Detection System - Web Application

A comprehensive machine learning web application for detecting gas leaks and fire hazards using sensor data. The system provides real-time risk assessment with three levels: **Safe**, **Warning**, and **Danger**.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install Flask pandas numpy scikit-learn matplotlib seaborn
```

### 2. Train the Model (if not already done)
```bash
python main.py
```

### 3. Start the Web Application
```bash
python run_web.py
```

### 4. Open Your Browser
Navigate to: `http://localhost:5000`

## 🌐 Web Interface Features

### Dashboard
- System overview and status
- Quick access to all features
- Health monitoring

### Single Prediction
- Interactive form for sensor readings
- Real-time validation
- Sample data presets
- Detailed results with recommendations

### Batch Processing
- CSV file upload
- Bulk prediction processing
- Results download
- Summary statistics

### System Status
- Model health monitoring
- Component status checks
- Troubleshooting guides

## 📊 API Endpoints

### Prediction API
```bash
POST /api/predict
Content-Type: application/json

{
  "temperature": 25.0,
  "humidity": 50.0,
  "co2": 400,
  "methane": 0.0,
  "smoke": 5.0,
  "air_quality": 100,
  "room_type": "living_room"
}
```

### Response
```json
{
  "success": true,
  "prediction": "Safe",
  "probabilities": {
    "Safe": 0.85,
    "Warning": 0.12,
    "Danger": 0.03
  },
  "result": {
    "risk_level": "SAFE",
    "confidence": "85.0%",
    "description": "Normal conditions detected...",
    "recommendations": [...]
  }
}
```

## 📁 File Structure

```
gas-fire-detection/
├── app.py                   # Flask web application
├── predict.py               # Prediction engine
├── main.py                  # Model training pipeline
├── run_web.py              # Web app launcher
├── test_simple.py          # System testing
├── templates/              # HTML templates
│   ├── base.html           # Base template
│   ├── index.html          # Dashboard
│   ├── predict.html        # Single prediction
│   ├── batch.html          # Batch processing
│   ├── status.html         # System status
│   └── about.html          # Documentation
├── static/                 # Static files (CSS, JS)
├── uploads/                # File upload directory
├── safehome_sensor_data.csv # Training dataset
├── gas_fire_model.pkl      # Trained model
└── requirements.txt        # Dependencies
```

## 🔧 Configuration

### Environment Variables
- `FLASK_ENV=development` (for development)
- `FLASK_DEBUG=1` (for debugging)

### File Upload Limits
- Maximum file size: 16MB
- Supported formats: CSV

## 🧪 Testing

### Test the Prediction System
```bash
python test_simple.py
```

### Test Sample Predictions
The web interface includes sample data for:
- Safe conditions (normal environment)
- Warning conditions (elevated risk)
- Danger conditions (high risk)

## 📈 Model Information

- **Algorithm**: Random Forest Classifier
- **Accuracy**: >95% on test data
- **Features**: 13 sensor and contextual parameters
- **Classes**: Safe, Warning, Danger

### Input Parameters
- Temperature (°C): -50 to 100
- Humidity (%): 0 to 100
- CO2 (ppm): 0 to 10,000
- Methane (ppm): ≥ 0
- Smoke Density: ≥ 0
- Air Quality Index: ≥ 0
- Room Type: kitchen, living_room, bedroom, basement, garage

## 🔒 Security Features

- Input validation and sanitization
- File upload restrictions
- Error handling and logging
- Safe file processing

## 🚨 Safety Guidelines

### Risk Level Interpretations:

**🟢 SAFE (Normal)**
- All sensors within normal ranges
- No immediate action required
- Continue regular monitoring

**🟡 WARNING (Elevated Risk)**
- Some concerning readings detected
- Increase monitoring frequency
- Consider investigation or ventilation
- Prepare for potential escalation

**🔴 DANGER (High Risk)**
- Critical readings detected
- **IMMEDIATE ACTION REQUIRED**
- Evacuate area if necessary
- Contact emergency services
- Do not ignore this warning

### Important Notes:
- This system is a **detection aid**, not a replacement for professional safety equipment
- Always follow local safety protocols and regulations
- Regular calibration of sensors is essential
- In case of doubt, prioritize safety and evacuate

## 🛠️ Troubleshooting

### Common Issues:

**Model not loaded**
```bash
python main.py  # Train the model first
```

**Web app won't start**
```bash
pip install -r requirements.txt  # Install dependencies
```

**CSV upload fails**
- Check file format (must be CSV)
- Verify required columns are present
- Ensure file size < 16MB

**Prediction errors**
- Validate input ranges
- Check room type values
- Ensure all required fields are filled

## 📞 Support

- Check the system status page for diagnostics
- Review error messages in the web interface
- Test with sample data first
- Verify model training completed successfully

---

**⚠️ SAFETY REMINDER**: This system is designed to assist in hazard detection but should not be the sole safety measure. Always follow proper safety protocols and use certified safety equipment in critical applications.