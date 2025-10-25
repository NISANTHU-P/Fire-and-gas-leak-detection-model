# 🔥 Gas Fire Detection System

A comprehensive machine learning system for detecting gas leaks and fire hazards using sensor data. The system provides real-time risk assessment with three levels: **Safe**, **Warning**, and **Danger**.

## 🚀 Features

- **Machine Learning Model**: Random Forest classifier trained on sensor data
- **Real-time Predictions**: Instant risk assessment from sensor readings
- **Multiple Interfaces**: GUI application, command-line tool, and batch processing
- **Comprehensive Analysis**: Feature importance, confusion matrix, and detailed reporting
- **Easy Deployment**: Complete pipeline with preprocessing, training, and prediction

## 📊 Dataset

The system uses sensor data with the following features:
- **Environmental**: Temperature, Humidity, CO2 levels
- **Gas Detection**: Methane levels, Gas ratios
- **Air Quality**: Smoke density, Air quality index, Air degradation
- **Risk Metrics**: Fire risk score, Heat index
- **Context**: Room type, Time (hour, day of week)

**Target Classes**:
- 🟢 **Safe**: Normal conditions, no immediate hazard
- 🟡 **Warning**: Elevated risk, monitoring recommended
- 🔴 **Danger**: High risk, immediate action required

## 🛠️ Installation

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Quick Setup
1. Clone or download the project files
2. Ensure `safehome_sensor_data.csv` is in the project directory
3. Run the pipeline:
```bash
python run_pipeline.py
```

## 🎯 Usage

### Option 1: Interactive Menu (Recommended)
```bash
python run_pipeline.py
```
This launches an interactive menu with options for:
- Training the model
- Launching the GUI
- Command-line predictions
- System status checks

### Option 2: Direct Commands

#### Train the Model
```bash
python run_pipeline.py --mode train
```

#### Launch GUI Application
```bash
python run_pipeline.py --mode ui
```

#### Command Line Prediction
```bash
python predict.py --interactive
```

#### Batch Prediction from CSV
```bash
python predict.py --csv input_data.csv --output results.csv
```

#### Single Prediction with Parameters
```bash
python predict.py --temperature 45.5 --methane 25.3 --smoke 35.2 --room_type kitchen
```

## 🖥️ GUI Application

The GUI provides an intuitive interface for making predictions:

### Features:
- **Input Fields**: Easy-to-use forms for all sensor parameters
- **Real-time Validation**: Input range checking and validation
- **Visual Results**: Color-coded risk levels and probability breakdowns
- **Risk Interpretation**: Detailed explanations and recommendations
- **Batch Processing**: Load and process multiple readings

### Usage:
1. Run `python prediction_ui.py` or use the menu system
2. Enter sensor readings in the input fields
3. Click "Predict Risk Level"
4. View detailed results and recommendations

## 📈 Model Performance

The Random Forest model provides:
- **High Accuracy**: Typically >95% on test data
- **Feature Importance**: Identifies key risk factors
- **Probability Estimates**: Confidence levels for predictions
- **Robust Performance**: Handles missing values and outliers

### Key Features by Importance:
1. Fire Risk Score
2. Temperature
3. Methane Levels
4. Smoke Density
5. Air Quality Index

## 🔧 API Reference

### GasFirePredictor Class

```python
from predict import GasFirePredictor

# Initialize predictor
predictor = GasFirePredictor('gas_fire_model.pkl')

# Make prediction
prediction, probabilities = predictor.predict_from_values(
    temperature_c=35.5,
    humidity_percent=65.0,
    methane_ppm=15.2,
    smoke_density=25.8,
    room_type='kitchen'
)

print(f"Risk Level: {prediction}")
print(f"Confidence: {max(probabilities)*100:.1f}%")
```

### Batch Processing

```python
# Process CSV file
results_df = predictor.predict_from_csv('sensor_data.csv', 'predictions.csv')
print(results_df['predicted_status'].value_counts())
```

## 📁 File Structure

```
gas-fire-detection/
├── main.py                 # Complete training pipeline
├── prediction_ui.py        # GUI application
├── predict.py              # Command-line prediction tool
├── run_pipeline.py         # Interactive pipeline runner
├── preprocess.py           # Data preprocessing (legacy)
├── train.py                # Model training (legacy)
├── test.py                 # Model evaluation (legacy)
├── safehome_sensor_data.csv # Training dataset
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── gas_fire_model.pkl     # Trained model (generated)
```

## 🚨 Safety Guidelines

### Risk Level Interpretations:

**🟢 Safe (Normal)**
- All sensors within normal ranges
- No immediate action required
- Continue regular monitoring

**🟡 Warning (Elevated Risk)**
- Some concerning readings detected
- Increase monitoring frequency
- Consider investigation or ventilation
- Prepare for potential escalation

**🔴 Danger (High Risk)**
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

## 🔬 Technical Details

### Model Architecture:
- **Algorithm**: Random Forest Classifier
- **Features**: 13 sensor and contextual features
- **Classes**: 3 risk levels (Safe, Warning, Danger)
- **Preprocessing**: StandardScaler normalization, Label encoding

### Performance Metrics:
- **Accuracy**: >95% on test set
- **Precision/Recall**: Balanced across all classes
- **Feature Importance**: Quantified contribution of each sensor

### Data Processing:
- **Missing Values**: Handled via imputation or removal
- **Outliers**: Robust model handles extreme values
- **Scaling**: StandardScaler for numerical features
- **Encoding**: Label encoding for categorical variables

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is provided for educational and research purposes. Please ensure compliance with local safety regulations when deploying in production environments.

## 🆘 Troubleshooting

### Common Issues:

**Model file not found**
```bash
# Train the model first
python run_pipeline.py --mode train
```

**Missing dependencies**
```bash
# Install required packages
pip install -r requirements.txt
```

**CSV format errors**
```bash
# Ensure CSV has correct column names and data types
# Check the sample data format in safehome_sensor_data.csv
```

**GUI not launching**
```bash
# Check tkinter installation
python -c "import tkinter; print('tkinter OK')"
```

### Getting Help:

1. Check the system status: `python run_pipeline.py --mode status`
2. Verify requirements: Use the interactive menu option 5
3. Review error messages carefully
4. Ensure data file is in the correct location

## 📞 Support

For technical support or questions:
- Check the troubleshooting section above
- Review the code comments for implementation details
- Test with the provided sample data first

---

**⚠️ SAFETY REMINDER**: This system is designed to assist in hazard detection but should not be the sole safety measure. Always follow proper safety protocols and use certified safety equipment in critical applications.