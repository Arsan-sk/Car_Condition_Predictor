# 🚗 Car Condition Prediction

A machine learning application that predicts car condition/acceptability based on various car attributes using a k-Nearest Neighbors (KNN) classifier. The project features an interactive Streamlit web interface for easy prediction and model evaluation.

## 📋 Project Overview

**Car Condition Prediction** is a classification project that determines how suitable a car is based on factors like buying price, maintenance cost, number of doors, passenger capacity, luggage boot size, and safety features. The model classifies cars into four categories:
- **❌ Unacc** (Unacceptable)
- **⚠️ Acc** (Acceptable)
- **🚗 Good** (Good Condition)
- **🏎️ VGood** (Very Good)

## 🎯 What It Does

This application enables users to:
- **Input car specifications** through an interactive web form
- **Get instant predictions** on car condition/acceptability
- **View detailed metrics** including accuracy and model performance
- **Debug predictions** with detailed information about nearest neighbors
- **Visualize model evaluation** with confusion matrices and classification reports

## 🛠️ Features

✨ **Key Features:**
- **Interactive Streamlit UI** - User-friendly interface for car feature input
- **Real-time Predictions** - Instant classification of car conditions
- **Debug Mode** - View prediction internals including nearest neighbors distances
- **Model Metrics** - Comprehensive evaluation metrics and accuracy scores
- **Trained KNN Model** - Pre-trained k-nearest neighbors classifier
- **Emoji Indicators** - Visual feedback for prediction results

## 📦 Project Structure

```
Car_Condition_predict/
├── main.py                    # Streamlit application & prediction logic
├── Car_Condition.pickle       # Pre-trained KNN model
├── car.data                   # Training/evaluation dataset
├── .git/                      # Git repository
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Arsan-sk/Car_Condition_predict.git
   cd Car_Condition_predict
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install streamlit scikit-learn pandas numpy matplotlib seaborn
   ```

### Running the Application

Start the Streamlit application:
```bash
streamlit run main.py
```

The application will open in your default browser at `http://localhost:8501`

## 💻 How to Use

1. **Navigate to Prediction Page:**
   - Select car features from dropdown menus:
     - Buying Price (low, med, high, vhigh)
     - Maintenance Cost (low, med, high, vhigh)
     - Number of Doors (2, 3, 4, 5+)
     - Passenger Capacity (2, 4, more)
     - Luggage Boot Size (small, med, big)
     - Safety Level (low, med, high)

2. **Get Prediction:**
   - Click "🔍 Predict Now" button
   - View the car condition prediction with emoji indicators

3. **Debug Mode (Optional):**
   - Check "🔍 Debug Mode" to see prediction internals
   - View encoded input values and nearest neighbors information

4. **View Metrics:**
   - Access the Model Metrics page to see:
     - Overall accuracy percentage
     - Detailed classification reports
     - Confusion matrices

## 📊 Model Details

- **Algorithm:** k-Nearest Neighbors (KNN)
- **Training Dataset:** UCI Car Evaluation Dataset
- **Features:** 6 categorical attributes
- **Classes:** 4 (unacc, acc, good, vgood)
- **Model File:** `Car_Condition.pickle`

## 📂 File Descriptions

- **main.py** - Complete Streamlit application including:
  - Model loading and caching
  - Data preprocessing and label encoding
  - Prediction page with interactive form
  - Metrics evaluation page
  - Debug mode for model interpretation

- **Car_Condition.pickle** - Pre-trained KNN classifier model (binary format)

- **car.data** - CSV dataset containing car evaluation records with features and target class

## 🔧 Technologies Used

- **Python** - Programming language
- **Streamlit** - Web application framework
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **matplotlib & seaborn** - Data visualization

## 👤 Author

**Sheikh Mohammed Arsan**
- GitHub: [@Arsan-sk](https://github.com/Arsan-sk)
- A passionate machine learning developer focusing on building practical AI solutions and learning real-world ML applications.

## 📝 License

This project is open-source and available for educational and learning purposes.

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for any improvements or bug fixes.

## ❓ Troubleshooting

- **Model file not found:** Ensure `Car_Condition.pickle` is in the project directory
- **Data file not found:** Ensure `car.data` is in the project directory
- **Port already in use:** Streamlit will auto-select another port if 8501 is busy

## 📧 Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.

---

**Happy Predicting! 🚗✨**
