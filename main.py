# app.py
import streamlit as st
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize

# ------------------ Config ------------------
MODEL_PATH = "Car_Condition.pickle"
DATA_PATH = "car.data"  # only used in metrics/info/visuals if needed
LABELS = ["unacc", "acc", "good", "vgood"]
EMOJI_MAP = {
    "unacc": "❌",
    "acc": "⚠️",
    "good": "🚗",
    "vgood": "🏎️"
}

# ------------------ Helper Functions ------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

LABEL_MAPS = {
    "buying": {"low": 0, "med": 1, "high": 2, "vhigh": 3},
    "maint": {"low": 0, "med": 1, "high": 2, "vhigh": 3},
    "doors": {"2": 0, "3": 1, "4": 2, "5more": 3},
    "persons": {"2": 0, "4": 1, "more": 2},
    "lug_boot": {"small": 0, "med": 1, "big": 2},
    "safety": {"low": 0, "med": 1, "high": 2},
    "class": {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}
}

# ------------------ UI Functions ------------------
def user_input_form():
    st.subheader("🚗 Enter Car Features")
    buying = st.selectbox("Buying Price", list(LABEL_MAPS["buying"].keys()))
    maint = st.selectbox("Maintenance", list(LABEL_MAPS["maint"].keys()))
    doors = st.selectbox("Doors", list(LABEL_MAPS["doors"].keys()))
    persons = st.selectbox("Persons Capacity", list(LABEL_MAPS["persons"].keys()))
    lug_boot = st.selectbox("Luggage Boot Size", list(LABEL_MAPS["lug_boot"].keys()))
    safety = st.selectbox("Safety Level", list(LABEL_MAPS["safety"].keys()))
    debug = st.checkbox("🔍 Debug Mode (show prediction internals)")
    return [buying, maint, doors, persons, lug_boot, safety], debug

def encode_input(values):
    keys = list(LABEL_MAPS.keys())[:-1]  # exclude "class"
    return [LABEL_MAPS[key][val] for key, val in zip(keys, values)]

def display_prediction(pred, debug=False, encoded_input=None, model=None):
    label = LABELS[pred[0]]
    emoji = EMOJI_MAP[label]
    if label in ["unacc", "acc"]:
        st.error(f"Prediction: **{label.upper()}** {emoji}")
    else:
        st.success(f"Prediction: **{label.upper()}** {emoji}")

    if debug and encoded_input is not None and model is not None:
        st.write("### 🔎 Debug Info")
        st.write("**Encoded Input:**", encoded_input)
        try:
            distances, indices = model.kneighbors([encoded_input])
            st.write("**Nearest Neighbors Distances:**", distances[0])
            st.write("**Nearest Neighbors Indices:**", indices[0])
        except Exception as e:
            st.warning(f"Could not compute neighbors: {e}")

# ------------------ Pages ------------------
def prediction_page():
    st.title("🧠 Car Condition Classifier")
    inputs, debug = user_input_form()
    if st.button("🔍 Predict Now"):
        encoded = encode_input(inputs)
        model = load_model()
        prediction = model.predict([encoded])
        display_prediction(prediction, debug=debug, encoded_input=encoded, model=model)

def metrics_page():
    st.title("📊 Model Metrics")
    df = load_data()
    features = df.iloc[:, :-1].apply(lambda col: col.map(LABEL_MAPS[col.name]))
    target = df["class"].map(LABEL_MAPS["class"])

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.1, random_state=42)
    model = load_model()
    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{acc*100:.2f}%")

    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=LABELS, yticklabels=LABELS, cmap="Blues")
    st.pyplot(fig)

    st.write("### Classification Report")
    report = classification_report(y_test, y_pred, target_names=LABELS, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

def model_info_page():
    st.title("⚙️ Model Information")
    model = load_model()
    st.write("**Model Type:**", type(model).__name__)
    st.write("**Hyperparameters:** Pre-trained (k=5 assumed)")
    st.write("**Distance Metric:** Euclidean")
    st.write("**Training Size:** 90%")
    with st.expander("🔧 Advanced Settings"):
        st.code("""
# Model trained externally:
# model = KNeighborsClassifier(n_neighbors=5)
# model.fit(x_train, y_train)
        """)

def about_page():
    st.title("📖 About This App")
    st.markdown("""
### Project Description
This app uses a trained K-Nearest Neighbors (KNN) model to classify car conditions based on user input.

**Features:**
- Dynamic input fields for predicting car acceptability.
- Visual feedback with colored labels and emojis.
- Visual analytics on model performance.
- Debug mode to inspect prediction logic.

**Dataset:** UCI Car Evaluation Data  
**Model Output:** One of 4 categories – Unacceptable, Acceptable, Good, Very Good.

---
### About Me
I'm Arshan, a self-taught programmer passionate about full-stack development, automation, cybersecurity, hardware projects, and AI-powered web tools.  
Check out more on [GitHub](https://github.com/Arshan-sk).
    """)

# ------------------ Main App ------------------
def main():
    st.set_page_config(page_title="Car Classifier", layout="centered", initial_sidebar_state="expanded")

    pages = {
        "🔍 Predict": prediction_page,
        "📊 Metrics": metrics_page,
        "⚙️ Model Info": model_info_page,
        "📖 About": about_page
    }

    st.sidebar.title("🚗 Car Classifier App")
    choice = st.sidebar.radio("Navigate", list(pages.keys()))

    pages[choice]()

if __name__ == "__main__":
    main()
