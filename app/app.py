
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from pathlib import Path

# --------------------------------
# 1. PROJECT PATHS
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "disease_ann_model.keras"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
SCALER_PATH = MODEL_DIR / "feature_scaler.pkl"
FEATURES_PATH = MODEL_DIR / "symptom_columns.pkl"


# --------------------------------
# 2. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Medical Symptom Classifier",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical Symptom Classifier")

st.write(
    "Select symptoms to explore the output of "
    "our educational ANN classification model."
)

st.warning(
    "Educational project only. This application is not "
    "a medical diagnostic tool. Predictions are not "
    "medical advice. Please consult a qualified doctor "
    "for symptoms or health concerns."
)


# --------------------------------
# 3. LOAD TRAINED MODEL AND FILES
# --------------------------------

@st.cache_resource
def load_artifacts():

    model = tf.keras.models.load_model(MODEL_PATH)

    label_encoder = joblib.load(ENCODER_PATH)

    scaler = joblib.load(SCALER_PATH)

    symptom_columns = joblib.load(FEATURES_PATH)

    return model, label_encoder, scaler, symptom_columns


try:

    model, label_encoder, scaler, symptom_columns = (
        load_artifacts()
    )

except Exception as e:

    st.error(f"Could not load model files: {e}")

    st.stop()


# --------------------------------
# 4. SYMPTOM SELECTION
# --------------------------------

st.subheader("Select Your Symptoms")

selected_symptoms = st.multiselect(
    "Choose one or more symptoms:",
    options=symptom_columns,
    placeholder="Search and select symptoms..."
)


# --------------------------------
# 5. PREDICTION
# --------------------------------

if st.button("Predict", use_container_width=True):

    if not selected_symptoms:

        st.warning("Please select at least one symptom.")

    else:

        # Create a binary feature row in saved feature order
        input_data = pd.DataFrame(
            [[
                1 if symptom in selected_symptoms else 0
                for symptom in symptom_columns
            ]],
            columns=symptom_columns
        )

        # Apply the same scaler used during training
        scaled_input = scaler.transform(input_data)

        # Generate model scores
        predictions = model.predict(
            scaled_input,
            verbose=0
        )

        # Get top 3 model outputs
        top_indices = np.argsort(
            predictions[0]
        )[::-1][:3]

        top_scores = predictions[0][top_indices]

        top_diseases = label_encoder.inverse_transform(
            top_indices
        )

        st.subheader("Model Output")

        st.write(
            "The following are the top classes predicted "
            "by the educational ANN model:"
        )

        for rank, (disease, score) in enumerate(
            zip(top_diseases, top_scores),
            start=1
        ):

            st.markdown(
                f"**{rank}. {disease.title()}**"
            )

            st.progress(float(score))

            st.caption(
                f"Model score: {score:.4f}"
            )

        st.info(
            "These scores are outputs of the trained model. "
            "They are not clinically validated probabilities "
            "or evidence of a disease."
        )


# --------------------------------
# 6. FOOTER
# --------------------------------

st.divider()

st.caption(
    "ANN Medical Symptom Classification Project | "
    "Educational use only"
)
    