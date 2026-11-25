"""
Loan Approval Predictor - Streamlit front end

Pipeline (same order the model was trained with):
    user inputs -> OneHotEncoder (education, self_employed)
                -> MinMax scaling (8 numeric columns)
                -> KNeighborsClassifier -> 0 / 1 -> LabelEncoder decode

Run:
    pip install -r requirements.txt
    streamlit run app.py

Keep model_prod.pkl in the same folder as this file.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model_prod.pkl"

CAT_COLS = ["education", "self_employed"]

# The pickle contains only {"model", "OHE", "LE"} - the scaler was NOT saved.
# The model was trained on MinMax-scaled values, so the scaler is rebuilt here
# from the (min, max) of each column in the training data.
# no_of_dependents was NOT scaled during training (raw values 0-5).
SCALE_RANGES = {
    "income_annum": (200_000, 9_900_000),
    "loan_amount": (300_000, 39_500_000),
    "loan_term": (2, 20),
    "cibil_score": (300, 900),
    "residential_assets_value": (-100_000, 29_100_000),
    "commercial_assets_value": (0, 19_000_000),
    "luxury_assets_value": (300_000, 39_200_000),
    "bank_asset_value": (0, 14_700_000),
}


@st.cache_resource
def load_artifacts():
    art = joblib.load(MODEL_PATH)  # file was saved with joblib
    return art["model"], art["OHE"], art["LE"]


def preprocess(raw: dict, model, ohe) -> pd.DataFrame:
    """Turn the form inputs into the exact 13-column frame the model expects."""
    # 1) one-hot encode the categorical inputs
    cat_df = pd.DataFrame([{c: raw[c] for c in CAT_COLS}])
    encoded = pd.DataFrame(ohe.transform(cat_df), columns=ohe.get_feature_names_out(CAT_COLS))

    # 2) MinMax-scale the numeric inputs: (x - min) / (max - min)
    scaled = {c: (raw[c] - lo) / (hi - lo) for c, (lo, hi) in SCALE_RANGES.items()}
    scaled["no_of_dependents"] = raw["no_of_dependents"]  # not scaled in training
    numeric = pd.DataFrame([scaled])

    # 3) combine and put the columns in the order the model was trained on
    return pd.concat([encoded, numeric], axis=1)[list(model.feature_names_in_)]


# ----- UI -----
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Loan Approval Predictor")
st.write("Enter the applicant's details and click **Predict**.")

if not MODEL_PATH.exists():
    st.error(f"`{MODEL_PATH.name}` not found. Place it in the same folder as app.py.")
    st.stop()

model, ohe, le = load_artifacts()

# Decode model output using the saved LabelEncoder (classes_ = [' Approved', ' Rejected'])
DECODE = {int(code): str(name).strip() for code, name in zip(le.transform(le.classes_), le.classes_)}

with st.form("loan_form"):
    st.subheader("Applicant details")
    left, right = st.columns(2)

    with left:
        education = st.selectbox("Education", ohe.categories_[0], format_func=str.strip)
        self_employed = st.selectbox("Self employed", ohe.categories_[1], format_func=str.strip)
        no_of_dependents = st.number_input("Number of dependents", min_value=0, max_value=5, value=2, step=1)
        income_annum = st.number_input("Annual income (₹)", min_value=200_000, max_value=9_900_000, value=5_100_000, step=100_000)
        loan_amount = st.number_input("Loan amount (₹)", min_value=300_000, max_value=39_500_000, value=14_500_000, step=100_000)
        loan_term = st.number_input("Loan term (years)", min_value=2, max_value=20, value=10, step=1)

    with right:
        cibil_score = st.number_input("CIBIL score", min_value=300, max_value=900, value=600, step=1)
        residential_assets_value = st.number_input("Residential assets value (₹)", min_value=0, max_value=29_100_000, value=5_600_000, step=100_000)
        commercial_assets_value = st.number_input("Commercial assets value (₹)", min_value=0, max_value=19_400_000, value=3_700_000, step=100_000)
        luxury_assets_value = st.number_input("Luxury assets value (₹)", min_value=300_000, max_value=39_200_000, value=14_600_000, step=100_000)
        bank_asset_value = st.number_input("Bank asset value (₹)", min_value=0, max_value=14_700_000, value=4_600_000, step=100_000)

    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    raw = {
        "education": education,
        "self_employed": self_employed,
        "no_of_dependents": no_of_dependents,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
    }

    X = preprocess(raw, model, ohe)
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    confidence = proba[list(model.classes_).index(pred)]
    label = DECODE[pred]

    st.divider()
    if label == "Approved":
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    c1, c2 = st.columns(2)
    c1.metric("Model output", f"{pred}  →  {label}")
    c2.metric("Confidence", f"{confidence:.0%}")

    with st.expander("Processed features sent to the model"):
        st.dataframe(X.T.rename(columns={0: "value"}))
