"""Streamlit app: score a single transaction for fraud.

Run the notebook first (it writes fraud_detection_pipeline.pkl and
model_metadata.json), then:  streamlit run streamlit_app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
MODEL_PATH = BASE / "fraud_detection_pipeline.pkl"
META_PATH = BASE / "model_metadata.json"

FRAUD_TYPES = ["TRANSFER", "CASH_OUT"]

EXAMPLES = {
    "Suspicious: TRANSFER draining an account": dict(
        type="TRANSFER", amount=181.0, old_o=181.0, new_o=0.0, old_d=0.0, new_d=0.0),
    "Typical: ordinary CASH_OUT": dict(
        type="CASH_OUT", amount=5000.0, old_o=120000.0, new_o=115000.0,
        old_d=30000.0, new_d=35000.0),
}
DEFAULT_EXAMPLE = "Suspicious: TRANSFER draining an account"

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="centered")


# --------------------------------------------------------------- model loading
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    return model, meta


if not MODEL_PATH.exists():
    st.error("`fraud_detection_pipeline.pkl` not found. Run `fraud_detection.ipynb` "
             "first to train and save the model.")
    st.stop()

model, meta = load_artifacts()
INPUT_COLUMNS = meta.get("input_columns")
TRAINED_THRESHOLD = float(meta.get("threshold", 0.5))


# --------------------------------------------------------------- feature logic
def add_features(row: pd.DataFrame) -> pd.DataFrame:
    """Must match add_features() in fraud_detection.ipynb (Section 9)."""
    out = row.copy()
    out["balanceDiffOrig"] = out["oldbalanceOrg"] - out["newbalanceOrig"]
    out["balanceDiffDest"] = out["newbalanceDest"] - out["oldbalanceDest"]
    out["errorBalanceOrig"] = out["newbalanceOrig"] + out["amount"] - out["oldbalanceOrg"]
    out["errorBalanceDest"] = out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]
    out["origEmptied"] = ((out["oldbalanceOrg"] > 0) & (out["newbalanceOrig"] == 0)).astype(int)
    out["destUntouched"] = ((out["oldbalanceDest"] == 0) & (out["newbalanceDest"] == 0)
                            & (out["amount"] > 0)).astype(int)
    out["amountToBalance"] = (out["amount"] / (out["oldbalanceOrg"] + 1)).clip(0, 10)
    return out


# --------------------------------------------------------------- session state
for key, value in EXAMPLES[DEFAULT_EXAMPLE].items():
    st.session_state.setdefault(key, value)


def load_example(name: str):
    for key, value in EXAMPLES[name].items():
        st.session_state[key] = value


# --------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About")
    st.write("Scores a **TRANSFER** or **CASH_OUT** transaction with a class-weighted "
             "Logistic Regression pipeline trained on the PaySim dataset.")
    threshold = st.slider("Decision threshold", 0.05, 0.99, min(max(TRAINED_THRESHOLD, 0.05), 0.99), 0.01,
                          help="Alert when fraud probability is at or above this value. "
                               "Default is the value tuned on validation data in the notebook.")
    m = meta.get("test_metrics_tuned_threshold")
    if m:
        st.subheader("Held-out test metrics")
        st.caption(f"at the tuned threshold ({TRAINED_THRESHOLD:.2f})")
        c1, c2 = st.columns(2)
        c1.metric("Recall", f"{m['recall']:.1%}")
        c2.metric("Precision", f"{m['precision']:.1%}")
        st.metric("PR-AUC", f"{m['pr_auc']:.3f}")
    st.caption("Probabilities come from a class-weighted model, so they are inflated "
               "relative to real-world fraud rates. Use them to rank risk, not as "
               "literal probabilities.")

# --------------------------------------------------------------- main page
st.title("🛡️ Transaction Fraud Detection")
st.write("Enter the transaction details, or load an example, then score it.")

st.subheader("Examples")
cols = st.columns(len(EXAMPLES))
for col, name in zip(cols, EXAMPLES):
    col.button(name, on_click=load_example, args=(name,), use_container_width=True)

with st.form("transaction_form"):
    tx_type = st.selectbox("Transaction type", FRAUD_TYPES, key="type")
    amount = st.number_input("Amount", min_value=0.0, step=100.0, key="amount")
    left, right = st.columns(2)
    with left:
        st.markdown("**Sender**")
        old_o = st.number_input("Balance before", min_value=0.0, key="old_o")
        new_o = st.number_input("Balance after", min_value=0.0, key="new_o")
    with right:
        st.markdown("**Receiver**")
        old_d = st.number_input("Balance before ", min_value=0.0, key="old_d")
        new_d = st.number_input("Balance after ", min_value=0.0, key="new_d")
    submitted = st.form_submit_button("Score transaction", type="primary")

if submitted:
    raw = pd.DataFrame([{
        "type": tx_type, "amount": amount,
        "oldbalanceOrg": old_o, "newbalanceOrig": new_o,
        "oldbalanceDest": old_d, "newbalanceDest": new_d,
    }])
    features = add_features(raw)
    if INPUT_COLUMNS:
        features = features[INPUT_COLUMNS]

    proba = float(model.predict_proba(features)[0, 1])

    st.divider()
    if proba >= threshold:
        st.error(f"🚨 **FRAUD ALERT**: fraud score {proba:.1%} (threshold {threshold:.0%})")
    else:
        st.success(f"✅ **Looks legitimate**: fraud score {proba:.1%} (threshold {threshold:.0%})")
    st.progress(min(max(proba, 0.0), 1.0))

    notes = []
    if amount > old_o:
        notes.append("The amount is larger than the sender's balance before the transaction.")
    if old_o > 0 and new_o == 0:
        notes.append("The sender's account is completely emptied.")
    if old_d == 0 and new_d == 0 and amount > 0:
        notes.append("The receiver's balance does not change at all.")
    if abs(new_o + amount - old_o) > 0.01:
        notes.append("The sender's balances do not reconcile with the amount.")
    if abs(old_d + amount - new_d) > 0.01:
        notes.append("The receiver's balances do not reconcile with the amount.")
    if notes:
        st.subheader("Risk signals in this transaction")
        for note in notes:
            st.write(f"- {note}")

    with st.expander("Engineered features sent to the model"):
        st.dataframe(add_features(raw).drop(columns=["type"]).T.rename(columns={0: "value"}))
