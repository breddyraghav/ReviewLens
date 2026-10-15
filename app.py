
import os
import joblib
import streamlit as st

# App configuration
st.set_page_config(
    page_title="ReviewLens | Sentiment Analyzer",
    page_icon="🔎",
    layout="centered"
)

# Load the saved model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "reviewlens_model.joblib"
)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

st.title("ReviewLens")
st.subheader("Understand the sentiment behind a review")
st.write(
    "Enter a review to see whether the trained NLP model "
    "classifies it as positive or negative."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Place reviewlens_model.joblib "
        "inside the models folder."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

review = st.text_area(
    "Enter your review",
    placeholder="Example: The story was engaging and the acting was excellent...",
    height=150
)

if st.button("Analyze Review", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        prediction = model.predict([review])[0]
        probabilities = model.predict_proba([review])[0]

        # The classifier was trained with labels:
        # 0 = negative, 1 = positive
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = max(probabilities) * 100

        st.divider()
        st.subheader("Analysis Result")

        if sentiment == "Positive":
            st.success(f"Sentiment: {sentiment}")
        else:
            st.error(f"Sentiment: {sentiment}")

        st.metric("Model confidence", f"{confidence:.1f}%")

        st.caption(
            "This model was trained on IMDb movie reviews. "
            "It may not perform reliably on product reviews. "
            "Confidence is a model probability, not a guarantee."
        )
