import os
import streamlit as st

st.set_page_config(
    page_title="ReviewLens | Sentiment Analyzer",
    page_icon="🔎",
    layout="centered"
)

# Check that required packages are installed
try:
    import joblib
except ModuleNotFoundError:
    st.error("The joblib package is missing.")
    st.info(
        "Open Streamlit Community Cloud, select Manage app, "
        "and reboot the app to reinstall its dependencies."
    )
    st.stop()

# Locate the trained model in the repository root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "reviewlens_model.joblib"
)

st.title("ReviewLens")
st.subheader("Understand the sentiment behind a review")

st.write(
    "Enter a movie review to predict whether its sentiment "
    "is positive or negative."
)

# Check whether the model file exists
if not os.path.isfile(MODEL_PATH):
    st.error(
        "Model file not found. Make sure "
        "reviewlens_model.joblib is uploaded beside app.py "
        "in your GitHub repository."
    )
    st.stop()

# Load the trained model
@st.cache_resource
def load_model(path):
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
except Exception as error:
    st.error("The trained model could not be loaded.")
    st.code(str(error))
    st.stop()

# Collect the user's review
review = st.text_area(
    "Enter your review",
    placeholder=(
        "Example: The story was engaging and "
        "the acting was excellent..."
    ),
    height=150
)

if st.button(
    "Analyze Review",
    type="primary",
    use_container_width=True
):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        try:
            prediction = model.predict([review])[0]
            probabilities = model.predict_proba([review])[0]

            sentiment = (
                "Positive" if prediction == 1 else "Negative"
            )
            confidence = max(probabilities) * 100

            st.divider()
            st.subheader("Analysis Result")

            if sentiment == "Positive":
                st.success(f"Sentiment: {sentiment}")
            else:
                st.error(f"Sentiment: {sentiment}")

            st.metric(
                "Model confidence",
                f"{confidence:.1f}%"
            )

            st.caption(
                "This model was trained on IMDb movie reviews. "
                "Results on product reviews may be less reliable. "
                "Confidence is not a guarantee of correctness."
            )

        except Exception as error:
            st.error("Could not analyze this review.")
            st.code(str(error))
