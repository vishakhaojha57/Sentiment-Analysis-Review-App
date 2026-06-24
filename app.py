import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.sentiment-card {
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    background-color: #1E293B;
    border: 1px solid #334155;
    margin-bottom: 20px;
}

.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #374151;
}

.big-emoji {
    font-size: 70px;
}

</style>
""", unsafe_allow_html=True)



# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment"
    )

  

    return tokenizer, model

tokenizer, model = load_models()

# ---------------- FUNCTIONS ----------------
def roberta_scores(text):

    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    output = model(**encoded)

    scores = softmax(
        output.logits.detach().numpy()[0]
    )

    return {
        "Negative": float(scores[0]),
        "Neutral": float(scores[1]),
        "Positive": float(scores[2])
    }



def get_sentiment(scores):

    label = max(scores, key=scores.get)

    if label == "Positive":
        return "😊", "Positive Sentiment"

    elif label == "Negative":
        return "😡", "Negative Sentiment"

    else:
        return "😐", "Neutral Sentiment"

# ---------------- HEADER ----------------
st.title("AI-Powered Sentiment Analyzer")
st.caption(
    "Analyze customer reviews using AI-powered sentiment analysiss"
)

st.divider()

# ---------------- INPUT ----------------
review = st.text_area(
    "📝 Enter Review",
    height=180,
    placeholder="Type a review here..."
)

# ---------------- BUTTON ----------------
if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if review.strip() == "":
        st.warning("Please enter a review.")
        st.stop()

    # MODEL PREDICTIONS
    roberta_result = roberta_scores(review)


    emoji, final_sentiment = get_sentiment(roberta_result)

    confidence = max(roberta_result.values()) * 100

    # ---------------- RESULT CARD ----------------
    st.markdown(
        f"""
        <div class="sentiment-card">

            <div class="big-emoji">
                {emoji}
            </div>

            <h1>{final_sentiment}</h1>

            <h3>
                Confidence: {confidence:.2f}%
            </h3>

        </div>
        """,
        unsafe_allow_html=True
    )



    # ---------------- PROGRESS BARS ----------------

    st.subheader("📈 Confidence Breakdown")

    st.write(
        f"😊 Positive : {roberta_result['Positive']*100:.2f}%"
    )

    st.progress(
        int(roberta_result["Positive"] * 100)
    )

    st.write(
        f"😐 Neutral : {roberta_result['Neutral']*100:.2f}%"
    )

    st.progress(
        int(roberta_result["Neutral"] * 100)
    )

    st.write(
        f"😡 Negative : {roberta_result['Negative']*100:.2f}%"
    )

    st.progress(
        int(roberta_result["Negative"] * 100)
    )

    st.divider()

    # ---------------- DONUT CHART ----------------

    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(roberta_result.keys()),
                values=list(roberta_result.values()),
                hole=0.65
            )
        ]
    )

    fig.update_layout(
        title="Sentiment Distribution",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------- REVIEW SUMMARY ----------------
    
    st.subheader("📄 Review Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Review Length", f"{len(review.split())} words")
        
    with col2:
        st.metric("Sentiment", final_sentiment)
        
    with col3:
        st.metric("Confidence", f"{confidence:.2f}%")

    