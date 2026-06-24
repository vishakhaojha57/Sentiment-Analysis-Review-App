# AI-Powered Sentiment Analysis Dashboard

An interactive Streamlit-based web application that analyzes customer reviews using a Transformer-based Natural Language Processing (NLP) model.

The application leverages the **RoBERTa Sentiment Analysis Model** (`cardiffnlp/twitter-roberta-base-sentiment`) to classify user reviews into:

* 😊 Positive Sentiment
* 😐 Neutral Sentiment
* 😡 Negative Sentiment

The dashboard provides confidence scores, sentiment distribution visualizations, and an intuitive user interface for real-time sentiment analysis.

---

🚀 [Live Demo](https://sentiment-analysis-review-app-cjmqtnok5jcvfqvmkbawrg.streamlit.app/)

---

## Features

### Real-Time Sentiment Analysis

Users can enter any review or feedback and instantly receive sentiment predictions.

### 😊 Sentiment Classification

The application categorizes text into:

* Positive
* Neutral
* Negative

### 📈 Confidence Breakdown

Displays confidence percentages for each sentiment category using interactive progress bars.

### 🍩 Sentiment Distribution Visualization

Interactive donut chart showing probability distribution across sentiment classes.

### 📄 Review Summary

Provides:

* Review length
* Predicted sentiment
* Confidence score


## Machine Learning Model

### Model Used

**RoBERTa Transformer**



## 🛠️ Technologies Used

### Frontend

* Streamlit

### Machine Learning

* Hugging Face Transformers
* RoBERTa Sentiment Model

### Data Processing

* NumPy
* SciPy

### Visualization

* Plotly

### Programming Language

* Python

---

## 📂 Project Structure

```text
AI-Sentiment-Analyzer/
│
├── app.py
├── requirements.txt
├── README.md

```

## Run Application

```bash
streamlit run app.py
```

The application will open automatically in your browser.

---

## 📸 Sample Input

```text
This product exceeded my expectations.
The quality is amazing and delivery was very fast.
```

### Output

```text
😊 Positive Sentiment

Confidence Score: 98.76%



## 👩‍💻 Author

**Vishakha**
